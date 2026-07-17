from __future__ import annotations

import contextlib
import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str = "Qwen/Qwen3-8B"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def encode_prompt(tokenizer, prompt: str, max_input_tokens: int):
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=True)
    prompt_len = int(encoded["input_ids"].shape[1])
    if prompt_len > max_input_tokens:
        raise ValueError(f"Prompt has {prompt_len} tokens, exceeds max_input_tokens={max_input_tokens}")
    return encoded, prompt_len


def to_device(batch: dict[str, Any], model) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    return {k: v.to(device) for k, v in batch.items()}


def press_context(press, model):
    return contextlib.nullcontext() if press is None else press(model)

def quantized_cache_nbits(compression_ratio: float, backend: str | None = None) -> int:
    explicit = os.environ.get("KVBENCH_QUANTIZED_CACHE_NBITS")
    if explicit:
        return int(explicit)
    if compression_ratio >= 0.75:
        return 2
    if compression_ratio >= 0.50:
        return 4
    if backend == "quanto":
        return 4
    return 8


def quantized_cache_settings(compression_ratio: float) -> dict[str, Any]:
    backend = os.environ.get("KVBENCH_QUANTIZED_CACHE_BACKEND", "hqq")
    nbits = quantized_cache_nbits(compression_ratio, backend)
    q_group_size = int(os.environ.get("KVBENCH_QUANTIZED_CACHE_GROUP_SIZE", "64"))
    residual_length = int(os.environ.get("KVBENCH_QUANTIZED_CACHE_RESIDUAL_LENGTH", "128"))
    return {
        "backend": backend,
        "nbits": nbits,
        "q_group_size": q_group_size,
        "residual_length": residual_length,
        "target_compression_ratio": compression_ratio,
        "idealized_bit_retained_budget_vs_bf16": nbits / 16.0,
        "idealized_bit_compression_ratio_vs_bf16": 1.0 - (nbits / 16.0),
    }


def create_quantized_cache(model, compression_ratio: float):
    from transformers import QuantizedCache

    settings = quantized_cache_settings(compression_ratio)
    return QuantizedCache(
        backend=settings["backend"],
        config=model.config,
        nbits=settings["nbits"],
        q_group_size=settings["q_group_size"],
        residual_length=settings["residual_length"],
    )


def prefill_kwargs(model, cache_factory=None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"use_cache": True}
    logits_to_keep = os.environ.get("KVBENCH_PREFILL_LOGITS_TO_KEEP", "1")
    if logits_to_keep:
        kwargs["logits_to_keep"] = int(logits_to_keep)
    if cache_factory is not None:
        kwargs["past_key_values"] = cache_factory(model)
    return kwargs

def model_step(model, input_ids, past_key_values, logical_position: int):
    attention_mask = torch.ones((1, logical_position + 1), dtype=torch.long, device=input_ids.device)
    position_ids = torch.tensor([[logical_position]], dtype=torch.long, device=input_ids.device)
    cache_position = torch.tensor([logical_position], dtype=torch.long, device=input_ids.device)
    base = {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": True}
    variants = (
        {**base, "attention_mask": attention_mask, "position_ids": position_ids, "cache_position": cache_position},
        {**base, "attention_mask": attention_mask, "position_ids": position_ids},
        base,
    )
    last_exc = None
    for kwargs in variants:
        try:
            return model(**kwargs)
        except TypeError as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, max_input_tokens: int, press=None, cache_factory=None) -> dict[str, Any]:
    encoded, prompt_len = encode_prompt(tokenizer, prompt, max_input_tokens)
    inputs = to_device(encoded, model)
    generated: list[int] = []
    with torch.no_grad():
        with press_context(press, model):
            out = model(**inputs, **prefill_kwargs(model, cache_factory))
            logits = out.logits[:, -1, :]
            past = out.past_key_values
            next_token = torch.argmax(logits, dim=-1)
            token_id = int(next_token.item())
            generated.append(token_id)
            eos_id = tokenizer.eos_token_id
            for index in range(1, max_new_tokens):
                if eos_id is not None and token_id == int(eos_id):
                    break
                out = model_step(model, next_token.reshape(1, 1), past, prompt_len + index - 1)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1)
                token_id = int(next_token.item())
                generated.append(token_id)
    text = tokenizer.decode(torch.tensor(generated, dtype=torch.long), skip_special_tokens=True).strip()
    return {"text": text, "prompt_tokens": prompt_len, "output_tokens": len(generated)}


def answer_nll(model, tokenizer, prompt: str, answer: str, max_input_tokens: int, press=None, cache_factory=None) -> float | None:
    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids
    if answer_ids.numel() == 0:
        return None
    encoded, prompt_len = encode_prompt(tokenizer, prompt, max_input_tokens)
    inputs = to_device(encoded, model)
    answer_ids = answer_ids.to(inputs["input_ids"].device)
    losses: list[float] = []
    with torch.no_grad():
        with press_context(press, model):
            out = model(**inputs, **prefill_kwargs(model, cache_factory))
            logits = out.logits[:, -1, :]
            past = out.past_key_values
            for i in range(answer_ids.shape[1]):
                target = answer_ids[:, i]
                losses.append(float(torch.nn.functional.cross_entropy(logits.float(), target, reduction="none").item()))
                if i == answer_ids.shape[1] - 1:
                    break
                out = model_step(model, target.reshape(1, 1), past, prompt_len + i)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
    return sum(losses) / len(losses)
