from __future__ import annotations

import inspect
from types import MethodType
from typing import Any

import torch


class RetainedTracker:
    def __init__(self) -> None:
        self.records: dict[str, Any] = {}

    def record(self, layer_idx: int, indices: torch.Tensor) -> None:
        arr = indices.detach().cpu()
        if arr.ndim == 3:
            arr = arr[0]
        self.records[str(layer_idx)] = arr.tolist()

    def flattened_positions(self) -> set[int]:
        positions: set[int] = set()
        for heads in self.records.values():
            if not isinstance(heads, list):
                continue
            for head in heads:
                if isinstance(head, list):
                    positions.update(int(x) for x in head)
        return positions


def _instantiate(cls, **kwargs):
    sig = inspect.signature(cls)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**accepted)


def create_press(method_name: str, compression_ratio: float):
    import kvpress

    if method_name == "ChunkKVPress_Knorm":
        base = _instantiate(kvpress.KnormPress, compression_ratio=compression_ratio)
        cls = kvpress.ChunkKVPress
        for kwargs in (
            {"base_press": base},
            {"press": base},
            {"base_press": base, "chunk_length": 20},
            {"press": base, "chunk_length": 20},
        ):
            try:
                return _instantiate(cls, **kwargs)
            except Exception:
                continue
        return cls(base)

    if method_name == "ThinKPress":
        if compression_ratio > 0.5:
            raise ValueError("ThinKPress cannot realize overall KV compression_ratio > 0.5 because it prunes key channels but not values")
        return _instantiate(kvpress.ThinKPress, key_channel_compression_ratio=2 * compression_ratio, window_size=32)

    if method_name == "AdaKVPress":
        base = _instantiate(kvpress.SnapKVPress, compression_ratio=compression_ratio, window_size=64, kernel_size=5)
        return _instantiate(kvpress.AdaKVPress, press=base, alpha_safeguard=0.2)

    cls = getattr(kvpress, method_name)
    defaults = {
        "compression_ratio": compression_ratio,
        "n_sink": 4,
        "window_size": 64,
        "kernel_size": 5,
    }
    return _instantiate(cls, **defaults)


def attach_retained_tracker(press) -> RetainedTracker:
    tracker = RetainedTracker()
    if press is None:
        return tracker

    def scorer_compress_with_recording(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        k_len = keys.shape[2]
        n_kept = max(1, int(k_len * (1 - self.compression_ratio)))
        indices = scores.topk(n_kept, dim=-1).indices
        layer_idx = int(getattr(module, "layer_idx", len(tracker.records)))
        tracker.record(layer_idx, indices)
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        return keys.gather(2, gather_indices).contiguous(), values.gather(2, gather_indices).contiguous()

    def chunkkv_compress_with_recording(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.press.compression_ratio == 0:
            return keys, values
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        kv_len = keys.shape[2]
        num_complete_chunks = kv_len // self.chunk_length
        remaining_tokens = kv_len % self.chunk_length
        layer_idx = int(getattr(module, "layer_idx", len(tracker.records)))
        if num_complete_chunks == 0:
            n_kept = max(1, int(kv_len * (1 - self.press.compression_ratio)))
            indices = scores.topk(n_kept, dim=-1).indices
            tracker.record(layer_idx, indices)
            gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
            return keys.gather(2, gather_indices).contiguous(), values.gather(2, gather_indices).contiguous()
        main_scores = scores[..., : num_complete_chunks * self.chunk_length]
        main_chunk_scores = main_scores.sum(dim=1).view(-1, num_complete_chunks, self.chunk_length).mean(dim=-1)
        if remaining_tokens > 0:
            remaining_scores = scores[..., -remaining_tokens:]
            remaining_chunk_score = remaining_scores.sum(dim=1).mean(dim=-1, keepdim=True)
            chunk_scores = torch.cat([main_chunk_scores, remaining_chunk_score], dim=-1)
        else:
            chunk_scores = main_chunk_scores
        n_chunks = num_complete_chunks + int(remaining_tokens > 0)
        n_chunks_kept = max(1, int(n_chunks * (1 - self.press.compression_ratio)))
        top_chunks = chunk_scores.topk(n_chunks_kept, dim=-1).indices[0]
        selected_chunks = []
        for chunk_idx in top_chunks:
            chunk_idx_int = int(chunk_idx.item())
            if chunk_idx_int < num_complete_chunks:
                start_idx = chunk_idx_int * self.chunk_length
                selected_chunks.append(torch.arange(start_idx, start_idx + self.chunk_length, device=keys.device))
            else:
                selected_chunks.append(torch.arange(num_complete_chunks * self.chunk_length, kv_len, device=keys.device))
        flat_indices = torch.cat(selected_chunks).sort()[0]
        tracker.record(layer_idx, flat_indices.view(1, 1, -1).expand(keys.shape[0], keys.shape[1], -1))
        gather_indices = flat_indices.view(1, 1, -1, 1).expand(keys.shape[0], keys.shape[1], -1, module.head_dim)
        return keys.gather(2, gather_indices).contiguous(), values.gather(2, gather_indices).contiguous()

    def adakv_compress_with_recording(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs).clone()
        bsz, num_key_value_heads, k_len = scores.shape
        n_kept = int(k_len * (1 - self.compression_ratio))
        n_safe = int(n_kept * self.alpha_safeguard)
        if n_safe > 0:
            top_indices = torch.topk(scores, n_safe, dim=-1).indices
            scores.scatter_(-1, top_indices, torch.finfo(scores.dtype).max)
        n_pruned = num_key_value_heads * (k_len - n_kept)
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()
        batch_indices = torch.arange(bsz, device=indices.device).repeat_interleave(n_pruned)
        head_indices = indices // k_len
        seq_indices = indices % k_len
        layer_idx = int(getattr(module, "layer_idx", len(tracker.records)))
        pruned_mask = torch.zeros((bsz, num_key_value_heads, k_len), dtype=torch.bool, device=indices.device)
        pruned_mask[batch_indices, head_indices, seq_indices] = True
        kept_mask = ~pruned_mask[0]
        tracker.records[str(layer_idx)] = [
            torch.nonzero(kept_mask[head_idx], as_tuple=False).flatten().detach().cpu().tolist()
            for head_idx in range(num_key_value_heads)
        ]
        module.masked_key_indices = (batch_indices, head_indices, seq_indices)
        return keys, values

    press_name = type(press).__name__
    if press_name == "ChunkKVPress" and hasattr(press, "press"):
        press.compress = MethodType(chunkkv_compress_with_recording, press)
    elif press_name == "AdaKVPress" and hasattr(press, "press"):
        press.compress = MethodType(adakv_compress_with_recording, press)
    elif hasattr(press, "score") and hasattr(press, "compression_ratio"):
        press.compress = MethodType(scorer_compress_with_recording, press)
    return tracker
