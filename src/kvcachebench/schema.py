from __future__ import annotations

from typing import Any


METHODS = [
    ("StreamingLLMPress", "recency_attention_sink_pruning"),
    ("SnapKVPress", "attention_observation_pruning"),
    ("TOVAPress", "last_query_attention_pruning"),
    ("KeyDiffPress", "key_space_signal_pruning"),
    ("ThinKPress", "key_channel_pruning"),
    ("ChunkKVPress_Knorm", "chunk_semantic_preservation"),
    ("AdaKVPress", "adaptive_headwise_pruning"),
    ("QuantizedCache", "hf_kv_cache_quantization"),
]
TASKS = {
    "niah_single_1": "ruler_single_niah",
    "niah_single_2": "ruler_single_niah",
    "niah_single_3": "ruler_single_niah",
    "niah_multikey_1": "ruler_multikey_niah",
    "niah_multikey_2": "ruler_multikey_niah",
    "niah_multikey_3": "ruler_multikey_niah",
    "niah_multivalue": "ruler_multivalue_niah",
    "niah_multiquery": "ruler_multiquery_niah",
    "vt": "ruler_variable_tracking",
    "cwe": "ruler_common_words_extraction",
    "fwe": "ruler_frequent_words_extraction",
}

MAX_NEW_TOKENS = {
    "ruler_single_niah": 64,
    "ruler_multikey_niah": 64,
    "ruler_multivalue_niah": 64,
    "ruler_multiquery_niah": 64,
    "ruler_variable_tracking": 128,
    "ruler_common_words_extraction": 256,
    "ruler_frequent_words_extraction": 256,
}

REQUIRED_FIELDS = [
    "experiment_id",
    "sample_id",
    "task_key",
    "official_task_name",
    "context_length_tokens",
    "model_id",
    "method_name",
    "method_family",
    "compression_ratio",
    "retained_budget",
    "full_run_id",
    "compressed_run_id",
    "prompt_hash",
    "reference_answer",
    "raw_output_full",
    "raw_output_compressed",
    "extracted_answer_full",
    "extracted_answer_compressed",
    "full_score",
    "compressed_score",
    "score_drop",
    "full_correct",
    "compressed_correct",
    "CIF",
    "ERR_slot",
    "ECov_slot",
    "retention_semantics",
    "failure_signature",
    "full_gold_NLL",
    "compressed_gold_NLL",
    "delta_NLL",
    "GPR",
    "metric_applicability_notes",
]


def na(reason: str) -> dict[str, Any]:
    return {"value": "N/A", "reason": reason}
