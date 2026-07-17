# Data Format

## Selected Failure Rows

Each row in `data/processed/selected_failures/*.jsonl` is a paired
FullCache/compressed-cache comparison. Important fields:

- `dataset`: one of `ruler8k`, `ruler16k`, `longbench_v2_proxy90`, `qasper_hotpot`.
- `sample_id`: benchmark-local sample identifier.
- `method_name`: implementation identifier, e.g. `StreamingLLMPress`.
- `compression_ratio`: fraction removed by the method implementation.
- `retained_budget`: reported retained KV fraction, equal to `1 - compression_ratio` for retention methods.
- `full_correct`, `compressed_correct`: paired correctness values.
- `ERR`, `ECov`, `DRR`: evidence retention / coverage / distractor diagnostics when applicable.
- `delta_NLL`, `GPR`: logit-level answer likelihood diagnostics.
- `metric_applicability_notes`: method-specific notes for N/A values.

Rows are selected only when `full_correct` is true and `compressed_correct` is
false. N/A metrics are explicit JSON objects of the form
`{ "value": "N/A", "reason": "..." }`.

## Context-Demand Rows

`data/context_demand/ruler8k_context_demand_dataset.jsonl` adds RULER-8K
demand labels to selected failure rows. The released dimensions are:

- `evidence_presence`
- `cue_integrity`
- `long_range_retrieval`
- `exact_token_fidelity`
- `answer_format_alignment`
- `logit_stability_demand`
- `attention_accessibility_demand`

These labels describe what the input sample requires. They are intentionally
separated from measured compression diagnostics such as ERR, EAR, and
delta-NLL.
