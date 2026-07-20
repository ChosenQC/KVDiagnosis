# Data Format

## Full-Population Ledger

`data/processed/full_population` is normalized around the shared FullCache
control:

- `fullcache.jsonl.gz` contains one FullCache record for each of 2,600 sources.
- `compressed_runs/*.jsonl.gz` contains all 62,400 planned method-setting
  records: 59,800 supported runs and 2,600 explicit ThinK/25% N/A records.
- `summary.json` records dataset counts and the four outcome transitions.

Each compressed row links to `fullcache_key` and records its method, configured
ratio, output, score, correctness, support status, and transition. The unique
compressed-run key is:

```text
(dataset, sample_id, method_name, retained_budget)
```

The release checker verifies that all supported C->W keys in this ledger equal
the 12,520 selected diagnostic rows exactly.

## Selected Failure Rows

Each row in `data/processed/selected_failures/*.jsonl` is a paired
FullCache/compressed comparison selected only when:

```text
full_correct == true and compressed_correct == false
```

Identity and execution fields include:

- `dataset`: `ruler8k`, `ruler16k`, `qasper`, or `hotpotqa`.
- `sample_id`, `prompt_hash`: stable benchmark-local identifiers.
- `method_name`, `method_family`, `method_display`.
- `compression_ratio`, `retained_budget`: method configuration values.
- `model_id`, `model_revision`, paired run IDs and task metadata.
- paired raw/extracted outputs, references, scores, and correctness.

Diagnostic fields include:

- `retention_metric_schema = "kvbench.slot_ecov.v1"`
- `ERR_slot`, `ECov_slot`, `ECov_slot_threshold`
- `retention_semantics`
- `delta_NLL`, `GPR`, `KL`, `TopK`, `EAR`, `gold_rank_shift`
- `attention_available`, `topk_available`
- `failure_signature`, `primitive_flags`

Unavailable optional metrics are JSON `null`. The availability booleans state
whether a missing attention or TopK value is semantically unavailable. Legacy
cross-slot-union `ERR`, `ECov`, and `DRR` are intentionally absent.

The unique row key is:

```text
(dataset, sample_id, method_name, compression_ratio)
```

## Context-Demand Rows

`data/context_demand/ruler8k_context_demand_dataset.jsonl` joins the 5,970
audited RULER-8K rows with deterministic task-template labels:

- evidence presence
- cue integrity
- long-range retrieval
- exact-token fidelity
- answer-format alignment
- logit stability demand
- attention accessibility demand

The file uses `ERR_slot_value` and `ECov_slot_value`. EAR and attention
demand are unavailable for methods without a valid eager-attention replay.

## Summaries and Audits

- `slot_ecov_summary.csv`: method/dataset/KV-ratio slot coverage means.
- `failure_signatures_by_dataset_method_budget.csv`: operational signature
  counts by cell.
- `failure_signature_totals.json`: frozen thresholds and total counts.
- `matched_method_pair_summary.csv`: matched-source failure overlap.
- `slot_ecov_execution_audit.json`: expected/completed keys and GPU telemetry
  gate summary.
- `artifact_manifest.json`: byte size, row count, and SHA-256 for every public
  data file.
