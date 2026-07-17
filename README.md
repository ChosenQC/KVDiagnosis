# KVCacheBench

KVCacheBench is a failure-focused benchmark companion for studying KV cache
compression. Instead of reporting only final accuracy, the released artifacts
isolate rows where the full-cache model is correct and a compressed-cache run
fails, then expose cache-, logit-, and attention-level diagnostics for those
failures.

This repository is the clean public package for the benchmark paper. It
contains:

- a small Python package for scoring, validating, and summarizing diagnostic rows;
- selected failure datasets for RULER-8K, RULER-16K, LongBench V2 proxy90,
  and Qasper/HotpotQA gold-evidence QA;
- RULER-8K context-demand annotations used to connect sample requirements to
  measured compression failures;
- summary CSV/JSON artifacts used by the paper tables and figures;
- scripts that regenerate the compact public failure corpus from full
  `paired_metrics.jsonl` files.

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .

kvcachebench validate data/processed/selected_failures/all_selected_failures.jsonl
kvcachebench summarize       data/processed/selected_failures/all_selected_failures.jsonl       --group-by dataset,method_name,retained_budget       --output /tmp/kvbench_summary.csv
```

Expected validation summary for the released selected-failure corpus:

- `13,163` selected failure rows
- `6,351` RULER-8K rows
- `5,603` RULER-16K rows
- `1,204` Qasper/HotpotQA rows
- `5` LongBench V2 proxy90 rows

## Data Layout

```text
data/processed/selected_failures/
  all_selected_failures.jsonl
  ruler8k.jsonl
  ruler16k.jsonl
  qasper_hotpot.jsonl
  longbench_v2_proxy90.jsonl
data/context_demand/
  ruler8k_context_demand_dataset.jsonl
  ruler8k_context_demand_dataset.csv
  summary.json
  validation_report.json
data/summaries/
  *_summary_by_*_budget.csv
  selected_failures_by_dataset_method_budget.csv
data/metadata/artifact_manifest.json
```

The selected-failure JSONL files do not include full benchmark prompts. They
include sample identifiers, method/budget metadata, outputs, correctness,
evidence diagnostics, and logit diagnostics. Original prompts can be
regenerated from the benchmark sources using the preparation scripts described
in `docs/reproduction.md`.

## Core Definition

A selected failure row is one paired run where:

```text
full_correct == true and compressed_correct == false
```

The full-cache and compressed-cache runs share the same model, prompt,
tokenizer, scoring rule, and decoding setup. This makes the row useful for
diagnosing compression-induced degradation rather than base-model failure.

## Environment Used in the Paper

The main experiments used Qwen3-8B with deterministic decoding. Non-quantized
KV compression methods were implemented with `kvpress==0.5.3`; Hugging Face
`QuantizedCache` was evaluated through the Transformers cache API. See
`requirements/requirements_current_experiment.txt` for the pinned research
environment and `docs/reproduction.md` for lighter analysis-only usage.

## Repository Status

This is a benchmark artifact repository. It is intended to be directly useful
for reviewers and follow-up method papers: users can inspect failure rows,
recompute aggregate diagnostics, and plug new method outputs into the same
schema.
