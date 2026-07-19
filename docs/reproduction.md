# Reproduction Notes

## Analysis-Only Reproduction

No GPU dependencies are required:

```bash
pip install -e .

kvcachebench validate \
  data/processed/selected_failures/all_selected_failures.jsonl

kvcachebench summarize \
  data/processed/selected_failures/all_selected_failures.jsonl \
  --group-by dataset,method_name,retained_budget \
  --output /tmp/selected_failures_summary.csv

python scripts/check_release.py
```

The release checker validates counts, unique keys, slot metric schema,
observational signatures, trace coverage, immutable model revision, sanitized
execution telemetry, file sizes, secret patterns, and every manifest hash.

## Rebuilding the Public Corpus

Full internal reruns produce paired metrics plus the frozen
`failure_regime_rows.csv` audit table. Re-export with:

```bash
python scripts/export_selected_failures.py \
  --results-root /path/to/KVbench/results \
  --audit-rows /path/to/failure_regime_rows.csv \
  --output-dir data/processed/selected_failures
```

The audit table is the authoritative key set. Export fails if any audited key is
missing from paired metrics, is not C->W, uses a non-release method, or appears
more than once.

Then rebuild summaries, the RULER-8K context view, sanitized execution audit,
and manifest:

```bash
python scripts/build_release_artifacts.py \
  --context-demand-source /path/to/ruler8k_context_demand_dataset.jsonl \
  --slot-summary-source /path/to/slot_ecov_summary.csv \
  --matched-pair-summary-source /path/to/matched_method_pair_summary.csv \
  --execution-audit-source /path/to/slot_ecov_audit.json
```

Run `python scripts/check_release.py` after every rebuild.

## Full Inference Environment

Full reruns require a GPU capable of loading Qwen3-8B. The frozen experiment
uses:

- Qwen/Qwen3-8B revision
  `b968826d9c46dd6066d109eabc6255188de91218`;
- deterministic decoding and paired prompt/scorer settings;
- `kvpress==0.5.3` for non-quantized methods;
- Hugging Face QuantizedCache through the Transformers cache API;
- the lock in `requirements/requirements_current_experiment.txt`.

Original prompts are regenerated from NVIDIA/RULER, Qasper, and HotpotQA.
Private model caches, raw source dumps, per-slot retained-position maps, and
scheduler logs are intentionally excluded from this public repository.
