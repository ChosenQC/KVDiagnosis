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

python scripts/regenerate_paper_artifacts.py \
  --output-dir paper_artifacts/generated
```

The release checker validates the complete population ledger, selected C->W
identity, unique keys, slot metric schema, observational signatures, trace
coverage, immutable model revision, sanitized execution telemetry, file sizes,
secret patterns, every manifest hash, and CPU-only paper-table regeneration.

## Rebuilding the Public Corpus

Build the normalized complete ledger from the frozen paired run files:

```bash
python scripts/build_full_population_ledger.py \
  --results-root /path/to/KVbench/results
```

The builder requires every source in every method-setting cell, preserves
ThinK/25% as explicit N/A, and rejects the build unless the resulting C->W key
set exactly matches the selected diagnostic corpus.

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

## Regenerating Paper Results

Install the analysis extra and regenerate the paper-facing CSVs and four main
figures directly from the public ledger and selected diagnostic rows:

```bash
pip install -e '.[analysis]'
python scripts/regenerate_paper_artifacts.py \
  --output-dir paper_artifacts/generated \
  --assets-dir assets
```

Use `--tables-only` for dependency-free CPU regeneration of all reported
population, overlap, diagnostic-profile, and QA-signature values.

## Full Inference Environment

Full reruns require a GPU capable of loading Qwen3-8B. The frozen experiment
uses:

- Qwen/Qwen3-8B revision
  `b968826d9c46dd6066d109eabc6255188de91218`;
- deterministic decoding and paired prompt/scorer settings;
- `kvpress==0.5.3` for non-quantized methods;
- Hugging Face QuantizedCache through the Transformers cache API;
- the lock in `requirements/requirements_current_experiment.txt`.
