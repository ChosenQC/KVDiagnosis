# KVDiagnosis Data

This directory contains the public artifact aligned with the final slot-level
paper analysis.

- `processed/full_population`: 2,600 normalized FullCache controls and all
  62,400 planned method-setting records, including 2,600 explicit N/A rows.
- `processed/selected_failures`: 12,520 audited FullCache-correct /
  compressed-wrong method-ratio rows with explicit measured, projected, or
  structural coverage applicability.
- `context_demand`: 5,970 RULER-8K rows with deterministic demand labels and
  corrected slot metrics.
- `summaries`: method/budget slot coverage, failure signatures, and matched
  failure-set comparisons.
- `audits`: sanitized completeness/GPU-utilization validation and the
  7,800-pair PyramidKV adapter audit.
- `examples`: a small multi-method JSONL sample.
- `metadata/artifact_manifest.json`: provenance, counts, bytes, and SHA-256
  hashes for every public data file.

Regeneration instructions are in `docs/reproduction.md`.
