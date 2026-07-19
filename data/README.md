# KVCacheBench Data

This directory contains the compact public artifact aligned with the final
slot-level paper analysis.

- `processed/selected_failures`: 12,520 audited FullCache-correct /
  compressed-wrong method-ratio rows.
- `context_demand`: 5,970 RULER-8K rows with deterministic demand labels and
  corrected slot metrics.
- `summaries`: method/budget slot coverage, failure signatures, and matched
  failure-set comparisons.
- `audits`: sanitized completeness and GPU-utilization validation.
- `examples`: a small multi-method JSONL sample.
- `metadata/artifact_manifest.json`: provenance, counts, bytes, and SHA-256
  hashes for every public data file.

Full prompts, model weights, raw source datasets, per-unit retained-position
maps, and scheduler logs are intentionally excluded. Regeneration instructions
are in `docs/reproduction.md`.
