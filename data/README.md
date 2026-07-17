# KVCacheBench Data

This directory contains compact public artifacts for the benchmark paper.

- `processed/selected_failures`: final FullCache-correct/compressed-wrong rows.
- `context_demand`: RULER-8K context-demand labels and validation report.
- `summaries`: aggregate CSV/JSON tables used by paper analyses.
- `examples`: small JSONL files for quick tests and tutorials.
- `metadata/artifact_manifest.json`: row counts, checksums, and provenance.

Full raw benchmark prompts and per-unit run files are intentionally not stored
here. They are large and are better regenerated from the official benchmark
sources using the documented scripts.
