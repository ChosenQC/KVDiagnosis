# GitHub Release Checklist

## Local Gates

```bash
python scripts/check_release.py
python -m venv /tmp/kvcachebench-venv
/tmp/kvcachebench-venv/bin/python -m pip install --no-deps -e .
/tmp/kvcachebench-venv/bin/kvcachebench validate \
  data/processed/selected_failures/all_selected_failures.jsonl
```

Expected counts:

- all selected failures: 12,520
- RULER-8K: 5,970
- RULER-16K: 5,396
- Qasper: 327
- HotpotQA: 827
- valid attention rows: 7,038
- valid TopK rows: 8,400

The release gate also verifies all manifest hashes, the immutable model
revision, 19/19 accepted telemetry jobs, the 75% average GPU-utilization gate,
and absence of deprecated PyramidKV/legacy-union artifacts.

## Push To GitHub

This local checkout currently has no configured `origin`. Create an empty
GitHub repository, then run:

```bash
scripts/push_to_github.sh git@github.com:OWNER/KVCacheBench.git
```

Do not commit tokens, SSH keys, model checkpoints, raw benchmark dumps, Slurm
logs, or per-unit `.done.json` files.

## After Push

- Replace `REPLACE_WITH_OWNER` in `CITATION.cff`.
- Confirm GitHub Actions passes.
- Add the immutable repository/release URL to the paper artifact statement.
- Tag the paper artifact, for example `v0.2.0-paper-artifact`.
