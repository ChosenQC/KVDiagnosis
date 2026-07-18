# GitHub Release Checklist

This repository is ready to push once a GitHub target repository exists.

## Local Gates

```bash
python scripts/check_release.py
python -m venv /tmp/kvcachebench-venv
/tmp/kvcachebench-venv/bin/python -m pip install --no-deps -e .
/tmp/kvcachebench-venv/bin/kvcachebench validate data/processed/selected_failures/all_selected_failures.jsonl
```

Expected counts:

- all selected failures: 13,163
- RULER-8K: 6,351
- RULER-16K: 5,603
- Qasper/HotpotQA: 1,204
- LongBench V2 proxy90: 5

## Push To GitHub

Create an empty GitHub repository, then run:

```bash
git remote add origin git@github.com:OWNER/KVCacheBench.git
git push -u origin main
```

If using HTTPS with a token:

```bash
git remote add origin https://github.com/OWNER/KVCacheBench.git
GIT_ASKPASS=/path/to/askpass GIT_TERMINAL_PROMPT=0 git push -u origin main
```

Do not commit tokens, SSH keys, model checkpoints, raw benchmark dumps, Slurm
logs, or per-unit `.done.json` files.

## After Push

- Replace `REPLACE_WITH_OWNER` in `CITATION.cff` with the final GitHub owner.
- Confirm the GitHub Actions CI workflow passes.
- Add the repository URL to the paper artifact/reproducibility statement.
- Optionally create a release tag, e.g. `v0.1.0-paper-artifact`.
