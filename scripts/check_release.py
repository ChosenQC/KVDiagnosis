#!/usr/bin/env python3
"""Run local release checks for the public KVCacheBench repository."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SECRET_RE = re.compile(
    r"olp_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"OPENAI_API_KEY|ANTHROPIC_API_KEY|\bsk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}"
)

EXPECTED_COUNTS = {
    "all": 13163,
    "ruler8k": 6351,
    "ruler16k": 5603,
    "qasper_hotpot": 1204,
    "longbench_v2_proxy90": 5,
}

def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)

def scan_secrets(root: Path) -> list[str]:
    hits: list[str] = []
    skip_dirs = {".git", "__pycache__", ".venv", ".pytest_cache"}
    for path in root.rglob("*"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if path == Path(__file__).resolve():
            continue
        if not path.is_file():
            continue
        if path.suffix in {".jsonl"} and "data/processed" in path.as_posix():
            # Model outputs contain arbitrary English text; release token checks are
            # still applied to docs/code/metadata below.
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SECRET_RE.search(text):
            hits.append(str(path.relative_to(root)))
    return hits

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-file-mb", type=float, default=90.0)
    args = parser.parse_args()
    root = args.root.resolve()
    data_path = root / "data/processed/selected_failures/all_selected_failures.jsonl"

    sys.path.insert(0, str(root / "src"))
    from kvcachebench.io import read_jsonl
    from kvcachebench.diagnostics import validate_selected_failure_rows

    rows = read_jsonl(data_path)
    report = validate_selected_failure_rows(rows)
    if not report["ok"]:
        raise SystemExit(json.dumps(report, indent=2))
    if report["rows"] != EXPECTED_COUNTS["all"]:
        raise SystemExit(f"unexpected row count: {report['rows']}")
    for dataset, expected in EXPECTED_COUNTS.items():
        if dataset == "all":
            continue
        observed = report["datasets"].get(dataset)
        if observed != expected:
            raise SystemExit(f"unexpected count for {dataset}: {observed} != {expected}")

    too_large = []
    for path in root.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > args.max_file_mb:
                too_large.append((str(path.relative_to(root)), round(size_mb, 2)))
    if too_large:
        raise SystemExit(f"files exceed {args.max_file_mb} MB: {too_large}")

    secret_hits = scan_secrets(root)
    if secret_hits:
        raise SystemExit(f"possible secrets found: {secret_hits}")

    run([sys.executable, "-m", "py_compile", *map(str, sorted((root / "src/kvcachebench").glob("*.py"))), str(root / "scripts/export_selected_failures.py")], root)
    print(json.dumps({"ok": True, "rows": report["rows"], "datasets": report["datasets"]}, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
