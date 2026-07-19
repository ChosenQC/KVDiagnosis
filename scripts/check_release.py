#!/usr/bin/env python3
"""Run local release checks for the public KVCacheBench repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SECRET_RE = re.compile(
    r"olp_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"OPENAI_API_KEY|ANTHROPIC_API_KEY|\bsk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}"
)

EXPECTED_COUNTS = {
    "all": 12520,
    "ruler8k": 5970,
    "ruler16k": 5396,
    "qasper": 327,
    "hotpotqa": 827,
}

EXPECTED_SIGNATURES = {
    "low_slot_coverage": 5047,
    "partial_slot_coverage": 2866,
    "high_coverage_likelihood_drift": 2145,
    "low_ear_candidate": 104,
    "decode_scorer_candidate": 405,
    "conflicting_retained_signals": 1556,
    "ambiguous": 397,
}

EXPECTED_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_ATTENTION_AVAILABLE = 7038
EXPECTED_TOPK_AVAILABLE = 8400
EXPECTED_CONTEXT_ATTENTION_AVAILABLE = 4608

DEPRECATED_PATHS = [
    "data/processed/selected_failures/longbench_v2_proxy90.jsonl",
    "data/processed/selected_failures/qasper_hotpot.jsonl",
    "data/summaries/completed_failure_summary.csv",
    "data/summaries/completed_failure_summary.json",
]


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def scan_secrets(root: Path) -> list[str]:
    hits: list[str] = []
    skip_dirs = {".git", "__pycache__", ".venv", ".pytest_cache", "dist"}
    for path in root.rglob("*"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if path == Path(__file__).resolve() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SECRET_RE.search(text):
            hits.append(str(path.relative_to(root)))
    return hits


def count_rows(path: Path) -> int | None:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if path.suffix == ".csv":
        with path.open(encoding="utf-8") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    return None


def verify_manifest(root: Path) -> dict[str, Any]:
    path = root / "data/metadata/artifact_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "kvcachebench.public_artifact.v0.2":
        raise SystemExit("unexpected artifact manifest schema")
    if manifest.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise SystemExit("artifact manifest model revision mismatch")
    if manifest.get("selected_failure_counts") != EXPECTED_COUNTS:
        raise SystemExit("artifact manifest selected-failure counts mismatch")
    if manifest.get("failure_signature_counts") != EXPECTED_SIGNATURES:
        raise SystemExit("artifact manifest failure-signature counts mismatch")

    for record in manifest.get("files", []):
        file_path = root / record["path"]
        if not file_path.is_file():
            raise SystemExit(f"manifest file missing: {record['path']}")
        if file_path.stat().st_size != record["bytes"]:
            raise SystemExit(f"manifest byte count mismatch: {record['path']}")
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise SystemExit(f"manifest digest mismatch: {record['path']}")
        if count_rows(file_path) != record["rows"]:
            raise SystemExit(f"manifest row count mismatch: {record['path']}")
    manifest_paths = {record["path"] for record in manifest["files"]}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in (root / "data").rglob("*")
        if path.is_file() and path != root / "data/metadata/artifact_manifest.json"
    }
    if manifest_paths != actual_paths:
        raise SystemExit(
            f"manifest coverage mismatch: missing={sorted(actual_paths - manifest_paths)}, "
            f"extra={sorted(manifest_paths - actual_paths)}"
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--max-file-mb", type=float, default=90.0)
    args = parser.parse_args()
    root = args.root.resolve()
    data_dir = root / "data/processed/selected_failures"
    data_path = data_dir / "all_selected_failures.jsonl"

    sys.path.insert(0, str(root / "src"))
    from kvcachebench.diagnostics import validate_selected_failure_rows
    from kvcachebench.io import read_jsonl

    rows = read_jsonl(data_path)
    report = validate_selected_failure_rows(rows)
    if not report["ok"]:
        raise SystemExit(json.dumps(report, indent=2))
    if report["rows"] != EXPECTED_COUNTS["all"]:
        raise SystemExit(f"unexpected row count: {report['rows']}")
    expected_datasets = {
        key: value for key, value in EXPECTED_COUNTS.items() if key != "all"
    }
    if report["datasets"] != expected_datasets:
        raise SystemExit(f"unexpected dataset counts: {report['datasets']}")
    if report["failure_signatures"] != dict(sorted(EXPECTED_SIGNATURES.items())):
        raise SystemExit(
            f"unexpected failure-signature counts: {report['failure_signatures']}"
        )
    if report["attention_available"] != EXPECTED_ATTENTION_AVAILABLE:
        raise SystemExit(
            f"unexpected attention coverage: {report['attention_available']}"
        )
    if report["topk_available"] != EXPECTED_TOPK_AVAILABLE:
        raise SystemExit(f"unexpected TopK coverage: {report['topk_available']}")
    if {row.get("model_revision") for row in rows} != {EXPECTED_MODEL_REVISION}:
        raise SystemExit("selected rows do not bind to one expected model revision")

    for dataset, expected in expected_datasets.items():
        individual = read_jsonl(data_dir / f"{dataset}.jsonl")
        if len(individual) != expected:
            raise SystemExit(
                f"unexpected count for {dataset}: {len(individual)} != {expected}"
            )
        if any(row.get("dataset") != dataset for row in individual):
            raise SystemExit(f"cross-dataset row found in {dataset}.jsonl")

    context_report = json.loads(
        (root / "data/context_demand/validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    if context_report.get("rows") != EXPECTED_COUNTS["ruler8k"]:
        raise SystemExit("context-demand row count mismatch")
    if context_report.get("attention_available") != EXPECTED_CONTEXT_ATTENTION_AVAILABLE:
        raise SystemExit("context-demand attention coverage mismatch")
    if context_report.get("complete") is not True:
        raise SystemExit("context-demand validation is incomplete")

    context_rows = read_jsonl(
        root / "data/context_demand/ruler8k_context_demand_dataset.jsonl"
    )
    if len(context_rows) != EXPECTED_COUNTS["ruler8k"]:
        raise SystemExit("context-demand JSONL row count mismatch")
    context_keys = {
        (
            row.get("sample_id"),
            row.get("method_name"),
            row.get("retained_budget"),
        )
        for row in context_rows
    }
    if len(context_keys) != len(context_rows):
        raise SystemExit("context-demand JSONL contains duplicate keys")
    if any(
        field in row
        for row in context_rows
        for field in ("ERR_value", "ECov_value")
    ):
        raise SystemExit("context-demand JSONL contains legacy cache metrics")
    if any(
        row.get("ERR_slot_status") != "available"
        or row.get("ECov_slot_status") != "available"
        for row in context_rows
    ):
        raise SystemExit("context-demand JSONL has incomplete slot metrics")
    if sum(row.get("EAR_status") == "available" for row in context_rows) != (
        EXPECTED_CONTEXT_ATTENTION_AVAILABLE
    ):
        raise SystemExit("context-demand JSONL attention coverage mismatch")
    if {row.get("model_revision") for row in context_rows} != {
        EXPECTED_MODEL_REVISION
    }:
        raise SystemExit("context-demand rows have an unexpected model revision")

    execution_audit = json.loads(
        (root / "data/audits/slot_ecov_execution_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if execution_audit.get("validated_rows") != EXPECTED_COUNTS["all"]:
        raise SystemExit("slot execution audit row count mismatch")
    if execution_audit.get("error_count") != 0:
        raise SystemExit("slot execution audit contains errors")
    telemetry = execution_audit.get("telemetry", {})
    if telemetry.get("jobs_found") != telemetry.get("jobs_expected"):
        raise SystemExit("slot execution telemetry is incomplete")
    if telemetry.get("all_average_gpu_utilization_at_least_75_percent") is not True:
        raise SystemExit("slot execution telemetry failed the utilization gate")

    for rel in DEPRECATED_PATHS:
        if (root / rel).exists():
            raise SystemExit(f"deprecated artifact is still present: {rel}")

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

    manifest = verify_manifest(root)
    scripts = [
        root / "scripts/export_selected_failures.py",
        root / "scripts/build_release_artifacts.py",
        root / "scripts/check_release.py",
    ]
    run(
        [
            sys.executable,
            "-m",
            "py_compile",
            *map(str, sorted((root / "src/kvcachebench").glob("*.py"))),
            *map(str, scripts),
        ],
        root,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "rows": report["rows"],
                "datasets": report["datasets"],
                "failure_signatures": report["failure_signatures"],
                "attention_available": report["attention_available"],
                "topk_available": report["topk_available"],
                "manifest_files": len(manifest["files"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
