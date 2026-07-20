#!/usr/bin/env python3
"""Run local release checks for the public KVDiagnosis repository."""

from __future__ import annotations

import argparse
import gzip
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
    "low_mapped_coverage": 5047,
    "partial_mapped_coverage": 2866,
    "high_mapped_coverage_likelihood_drift": 19,
    "structural_position_likelihood_drift": 2126,
    "low_ear_candidate": 104,
    "decode_scorer_candidate": 405,
    "conflicting_diagnostic_signals": 1556,
    "ambiguous": 397,
}

EXPECTED_COVERAGE_TYPES = {
    "measured_token_coverage": 6211,
    "projected_token_coverage": 2224,
    "structural_position_addressability": 4085,
}

EXPECTED_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_ATTENTION_AVAILABLE = 7038
EXPECTED_TOPK_AVAILABLE = 8400
EXPECTED_CONTEXT_ATTENTION_AVAILABLE = 4608
EXPECTED_FULL_POPULATION = {
    "source_count": 2600,
    "planned_compressed_runs": 62400,
    "supported_compressed_runs": 59800,
    "unsupported_compressed_runs": 2600,
    "fullcache_correct_eligible_pairs": 48898,
    "selected_C_to_W_rows": 12520,
}
EXPECTED_TRANSITIONS = {
    "C->C": 36378,
    "C->W": 12520,
    "W->C": 1004,
    "W->W": 9898,
}

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
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, mode="rt", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
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
    if manifest.get("schema_version") != "kvdiagnosis.public_artifact.v0.4":
        raise SystemExit("unexpected artifact manifest schema")
    if manifest.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise SystemExit("artifact manifest model revision mismatch")
    if manifest.get("selected_failure_counts") != EXPECTED_COUNTS:
        raise SystemExit("artifact manifest selected-failure counts mismatch")
    if manifest.get("failure_signature_counts") != EXPECTED_SIGNATURES:
        raise SystemExit("artifact manifest failure-signature counts mismatch")
    if manifest.get("coverage_type_counts") != EXPECTED_COVERAGE_TYPES:
        raise SystemExit("artifact manifest coverage-type counts mismatch")
    if manifest.get("full_population_counts") != EXPECTED_FULL_POPULATION:
        raise SystemExit("artifact manifest full-population counts mismatch")

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


def iter_jsonl_gz(path: Path):
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def verify_full_population(root: Path, selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    population_dir = root / "data/processed/full_population"
    summary = json.loads((population_dir / "summary.json").read_text(encoding="utf-8"))
    for field, expected in EXPECTED_FULL_POPULATION.items():
        if summary.get(field) != expected:
            raise SystemExit(
                f"full-population {field} mismatch: {summary.get(field)} != {expected}"
            )
    if summary.get("transitions") != EXPECTED_TRANSITIONS:
        raise SystemExit(f"full-population transitions mismatch: {summary.get('transitions')}")

    fullcache = list(iter_jsonl_gz(population_dir / "fullcache.jsonl.gz"))
    if len(fullcache) != EXPECTED_FULL_POPULATION["source_count"]:
        raise SystemExit("fullcache ledger row count mismatch")
    fullcache_keys = {row.get("fullcache_key") for row in fullcache}
    if len(fullcache_keys) != len(fullcache):
        raise SystemExit("fullcache ledger contains duplicate keys")

    compressed = []
    expected_dataset_runs = {
        "ruler8k": 26400,
        "ruler16k": 26400,
        "qasper": 4800,
        "hotpotqa": 4800,
    }
    for dataset, expected in expected_dataset_runs.items():
        rows = list(
            iter_jsonl_gz(population_dir / "compressed_runs" / f"{dataset}.jsonl.gz")
        )
        if len(rows) != expected or any(row.get("dataset") != dataset for row in rows):
            raise SystemExit(f"full-population dataset mismatch: {dataset}")
        compressed.extend(rows)
    if len(compressed) != EXPECTED_FULL_POPULATION["planned_compressed_runs"]:
        raise SystemExit("compressed-run ledger row count mismatch")
    run_keys = {
        (row.get("dataset"), row.get("sample_id"), row.get("method_name"), row.get("retained_budget"))
        for row in compressed
    }
    if len(run_keys) != len(compressed):
        raise SystemExit("compressed-run ledger contains duplicate keys")
    if any(row.get("fullcache_key") not in fullcache_keys for row in compressed):
        raise SystemExit("compressed-run ledger contains an unknown FullCache key")

    supported = [row for row in compressed if row.get("support_status") == "supported"]
    unsupported = [row for row in compressed if row.get("support_status") == "unsupported"]
    if len(supported) != EXPECTED_FULL_POPULATION["supported_compressed_runs"]:
        raise SystemExit("supported-run count mismatch")
    if len(unsupported) != EXPECTED_FULL_POPULATION["unsupported_compressed_runs"]:
        raise SystemExit("unsupported-run count mismatch")
    if any(
        row.get("method_name") != "ThinKPress" or row.get("retained_budget") != 0.25
        for row in unsupported
    ):
        raise SystemExit("unsupported ledger contains a non-ThinK/25% row")

    ledger_c_to_w = {
        (row["dataset"], row["sample_id"], row["method_name"], row["retained_budget"])
        for row in supported
        if row.get("outcome_transition") == "C->W"
    }
    selected_keys = {
        (row["dataset"], row["sample_id"], row["method_name"], row["retained_budget"])
        for row in selected_rows
    }
    if ledger_c_to_w != selected_keys:
        raise SystemExit("selected failures do not equal the ledger C->W population")
    return summary


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
    if report["coverage_types"] != dict(sorted(EXPECTED_COVERAGE_TYPES.items())):
        raise SystemExit(f"unexpected coverage types: {report['coverage_types']}")
    if report["attention_available"] != EXPECTED_ATTENTION_AVAILABLE:
        raise SystemExit(
            f"unexpected attention coverage: {report['attention_available']}"
        )
    if report["topk_available"] != EXPECTED_TOPK_AVAILABLE:
        raise SystemExit(f"unexpected TopK coverage: {report['topk_available']}")
    if {row.get("model_revision") for row in rows} != {EXPECTED_MODEL_REVISION}:
        raise SystemExit("selected rows do not bind to one expected model revision")

    population_summary = verify_full_population(root, rows)

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
    selected_context = {
        (row["sample_id"], row["method_name"], row["retained_budget"]): row
        for row in rows
        if row["dataset"] == "ruler8k"
    }
    for row in context_rows:
        key = (row["sample_id"], row["method_name"], row["retained_budget"])
        selected = selected_context[key]
        if row.get("coverage_type") != selected.get("coverage_type"):
            raise SystemExit("context-demand coverage type disagrees with selected corpus")
        for metric in ("ERR_slot", "ECov_slot"):
            if row.get(f"{metric}_status") != selected.get(f"{metric}_status"):
                raise SystemExit("context-demand slot applicability disagrees with selected corpus")
    expected_context_slot_metrics = sum(
        row.get("coverage_type")
        in {"measured_token_coverage", "projected_token_coverage"}
        for row in selected_context.values()
    )
    if context_report.get("slot_metrics_available") != expected_context_slot_metrics:
        raise SystemExit("context-demand slot-metric availability mismatch")
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

    pyramid_audit = json.loads(
        (root / "data/audits/pyramidkv_adapter_audit.json").read_text(encoding="utf-8")
    )
    if pyramid_audit.get("pair_count") != 7800:
        raise SystemExit("PyramidKV adapter audit pair count mismatch")
    for field in ("retained_original_positions_or_unit_mapping", "raw_output", "semantic_payload"):
        equality = pyramid_audit.get("overall_equality", {}).get(field, {})
        if equality.get("equal_pairs") != 7800 or equality.get("rate") != 1.0:
            raise SystemExit(f"PyramidKV adapter audit mismatch for {field}")

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
        root / "scripts/build_full_population_ledger.py",
        root / "scripts/build_release_artifacts.py",
        root / "scripts/regenerate_paper_artifacts.py",
        root / "scripts/migrate_coverage_applicability.py",
        root / "scripts/audit_pyramidkv_adapter.py",
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
    run(
        [
            sys.executable,
            str(root / "scripts/regenerate_paper_artifacts.py"),
            "--root",
            str(root),
            "--output-dir",
            "/tmp/kvcachebench-paper-artifacts",
            "--tables-only",
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
                "full_population": population_summary,
                "manifest_files": len(manifest["files"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
