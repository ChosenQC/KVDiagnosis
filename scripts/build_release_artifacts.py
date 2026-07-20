#!/usr/bin/env python3
"""Build the checked, paper-aligned public KVCacheBench artifact metadata."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_DATASETS = {
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

EXPECTED_FULL_POPULATION = {
    "source_count": 2600,
    "planned_compressed_runs": 62400,
    "supported_compressed_runs": 59800,
    "unsupported_compressed_runs": 2600,
    "fullcache_correct_eligible_pairs": 48898,
    "selected_C_to_W_rows": 12520,
}

MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator=chr(10))
        writer.writeheader()
        writer.writerows(rows)


def public_key(row: dict[str, Any]) -> tuple[str, str, float]:
    return (
        str(row["sample_id"]),
        str(row["method_name"]),
        round(float(row["retained_budget"]), 6),
    )


def metric_status(row: dict[str, Any], field: str, available: bool = True) -> tuple[Any, str, str | None]:
    value = row.get(field)
    if available and value is not None:
        return value, "available", None
    return None, "unavailable", "N/A: no valid diagnostic trace for this method/run"


def build_context_demand(
    public_rows: list[dict[str, Any]],
    source: Path,
    output_dir: Path,
) -> dict[str, Any]:
    ruler_rows = {
        public_key(row): row for row in public_rows if row["dataset"] == "ruler8k"
    }
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float]] = set()

    deprecated_prefixes = ("ERR_", "EAR_")
    deprecated_exact = {
        "delta_NLL_value",
        "delta_NLL_status",
        "delta_NLL_reason",
        "KL_value",
        "KL_status",
        "KL_reason",
        "TopK_value",
        "TopK_status",
        "TopK_reason",
    }

    for source_row in read_jsonl(source):
        row_key = public_key(source_row)
        diagnostic = ruler_rows.get(row_key)
        if diagnostic is None:
            continue
        row = {
            key: value
            for key, value in source_row.items()
            if not key.startswith(deprecated_prefixes) and key not in deprecated_exact
        }
        row["model_revision"] = MODEL_REVISION
        row["retention_metric_schema"] = "kvbench.slot_ecov.v1"
        row["retention_semantics"] = diagnostic["retention_semantics"]
        row["failure_signature"] = diagnostic["failure_signature"]
        row["ERR_slot_value"] = diagnostic["ERR_slot"]
        row["ERR_slot_status"] = "available"
        row["ECov_slot_value"] = diagnostic["ECov_slot"]
        row["ECov_slot_status"] = "available"
        for source_field, target_prefix, available in (
            ("delta_NLL", "delta_NLL", True),
            ("KL", "KL", True),
            ("TopK", "TopK", bool(diagnostic["topk_available"])),
            ("EAR", "EAR", bool(diagnostic["attention_available"])),
        ):
            value, status, reason = metric_status(diagnostic, source_field, available)
            row[f"{target_prefix}_value"] = value
            row[f"{target_prefix}_status"] = status
            row[f"{target_prefix}_reason"] = reason
        if not diagnostic["attention_available"]:
            row["attention_accessibility_demand_score"] = None
        output.append(row)
        seen.add(row_key)

    missing = sorted(set(ruler_rows) - seen)
    if missing:
        raise ValueError(f"context-demand source misses {len(missing)} audited keys: {missing[:5]}")
    output.sort(key=lambda row: public_key(row))
    write_jsonl(output_dir / "ruler8k_context_demand_dataset.jsonl", output)
    write_csv(output_dir / "ruler8k_context_demand_dataset.csv", output)

    report = {
        "schema_version": "kvcachebench.context_demand.v0.2",
        "rows": len(output),
        "expected_rows": EXPECTED_DATASETS["ruler8k"],
        "unique_samples": len({row["sample_id"] for row in output}),
        "duplicate_keys": len(output) - len({public_key(row) for row in output}),
        "slot_metrics_available": sum(
            row["ERR_slot_status"] == "available"
            and row["ECov_slot_status"] == "available"
            for row in output
        ),
        "attention_available": sum(row["EAR_status"] == "available" for row in output),
        "topk_available": sum(row["TopK_status"] == "available" for row in output),
        "complete": len(output) == EXPECTED_DATASETS["ruler8k"],
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "rows": report["rows"],
                "unique_samples": report["unique_samples"],
                "metrics": [
                    "ERR_slot",
                    "ECov_slot",
                    "delta_NLL",
                    "KL",
                    "TopK",
                    "EAR",
                ],
                "attention_available": report["attention_available"],
                "topk_available": report["topk_available"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def build_summaries(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    sys.path.insert(0, str(root / "src"))
    from kvcachebench.diagnostics import summarize_rows, validate_selected_failure_rows
    from kvcachebench.io import write_csv as package_write_csv

    report = validate_selected_failure_rows(rows)
    if not report["ok"]:
        raise ValueError(json.dumps(report, indent=2))

    summary_dir = root / "data" / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(
        rows, ["dataset", "method_name", "method_display", "retained_budget"]
    )
    package_write_csv(
        summary_dir / "selected_failures_by_dataset_method_budget.csv", summary
    )

    signature_fields = list(EXPECTED_SIGNATURES)
    groups: dict[tuple[str, str, str, float], Counter[str]] = defaultdict(Counter)
    for row in rows:
        group_key = (
            str(row["dataset"]),
            str(row["method_name"]),
            str(row["method_display"]),
            float(row["retained_budget"]),
        )
        groups[group_key][str(row["failure_signature"])] += 1
    signature_rows: list[dict[str, Any]] = []
    for group_key, counts in sorted(groups.items()):
        record: dict[str, Any] = dict(
            zip(
                ["dataset", "method_name", "method_display", "retained_budget"],
                group_key,
            )
        )
        record["failure_rows"] = sum(counts.values())
        for field in signature_fields:
            record[field] = counts[field]
        signature_rows.append(record)
    package_write_csv(
        summary_dir / "failure_signatures_by_dataset_method_budget.csv",
        signature_rows,
    )

    totals = {
        "schema_version": "kvcachebench.failure_signatures.v0.2",
        "failure_rows": len(rows),
        "counts": report["failure_signatures"],
        "percent": {
            key: value / len(rows) for key, value in report["failure_signatures"].items()
        },
        "attention_available": report["attention_available"],
        "topk_available": report["topk_available"],
        "thresholds": {
            "ECov_slot_span_fraction": 0.5,
            "low_slot_coverage_below": 0.5,
            "high_slot_coverage_at_least": 0.9,
            "likelihood_drift_delta_NLL_at_least": 1.0,
            "low_EAR_below": 0.5,
            "stable_abs_delta_NLL_at_most": 0.1,
            "stable_TopK_at_least_when_available": 0.9,
        },
        "interpretation": "Operational row-weighted signatures, not causal classes or deployment prevalence.",
    }
    (summary_dir / "failure_signature_totals.json").write_text(
        json.dumps(totals, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def sanitize_execution_audit(source: Path, output: Path) -> None:
    raw = json.loads(source.read_text(encoding="utf-8"))
    telemetry = list(raw.get("telemetry", {}).values())
    summary = {
        "schema_version": raw["schema_version"],
        "ECov_slot_threshold": raw["ecov_slot_threshold"],
        "model_revision": MODEL_REVISION,
        "sources": raw["sources"],
        "validated_rows": sum(
            int(item["completed_keys"]) for item in raw["sources"].values()
        ),
        "error_count": raw["error_count"],
        "telemetry": {
            "jobs_expected": 19,
            "jobs_found": len(telemetry),
            "all_average_gpu_utilization_at_least_75_percent": all(
                item.get("threshold_met") is True for item in telemetry
            ),
            "minimum_average_gpu_utilization_percent": min(
                float(item["average_gpu_utilization_percent"]) for item in telemetry
            ),
            "mean_average_gpu_utilization_percent": sum(
                float(item["average_gpu_utilization_percent"]) for item in telemetry
            )
            / len(telemetry),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def file_record(root: Path, path: Path) -> dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    rows: int | None = None
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, mode="rt", encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8") as handle:
            rows = max(sum(1 for _ in handle) - 1, 0)
    return {
        "path": rel,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": rows,
    }


def build_manifest(root: Path, validation: dict[str, Any]) -> None:
    data_root = root / "data"
    manifest_path = data_root / "metadata" / "artifact_manifest.json"
    files = [
        file_record(root, path)
        for path in sorted(data_root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    manifest = {
        "schema_version": "kvcachebench.public_artifact.v0.3",
        "name": "KVCacheBench public artifacts",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen/Qwen3-8B",
        "model_revision": MODEL_REVISION,
        "retention_metric_schema": "kvbench.slot_ecov.v1",
        "selected_failure_definition": "full_correct == true and compressed_correct == false",
        "selected_failure_counts": EXPECTED_DATASETS,
        "full_population_counts": EXPECTED_FULL_POPULATION,
        "failure_signature_counts": EXPECTED_SIGNATURES,
        "attention_available": validation["attention_available"],
        "topk_available": validation["topk_available"],
        "methods": [
            "StreamingLLMPress",
            "SnapKVPress",
            "TOVAPress",
            "KeyDiffPress",
            "ThinKPress",
            "ChunkKVPress_Knorm",
            "AdaKVPress",
            "QuantizedCache",
        ],
        "excluded_invalid_methods": ["PyramidKVPress"],
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--context-demand-source", type=Path)
    parser.add_argument("--slot-summary-source", type=Path)
    parser.add_argument("--matched-pair-summary-source", type=Path)
    parser.add_argument("--execution-audit-source", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()

    rows = read_jsonl(
        root / "data" / "processed" / "selected_failures" / "all_selected_failures.jsonl"
    )
    validation = build_summaries(root, rows)

    if validation["rows"] != EXPECTED_DATASETS["all"]:
        raise ValueError(f"unexpected total: {validation['rows']}")
    if validation["datasets"] != {
        key: value for key, value in EXPECTED_DATASETS.items() if key != "all"
    }:
        raise ValueError(f"unexpected dataset counts: {validation['datasets']}")
    if validation["failure_signatures"] != dict(sorted(EXPECTED_SIGNATURES.items())):
        raise ValueError(
            f"unexpected signature counts: {validation['failure_signatures']}"
        )

    if args.context_demand_source:
        build_context_demand(
            rows, args.context_demand_source, root / "data" / "context_demand"
        )
    if args.slot_summary_source:
        destination = root / "data" / "summaries" / "slot_ecov_summary.csv"
        destination.write_text(
            args.slot_summary_source.read_text(encoding="utf-8").replace(
                chr(13) + chr(10), chr(10)
            ),
            encoding="utf-8",
        )
    if args.matched_pair_summary_source:
        destination = (
            root / "data" / "summaries" / "matched_method_pair_summary.csv"
        )
        destination.write_text(
            args.matched_pair_summary_source.read_text(encoding="utf-8").replace(
                chr(13) + chr(10), chr(10)
            ),
            encoding="utf-8",
        )
    if args.execution_audit_source:
        sanitize_execution_audit(
            args.execution_audit_source,
            root / "data" / "audits" / "slot_ecov_execution_audit.json",
        )

    build_manifest(root, validation)
    print(
        json.dumps(
            {
                "ok": True,
                "rows": validation["rows"],
                "datasets": validation["datasets"],
                "failure_signatures": validation["failure_signatures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
