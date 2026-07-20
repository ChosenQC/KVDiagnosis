#!/usr/bin/env python3
"""Apply the coverage-applicability schema to committed public artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SELECTED_FILES = (
    "all_selected_failures.jsonl",
    "ruler8k.jsonl",
    "ruler16k.jsonl",
    "qasper.jsonl",
    "hotpotqa.jsonl",
)
STRUCTURAL_REASON = (
    "N/A: token positions remain addressable by construction; no measured or "
    "projected retained-token map exists"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def migrate_row(row: dict[str, Any], coverage_type_for_method, classify_failure_signature) -> dict[str, Any]:
    migrated = dict(row)
    coverage_type = coverage_type_for_method(str(migrated.get("method_name")))
    migrated["coverage_applicability_schema"] = "kvdiagnosis.coverage.v2"
    migrated["coverage_type"] = coverage_type
    if coverage_type == "structural_position_addressability":
        migrated["ERR_slot"] = None
        migrated["ECov_slot"] = None
        migrated["ERR_slot_status"] = "not_applicable"
        migrated["ECov_slot_status"] = "not_applicable"
        migrated["ERR_slot_reason"] = STRUCTURAL_REASON
        migrated["ECov_slot_reason"] = STRUCTURAL_REASON
        migrated["structural_position_addressability"] = True
    elif coverage_type in {"measured_token_coverage", "projected_token_coverage"}:
        migrated["ERR_slot_status"] = "available"
        migrated["ECov_slot_status"] = "available"
        migrated["ERR_slot_reason"] = None
        migrated["ECov_slot_reason"] = None
        migrated["structural_position_addressability"] = False
    else:
        migrated["ERR_slot"] = None
        migrated["ECov_slot"] = None
        migrated["ERR_slot_status"] = "not_applicable"
        migrated["ECov_slot_status"] = "not_applicable"
        migrated["ERR_slot_reason"] = "N/A: no valid original-token mapping"
        migrated["ECov_slot_reason"] = "N/A: no valid original-token mapping"
        migrated["structural_position_addressability"] = False
    signature, flags = classify_failure_signature(migrated)
    migrated["failure_signature"] = signature
    migrated["primitive_flags"] = flags
    return migrated


def migrate_summary_csv(path: Path, coverage_type_for_method) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        coverage_type = coverage_type_for_method(str(row["method_name"]))
        row["coverage_type"] = coverage_type
        if coverage_type == "structural_position_addressability":
            row["ERR_slot_mean"] = ""
            row["ECov_slot_mean"] = ""
    write_csv(path, rows)


def migrate_matched_csv(path: Path, coverage_type_for_method) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        left_type = coverage_type_for_method(str(row["method_left"]))
        right_type = coverage_type_for_method(str(row["method_right"]))
        row["left_coverage_type"] = left_type
        row["right_coverage_type"] = right_type
        if left_type == "structural_position_addressability":
            row["left_ECov_slot"] = ""
        if right_type == "structural_position_addressability":
            row["right_ECov_slot"] = ""
    write_csv(path, rows)


def migrate_context(root: Path, selected_rows: list[dict[str, Any]]) -> None:
    context_dir = root / "data/context_demand"
    path = context_dir / "ruler8k_context_demand_dataset.jsonl"
    selected = {
        (str(row["sample_id"]), str(row["method_name"]), round(float(row["retained_budget"]), 6)): row
        for row in selected_rows
        if row["dataset"] == "ruler8k"
    }
    rows = read_jsonl(path)
    for row in rows:
        key = (str(row["sample_id"]), str(row["method_name"]), round(float(row["retained_budget"]), 6))
        diagnostic = selected[key]
        for field in (
            "coverage_applicability_schema",
            "coverage_type",
            "structural_position_addressability",
        ):
            row[field] = diagnostic[field]
        for metric in ("ERR_slot", "ECov_slot"):
            row[f"{metric}_value"] = diagnostic[metric]
            row[f"{metric}_status"] = diagnostic[f"{metric}_status"]
            row[f"{metric}_reason"] = diagnostic[f"{metric}_reason"]
        row["failure_signature"] = diagnostic["failure_signature"]
    write_jsonl(path, rows)
    write_csv(context_dir / "ruler8k_context_demand_dataset.csv", rows)

    coverage_counts = Counter(row["coverage_type"] for row in rows)
    report_path = context_dir / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["slot_metrics_available"] = sum(row["ECov_slot_status"] == "available" for row in rows)
    report["coverage_types"] = dict(sorted(coverage_counts.items()))
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_path = context_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["coverage_types"] = dict(sorted(coverage_counts.items()))
    summary["slot_metrics_available"] = report["slot_metrics_available"]
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from kvcachebench.diagnostics import classify_failure_signature, coverage_type_for_method

    selected_dir = root / "data/processed/selected_failures"
    migrated_by_name: dict[str, list[dict[str, Any]]] = {}
    for name in SELECTED_FILES:
        rows = [
            migrate_row(row, coverage_type_for_method, classify_failure_signature)
            for row in read_jsonl(selected_dir / name)
        ]
        write_jsonl(selected_dir / name, rows)
        migrated_by_name[name] = rows

    for path in (
        root / "data/examples/selected_failures_sample.jsonl",
        root / "tests/fixtures/selected_failures_sample.jsonl",
    ):
        rows = [
            migrate_row(row, coverage_type_for_method, classify_failure_signature)
            for row in read_jsonl(path)
        ]
        write_jsonl(path, rows)

    all_rows = migrated_by_name["all_selected_failures.jsonl"]
    migrate_context(root, all_rows)
    migrate_summary_csv(root / "data/summaries/slot_ecov_summary.csv", coverage_type_for_method)
    migrate_matched_csv(root / "data/summaries/matched_method_pair_summary.csv", coverage_type_for_method)

    coverage_counts = Counter(row["coverage_type"] for row in all_rows)
    signature_counts = Counter(row["failure_signature"] for row in all_rows)
    print(
        json.dumps(
            {
                "rows": len(all_rows),
                "coverage_types": dict(sorted(coverage_counts.items())),
                "failure_signatures": dict(sorted(signature_counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
