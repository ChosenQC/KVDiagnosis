from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

SELECTED_FAILURE_REQUIRED_FIELDS = [
    "dataset",
    "sample_id",
    "method_name",
    "compression_ratio",
    "retained_budget",
    "full_correct",
    "compressed_correct",
]

NUMERIC_METRICS = [
    "full_score",
    "compressed_score",
    "score_drop",
    "ERR",
    "ECov",
    "DRR",
    "full_gold_NLL",
    "compressed_gold_NLL",
    "delta_NLL",
    "GPR",
]


def is_na(value: Any) -> bool:
    return isinstance(value, dict) and value.get("value") == "N/A"


def numeric(value: Any) -> float | None:
    if is_na(value):
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def mean(values: list[Any]) -> float | None:
    nums = [numeric(value) for value in values]
    nums = [value for value in nums if value is not None]
    return sum(nums) / len(nums) if nums else None


def validate_selected_failure_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    keys_seen: set[tuple[Any, ...]] = set()
    by_dataset: dict[str, int] = defaultdict(int)
    for index, row in enumerate(rows, start=1):
        missing = [field for field in SELECTED_FAILURE_REQUIRED_FIELDS if field not in row]
        if missing:
            errors.append(f"row {index}: missing fields {missing}")
        if row.get("full_correct") is not True:
            errors.append(f"row {index}: full_correct is not true")
        if row.get("compressed_correct") is not False:
            errors.append(f"row {index}: compressed_correct is not false")
        key = (
            row.get("dataset"),
            row.get("sample_id"),
            row.get("method_name"),
            row.get("compression_ratio"),
        )
        if key in keys_seen:
            errors.append(f"row {index}: duplicate dataset/sample/method/compression key {key}")
        keys_seen.add(key)
        by_dataset[str(row.get("dataset"))] += 1
    return {
        "ok": not errors,
        "rows": len(rows),
        "unique_source_samples": len({(row.get("dataset"), row.get("sample_id")) for row in rows}),
        "datasets": dict(sorted(by_dataset.items())),
        "errors": errors[:50],
        "error_count": len(errors),
    }


def summarize_rows(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rec = dict(zip(group_fields, key))
        rec["failure_rows"] = len(items)
        rec["unique_source_samples"] = len({item.get("sample_id") for item in items})
        for metric in NUMERIC_METRICS:
            vals = [item.get(metric) for item in items]
            rec[f"{metric}_mean"] = mean(vals)
            rec[f"{metric}_valid_n"] = sum(1 for value in vals if numeric(value) is not None)
        out.append(rec)
    return out
