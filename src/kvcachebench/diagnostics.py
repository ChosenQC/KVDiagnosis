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
    "retention_metric_schema",
    "retention_semantics",
    "coverage_applicability_schema",
    "coverage_type",
    "ERR_slot",
    "ERR_slot_status",
    "ERR_slot_reason",
    "ECov_slot",
    "ECov_slot_status",
    "ECov_slot_reason",
    "ECov_slot_threshold",
    "structural_position_addressability",
    "failure_signature",
    "attention_available",
    "topk_available",
]

NUMERIC_METRICS = [
    "full_score",
    "compressed_score",
    "score_drop",
    "ERR_slot",
    "ECov_slot",
    "full_gold_NLL",
    "compressed_gold_NLL",
    "delta_NLL",
    "GPR",
    "EAR",
    "KL",
    "TopK",
    "gold_rank_shift",
]

FAILURE_SIGNATURES = {
    "low_mapped_coverage",
    "partial_mapped_coverage",
    "high_mapped_coverage_likelihood_drift",
    "structural_position_likelihood_drift",
    "low_ear_candidate",
    "decode_scorer_candidate",
    "conflicting_diagnostic_signals",
    "ambiguous",
}

RELEASE_METHODS = {
    "StreamingLLMPress",
    "SnapKVPress",
    "TOVAPress",
    "KeyDiffPress",
    "ThinKPress",
    "ChunkKVPress_Knorm",
    "AdaKVPress",
    "QuantizedCache",
}

RETENTION_SEMANTICS = {
    "per_layer_per_kv_head_original_position_mapping",
    "all_positions_structurally_preserved",
}

COVERAGE_TYPES = {
    "measured_token_coverage",
    "projected_token_coverage",
    "structural_position_addressability",
    "not_applicable",
}

MEASURED_COVERAGE_METHODS = {
    "StreamingLLMPress",
    "SnapKVPress",
    "TOVAPress",
    "KeyDiffPress",
    "AdaKVPress",
}
PROJECTED_COVERAGE_METHODS = {"ChunkKVPress_Knorm"}
STRUCTURAL_COVERAGE_METHODS = {"ThinKPress", "QuantizedCache"}
MAPPED_COVERAGE_TYPES = {"measured_token_coverage", "projected_token_coverage"}


def is_na(value: Any) -> bool:
    return isinstance(value, dict) and value.get("value") == "N/A"


def numeric(value: Any) -> float | None:
    if is_na(value) or value is None:
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


def coverage_type_for_method(method_name: str) -> str:
    if method_name in MEASURED_COVERAGE_METHODS:
        return "measured_token_coverage"
    if method_name in PROJECTED_COVERAGE_METHODS:
        return "projected_token_coverage"
    if method_name in STRUCTURAL_COVERAGE_METHODS:
        return "structural_position_addressability"
    return "not_applicable"


def classify_failure_signature(row: dict[str, Any]) -> tuple[str, list[str]]:
    coverage_type = str(row.get("coverage_type"))
    ecov = numeric(row.get("ECov_slot"))
    delta_nll = numeric(row.get("delta_NLL"))
    ear = numeric(row.get("EAR"))
    topk = numeric(row.get("TopK"))
    flags: list[str] = []

    if coverage_type in MAPPED_COVERAGE_TYPES and ecov is not None:
        if ecov < 0.50:
            flags.append("low_mapped_coverage")
        elif ecov < 0.90:
            flags.append("partial_mapped_coverage")
        elif delta_nll is not None and delta_nll >= 1.0:
            flags.append("high_mapped_coverage_likelihood_drift")

    if (
        coverage_type == "structural_position_addressability"
        and delta_nll is not None
        and delta_nll >= 1.0
    ):
        flags.append("structural_position_likelihood_drift")

    positions_addressable = coverage_type == "structural_position_addressability" or (
        coverage_type in MAPPED_COVERAGE_TYPES
        and ecov is not None
        and ecov >= 0.90
    )
    if positions_addressable and row.get("attention_available") is True and ear is not None and ear < 0.50:
        flags.append("low_ear_candidate")
    if positions_addressable and delta_nll is not None and abs(delta_nll) <= 0.10:
        if row.get("topk_available") is not True or (topk is not None and topk >= 0.90):
            flags.append("decode_scorer_candidate")

    if not flags:
        return "ambiguous", flags
    if len(flags) > 1:
        return "conflicting_diagnostic_signals", flags
    return flags[0], flags


def validate_selected_failure_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    keys_seen: set[tuple[Any, ...]] = set()
    by_dataset: dict[str, int] = defaultdict(int)
    by_signature: dict[str, int] = defaultdict(int)
    by_coverage_type: dict[str, int] = defaultdict(int)

    for index, row in enumerate(rows, start=1):
        missing = [field for field in SELECTED_FAILURE_REQUIRED_FIELDS if field not in row]
        if missing:
            errors.append(f"row {index}: missing fields {missing}")
        if row.get("full_correct") is not True:
            errors.append(f"row {index}: full_correct is not true")
        if row.get("compressed_correct") is not False:
            errors.append(f"row {index}: compressed_correct is not false")
        if row.get("method_name") not in RELEASE_METHODS:
            errors.append(f"row {index}: method is not in the release set: {row.get('method_name')!r}")
        if row.get("retention_metric_schema") != "kvbench.slot_ecov.v1":
            errors.append(f"row {index}: unexpected retention metric schema")
        if row.get("coverage_applicability_schema") != "kvdiagnosis.coverage.v2":
            errors.append(f"row {index}: unexpected coverage applicability schema")
        if row.get("retention_semantics") not in RETENTION_SEMANTICS:
            errors.append(f"row {index}: unexpected retention semantics")
        coverage_type = str(row.get("coverage_type"))
        if coverage_type not in COVERAGE_TYPES:
            errors.append(f"row {index}: unexpected coverage type {coverage_type!r}")
        expected_coverage_type = coverage_type_for_method(str(row.get("method_name")))
        if coverage_type != expected_coverage_type:
            errors.append(
                f"row {index}: coverage type {coverage_type!r} does not match method; "
                f"expected {expected_coverage_type!r}"
            )
        if numeric(row.get("ECov_slot_threshold")) != 0.5:
            errors.append(f"row {index}: ECov_slot_threshold is not 0.5")
        if coverage_type in MAPPED_COVERAGE_TYPES:
            for field in ("ERR_slot", "ECov_slot"):
                value = numeric(row.get(field))
                if value is None or not 0.0 <= value <= 1.0:
                    errors.append(f"row {index}: {field} is not in [0, 1]")
                if row.get(f"{field}_status") != "available":
                    errors.append(f"row {index}: {field} is not marked available")
                if row.get(f"{field}_reason") is not None:
                    errors.append(f"row {index}: available {field} has an N/A reason")
            if row.get("structural_position_addressability") is not False:
                errors.append(f"row {index}: mapped coverage is marked structural")
        elif coverage_type == "structural_position_addressability":
            for field in ("ERR_slot", "ECov_slot"):
                if row.get(field) is not None:
                    errors.append(f"row {index}: structural {field} must be null")
                if row.get(f"{field}_status") != "not_applicable":
                    errors.append(f"row {index}: structural {field} must be N/A")
                if not str(row.get(f"{field}_reason") or "").startswith("N/A:"):
                    errors.append(f"row {index}: structural {field} lacks an N/A reason")
            if row.get("structural_position_addressability") is not True:
                errors.append(f"row {index}: structural addressability is not true")
        signature = str(row.get("failure_signature"))
        if signature not in FAILURE_SIGNATURES:
            errors.append(f"row {index}: unknown failure signature {signature!r}")
        expected_signature, expected_flags = classify_failure_signature(row)
        if signature != expected_signature:
            errors.append(
                f"row {index}: failure signature {signature!r} != {expected_signature!r}"
            )
        if row.get("primitive_flags") != expected_flags:
            errors.append(f"row {index}: primitive flags do not match frozen rules")
        if not isinstance(row.get("attention_available"), bool):
            errors.append(f"row {index}: attention_available is not boolean")
        if not isinstance(row.get("topk_available"), bool):
            errors.append(f"row {index}: topk_available is not boolean")
        if any(field in row for field in ("ERR", "ECov", "DRR")):
            errors.append(f"row {index}: deprecated cross-slot cache metrics are present")

        row_key = (
            row.get("dataset"),
            row.get("sample_id"),
            row.get("method_name"),
            row.get("compression_ratio"),
        )
        if row_key in keys_seen:
            errors.append(f"row {index}: duplicate dataset/sample/method/compression key {row_key}")
        keys_seen.add(row_key)
        by_dataset[str(row.get("dataset"))] += 1
        by_signature[signature] += 1
        by_coverage_type[coverage_type] += 1

    return {
        "ok": not errors,
        "rows": len(rows),
        "unique_source_samples": len(
            {(row.get("dataset"), row.get("sample_id")) for row in rows}
        ),
        "datasets": dict(sorted(by_dataset.items())),
        "failure_signatures": dict(sorted(by_signature.items())),
        "coverage_types": dict(sorted(by_coverage_type.items())),
        "attention_available": sum(row.get("attention_available") is True for row in rows),
        "topk_available": sum(row.get("topk_available") is True for row in rows),
        "errors": errors[:50],
        "error_count": len(errors),
    }


def summarize_rows(
    rows: list[dict[str, Any]], group_fields: list[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(
        groups.items(), key=lambda pair: tuple(str(value) for value in pair[0])
    ):
        record = dict(zip(group_fields, key))
        record["failure_rows"] = len(items)
        record["unique_source_samples"] = len(
            {item.get("sample_id") for item in items}
        )
        for metric in NUMERIC_METRICS:
            values = [item.get(metric) for item in items]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_valid_n"] = sum(
                1 for value in values if numeric(value) is not None
            )
        out.append(record)
    return out
