#!/usr/bin/env python3
"""Export the audited FullCache-correct/compressed-wrong public corpus."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DATASETS = {
    "ruler8k": {
        "path": "ruler8k_kvdench_qwen3_8b/paired_metrics.jsonl",
        "task": None,
    },
    "ruler16k": {
        "path": "ruler16k_kvdench_qwen3_8b/paired_metrics.jsonl",
        "task": None,
    },
    "qasper": {
        "path": "qasper_hotpot_evidence_kvdench_qwen3_8b/paired_metrics.jsonl",
        "task": "qasper",
    },
    "hotpotqa": {
        "path": "qasper_hotpot_evidence_kvdench_qwen3_8b/paired_metrics.jsonl",
        "task": "hotpotqa",
    },
}

MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

METHOD_DISPLAY = {
    "SnapKVPress": "SnapKV",
    "TOVAPress": "TOVA",
    "KeyDiffPress": "KeyDiff",
    "ThinKPress": "ThinK",
    "ChunkKVPress_Knorm": "ChunkKV",
    "AdaKVPress": "AdaKV",
    "StreamingLLMPress": "StreamingLLM",
    "QuantizedCache": "QuantizedCache",
}

REGIME_PUBLIC_NAMES = {
    "evidence_deletion": "low_slot_coverage",
    "partial_evidence": "partial_slot_coverage",
    "representation_drift": "high_coverage_likelihood_drift",
    "access_loss": "low_ear_candidate",
    "decode_scorer": "decode_scorer_candidate",
    "conflicting_retained_signals": "conflicting_retained_signals",
    "ambiguous": "ambiguous",
}

KEEP_FIELDS = [
    "experiment_id",
    "sample_id",
    "task_key",
    "official_task_name",
    "context_length_tokens",
    "model_id",
    "method_name",
    "method_family",
    "compression_ratio",
    "retained_budget",
    "full_run_id",
    "compressed_run_id",
    "prompt_hash",
    "reference_answer",
    "raw_output_full",
    "raw_output_compressed",
    "extracted_answer_full",
    "extracted_answer_compressed",
    "full_score",
    "compressed_score",
    "score_drop",
    "full_correct",
    "compressed_correct",
    "CIF",
    "full_gold_NLL",
    "compressed_gold_NLL",
    "delta_NLL",
    "GPR",
]

DIAGNOSTIC_FLOAT_FIELDS = [
    "ERR_slot",
    "ECov_slot",
    "delta_NLL",
    "GPR",
    "EAR",
    "KL",
    "TopK",
    "gold_rank_shift",
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def normalize_budget(value: Any) -> float:
    return round(float(value), 6)


def key(dataset: str, sample_id: Any, method: Any, budget: Any) -> tuple[str, str, str, float]:
    return dataset, str(sample_id), str(method), normalize_budget(budget)


def load_audited_rows(path: Path) -> dict[tuple[str, str, str, float], dict[str, Any]]:
    rows: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            dataset = str(raw["dataset"]).strip().lower().replace("-", "")
            if dataset in {"ruler8k", "ruler16k", "qasper", "hotpotqa"}:
                public_dataset = dataset
            else:
                raise ValueError(f"unsupported audited dataset: {raw['dataset']!r}")
            row_key = key(
                public_dataset,
                raw["sample_id"],
                raw["method_name"],
                raw["retained_budget"],
            )
            if row_key in rows:
                raise ValueError(f"duplicate audited key: {row_key}")
            regime = str(raw["regime"])
            if regime not in REGIME_PUBLIC_NAMES:
                raise ValueError(f"unknown failure regime: {regime}")
            primitive_flags = [
                REGIME_PUBLIC_NAMES[item]
                for item in str(raw.get("primitive_flags") or "").split(";")
                if item
            ]
            out: dict[str, Any] = {
                "retention_metric_schema": "kvbench.slot_ecov.v1",
                "retention_semantics": raw["retention_semantics"],
                "ECov_slot_threshold": 0.5,
                "failure_signature": REGIME_PUBLIC_NAMES[regime],
                "primitive_flags": primitive_flags,
                "attention_available": parse_bool(raw["attention_available"]),
                "topk_available": parse_bool(raw["topk_available"]),
            }
            for field in DIAGNOSTIC_FLOAT_FIELDS:
                out[field] = parse_float(raw.get(field))
            rows[row_key] = out
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--audit-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audited = load_audited_rows(args.audit_rows)
    matched: set[tuple[str, str, str, float]] = set()
    all_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for dataset, spec in DATASETS.items():
        rows: list[dict[str, Any]] = []
        for row in iter_jsonl(args.results_root / str(spec["path"])):
            task = spec["task"]
            if task is not None and row.get("official_task_name") != task:
                continue
            row_key = key(
                dataset,
                row.get("sample_id"),
                row.get("method_name"),
                row.get("retained_budget"),
            )
            diagnostic = audited.get(row_key)
            if diagnostic is None:
                continue
            if row.get("full_correct") is not True or row.get("compressed_correct") is not False:
                raise ValueError(f"audited key is not C->W in paired metrics: {row_key}")
            if row.get("method_name") not in METHOD_DISPLAY:
                raise ValueError(f"audited key uses a non-release method: {row_key}")
            out = {field: row.get(field) for field in KEEP_FIELDS if field in row}
            out["dataset"] = dataset
            out["model_revision"] = MODEL_REVISION
            out["method_display"] = METHOD_DISPLAY[str(row["method_name"])]
            out.update(diagnostic)
            rows.append(out)
            matched.add(row_key)

        rows.sort(
            key=lambda item: (
                str(item["sample_id"]),
                str(item["method_name"]),
                float(item["retained_budget"]),
            )
        )
        counts[dataset] = len(rows)
        with (args.output_dir / f"{dataset}.jsonl").open("w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        all_rows.extend(rows)

    missing = sorted(set(audited) - matched)
    if missing:
        raise ValueError(
            f"{len(missing)} audited keys were not found in paired metrics; examples: {missing[:5]}"
        )
    if len(all_rows) != len(audited):
        raise ValueError(f"exported {len(all_rows)} rows for {len(audited)} audited keys")

    all_rows.sort(
        key=lambda item: (
            str(item["dataset"]),
            str(item["sample_id"]),
            str(item["method_name"]),
            float(item["retained_budget"]),
        )
    )
    with (args.output_dir / "all_selected_failures.jsonl").open("w", encoding="utf-8") as handle:
        for item in all_rows:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps({"counts": counts, "total": len(all_rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
