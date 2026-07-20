#!/usr/bin/env python3
"""Build the normalized, complete KVCacheBench per-source run ledger."""

from __future__ import annotations

import argparse
import gzip
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
METHOD_DISPLAY = {
    "StreamingLLMPress": "StreamingLLM",
    "SnapKVPress": "SnapKV",
    "TOVAPress": "TOVA",
    "KeyDiffPress": "KeyDiff",
    "ThinKPress": "ThinK",
    "ChunkKVPress_Knorm": "ChunkKV",
    "AdaKVPress": "AdaKV",
    "QuantizedCache": "QuantizedCache",
}
METHODS = tuple(METHOD_DISPLAY)
BUDGETS = (0.75, 0.50, 0.25)
DATASETS = {
    "ruler8k": {
        "path": "ruler8k_kvdench_qwen3_8b/paired_metrics.jsonl",
        "task": None,
        "sources": 1100,
    },
    "ruler16k": {
        "path": "ruler16k_kvdench_qwen3_8b/paired_metrics.jsonl",
        "task": None,
        "sources": 1100,
    },
    "qasper": {
        "path": "qasper_hotpot_evidence_kvdench_qwen3_8b/paired_metrics.jsonl",
        "task": "qasper",
        "sources": 200,
    },
    "hotpotqa": {
        "path": "qasper_hotpot_evidence_kvdench_qwen3_8b/paired_metrics.jsonl",
        "task": "hotpotqa",
        "sources": 200,
    },
}

FULLCACHE_SOURCE_FIELDS = (
    "sample_id",
    "task_key",
    "official_task_name",
    "context_length_tokens",
    "model_id",
    "full_run_id",
    "prompt_hash",
    "reference_answer",
    "raw_output_full",
    "extracted_answer_full",
    "full_score",
    "full_correct",
    "full_gold_NLL",
)
COMPRESSED_SOURCE_FIELDS = (
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
    "compressed_run_id",
    "prompt_hash",
    "raw_output_compressed",
    "extracted_answer_compressed",
    "compressed_score",
    "score_drop",
    "compressed_correct",
    "CIF",
    "correct_to_wrong_flip",
    "wrong_to_correct_flip",
    "compressed_gold_NLL",
    "delta_NLL",
    "GPR",
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                    count += 1
    return count


def normalize_budget(value: Any) -> float:
    return round(float(value), 6)


def na_reason(value: Any) -> str | None:
    if isinstance(value, dict) and value.get("value") == "N/A":
        return str(value.get("reason") or "N/A: unsupported method-setting")
    return None


def nullable(value: Any) -> Any:
    return None if na_reason(value) is not None else value


def transition(full_correct: bool, compressed_correct: bool) -> str:
    return ("C" if full_correct else "W") + "->" + ("C" if compressed_correct else "W")


def fullcache_projection(dataset: str, row: dict[str, Any]) -> dict[str, Any]:
    projected = {field: row.get(field) for field in FULLCACHE_SOURCE_FIELDS}
    projected.update(
        {
            "schema_version": "kvcachebench.fullcache_run.v1",
            "dataset": dataset,
            "model_revision": MODEL_REVISION,
            "fullcache_key": f"{dataset}::{row['sample_id']}",
        }
    )
    return projected


def compressed_projection(dataset: str, row: dict[str, Any]) -> dict[str, Any]:
    budget = normalize_budget(row["retained_budget"])
    method = str(row["method_name"])
    unsupported = method == "ThinKPress" and budget == 0.25
    reason = na_reason(row.get("compressed_correct")) if unsupported else None
    if unsupported and reason is None:
        raise ValueError(f"ThinK/25% row is not explicit N/A: {dataset}/{row['sample_id']}")
    if not unsupported and not isinstance(row.get("compressed_correct"), bool):
        raise ValueError(
            f"supported row lacks boolean correctness: {dataset}/{row['sample_id']}/{method}/{budget}"
        )

    projected = {field: nullable(row.get(field)) for field in COMPRESSED_SOURCE_FIELDS}
    projected.update(
        {
            "schema_version": "kvcachebench.compressed_run.v1",
            "dataset": dataset,
            "model_revision": MODEL_REVISION,
            "method_display": METHOD_DISPLAY[method],
            "fullcache_key": f"{dataset}::{row['sample_id']}",
            "support_status": "unsupported" if unsupported else "supported",
            "unsupported_reason": reason,
            "outcome_transition": None
            if unsupported
            else transition(bool(row["full_correct"]), bool(row["compressed_correct"])),
        }
    )
    return projected


def selected_key(row: dict[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(row["dataset"]),
        str(row["sample_id"]),
        str(row["method_name"]),
        normalize_budget(row["retained_budget"]),
    )


def load_selected_keys(path: Path) -> set[tuple[str, str, str, float]]:
    return {selected_key(row) for row in iter_jsonl(path)}


def build_dataset(
    dataset: str,
    spec: dict[str, Any],
    results_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fullcache: dict[str, dict[str, Any]] = {}
    compressed: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, float]] = set()
    source_path = results_root / str(spec["path"])

    for row in iter_jsonl(source_path):
        if row.get("method_name") not in METHOD_DISPLAY:
            continue
        task = spec["task"]
        if task is not None and row.get("official_task_name") != task:
            continue

        budget = normalize_budget(row["retained_budget"])
        key = (str(row["sample_id"]), str(row["method_name"]), budget)
        if key in seen_keys:
            raise ValueError(f"duplicate run key in {dataset}: {key}")
        seen_keys.add(key)

        full_row = fullcache_projection(dataset, row)
        sample_id = str(row["sample_id"])
        existing = fullcache.get(sample_id)
        if existing is not None and existing != full_row:
            raise ValueError(f"FullCache fields differ across cells: {dataset}/{sample_id}")
        fullcache[sample_id] = full_row
        compressed.append(compressed_projection(dataset, row))

    expected_sources = int(spec["sources"])
    expected_runs = expected_sources * len(METHODS) * len(BUDGETS)
    if len(fullcache) != expected_sources:
        raise ValueError(f"{dataset}: {len(fullcache)} sources != {expected_sources}")
    if len(compressed) != expected_runs:
        raise ValueError(f"{dataset}: {len(compressed)} runs != {expected_runs}")

    cell_counts = Counter(
        (row["method_name"], normalize_budget(row["retained_budget"]))
        for row in compressed
    )
    expected_cells = {(method, budget) for method in METHODS for budget in BUDGETS}
    if set(cell_counts) != expected_cells or any(
        count != expected_sources for count in cell_counts.values()
    ):
        raise ValueError(f"{dataset}: incomplete method-setting cells: {cell_counts}")

    return (
        sorted(fullcache.values(), key=lambda row: str(row["sample_id"])),
        sorted(
            compressed,
            key=lambda row: (
                str(row["sample_id"]),
                str(row["method_name"]),
                -float(row["retained_budget"]),
            ),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed/full_population",
    )
    parser.add_argument(
        "--selected-failures",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data/processed/selected_failures/all_selected_failures.jsonl",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_fullcache: list[dict[str, Any]] = []
    all_compressed: list[dict[str, Any]] = []
    dataset_summary: dict[str, Any] = {}

    for dataset, spec in DATASETS.items():
        fullcache, compressed = build_dataset(dataset, spec, args.results_root)
        all_fullcache.extend(fullcache)
        all_compressed.extend(compressed)
        output_path = args.output_dir / "compressed_runs" / f"{dataset}.jsonl.gz"
        write_jsonl_gz(output_path, compressed)
        dataset_summary[dataset] = {
            "sources": len(fullcache),
            "planned_runs": len(compressed),
            "supported_runs": sum(row["support_status"] == "supported" for row in compressed),
            "unsupported_runs": sum(row["support_status"] == "unsupported" for row in compressed),
            "fullcache_correct_sources": sum(bool(row["full_correct"]) for row in fullcache),
            "C_to_W_rows": sum(row["outcome_transition"] == "C->W" for row in compressed),
        }

    all_fullcache.sort(key=lambda row: (str(row["dataset"]), str(row["sample_id"])))
    write_jsonl_gz(args.output_dir / "fullcache.jsonl.gz", all_fullcache)

    supported = [row for row in all_compressed if row["support_status"] == "supported"]
    c_to_w = {selected_key(row) for row in supported if row["outcome_transition"] == "C->W"}
    selected = load_selected_keys(args.selected_failures)
    if c_to_w != selected:
        raise ValueError(
            f"selected C->W mismatch: ledger_only={len(c_to_w - selected)}, "
            f"selected_only={len(selected - c_to_w)}"
        )

    transitions = Counter(row["outcome_transition"] for row in supported)
    fullcache_correct_keys = {
        full["fullcache_key"] for full in all_fullcache if full["full_correct"]
    }
    summary = {
        "schema_version": "kvcachebench.full_population_summary.v1",
        "model": "Qwen/Qwen3-8B",
        "model_revision": MODEL_REVISION,
        "source_count": len(all_fullcache),
        "planned_compressed_runs": len(all_compressed),
        "supported_compressed_runs": len(supported),
        "unsupported_compressed_runs": len(all_compressed) - len(supported),
        "fullcache_correct_eligible_pairs": sum(
            row["fullcache_key"] in fullcache_correct_keys for row in supported
        ),
        "transitions": dict(sorted(transitions.items())),
        "selected_C_to_W_rows": len(c_to_w),
        "datasets": dataset_summary,
        "methods": list(METHODS),
        "retained_budgets": list(BUDGETS),
        "unsupported_setting": {"method": "ThinKPress", "retained_budget": 0.25},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
