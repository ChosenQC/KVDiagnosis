#!/usr/bin/env python3
"""Audit raw SnapKV/PyramidKV pairs produced by the invalid tracker adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


RUN_PATTERN = re.compile(
    r"^(?P<sample>.+)__(?P<method>SnapKVPress|PyramidKVPress)__cr"
    r"(?P<ratio>[0-9.]+)\.done\.json$"
)
METHODS = ("SnapKVPress", "PyramidKVPress")
SEMANTIC_FIELDS = (
    "experiment_id",
    "sample_id",
    "task_key",
    "official_task_name",
    "context_length_tokens",
    "model_id",
    "compression_ratio",
    "retained_budget",
    "prompt_hash",
    "reference_answer",
    "raw_output",
    "extracted_answer",
    "score",
    "f1",
    "correct",
    "gold_NLL",
    "ERR",
    "ECov",
    "DRR",
    "retained_original_positions_or_unit_mapping",
    "metric_applicability_notes",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten_indices(value: Any) -> list[int]:
    if isinstance(value, list):
        result: list[int] = []
        for item in value:
            result.extend(flatten_indices(item))
        return result
    if isinstance(value, int):
        return [value]
    return []


def parse_run_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run roots must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).resolve()
    if not label or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid run root: {value}")
    return label, path


def index_runs(root: Path) -> dict[tuple[str, float], dict[str, Path]]:
    indexed: dict[tuple[str, float], dict[str, Path]] = {}
    for path in root.glob("*.done.json"):
        match = RUN_PATTERN.match(path.name)
        if match is None:
            continue
        key = (match.group("sample"), round(float(match.group("ratio")), 6))
        indexed.setdefault(key, {})[match.group("method")] = path
    incomplete = [key for key, methods in indexed.items() if set(methods) != set(METHODS)]
    if incomplete:
        raise ValueError(f"{root}: {len(incomplete)} incomplete pairs; examples={incomplete[:5]}")
    return indexed


def class_record(cls: type, config: dict[str, Any]) -> dict[str, Any]:
    source = Path(inspect.getsourcefile(cls) or "")
    return {
        "class": cls.__name__,
        "module": cls.__module__,
        "constructor_signature": str(inspect.signature(cls)),
        "config": config,
        "source_file": source.name,
        "source_sha256": digest_file(source),
    }


def provenance(adapter_source: Path | None, corrected_tracker_source: Path, source_commit: str) -> dict[str, Any]:
    import kvpress

    record = {
        "kvpress_version": importlib.metadata.version("kvpress"),
        "kvpress_release_source_commit": source_commit,
        "implementations": {
            "SnapKVPress": class_record(
                kvpress.SnapKVPress,
                {"compression_ratio": "{0.25,0.50,0.75}", "window_size": 64, "kernel_size": 5},
            ),
            "PyramidKVPress": class_record(
                kvpress.PyramidKVPress,
                {
                    "compression_ratio": "{0.25,0.50,0.75}",
                    "window_size": 64,
                    "kernel_size": 5,
                    "beta": 20,
                },
            ),
        },
        "corrected_public_tracker": {
            "source_file": corrected_tracker_source.name,
            "source_sha256": digest_file(corrected_tracker_source),
            "pyramid_budget_call": "PyramidKVPress.get_layer_budget(module, k_len)",
        },
    }
    if adapter_source is not None:
        record["invalid_experiment_adapter"] = {
            "source_file": adapter_source.name,
            "source_sha256": digest_file(adapter_source),
        }
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", type=parse_run_root, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-source", type=Path)
    parser.add_argument(
        "--corrected-tracker-source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src/kvcachebench/presses.py",
    )
    parser.add_argument("--kvpress-source-commit", default="fa7a0dc")
    args = parser.parse_args()

    equality_fields = (*SEMANTIC_FIELDS, "semantic_payload")
    overall = Counter({field: 0 for field in equality_fields})
    total_pairs = 0
    distinct_job_pairs = 0
    distinct_node_pairs = 0
    datasets: dict[str, Any] = {}
    examples: list[dict[str, Any]] = []

    for label, run_root in args.run_root:
        indexed = index_runs(run_root)
        dataset_counts = Counter({field: 0 for field in equality_fields})
        ratios = Counter()
        example_by_ratio: set[float] = set()
        for key in sorted(indexed):
            sample_id, compression_ratio = key
            paths = indexed[key]
            snap = json.loads(paths["SnapKVPress"].read_text(encoding="utf-8"))
            pyramid = json.loads(paths["PyramidKVPress"].read_text(encoding="utf-8"))
            total_pairs += 1
            ratios[f"{compression_ratio:.2f}"] += 1
            for field in SEMANTIC_FIELDS:
                if snap.get(field) == pyramid.get(field):
                    overall[field] += 1
                    dataset_counts[field] += 1
            semantic_equal = all(snap.get(field) == pyramid.get(field) for field in SEMANTIC_FIELDS)
            if semantic_equal:
                overall["semantic_payload"] += 1
                dataset_counts["semantic_payload"] += 1

            snap_meta = snap.get("metadata") or {}
            pyramid_meta = pyramid.get("metadata") or {}
            if snap_meta.get("slurm_job_id") != pyramid_meta.get("slurm_job_id"):
                distinct_job_pairs += 1
            if snap_meta.get("node") != pyramid_meta.get("node"):
                distinct_node_pairs += 1

            if compression_ratio not in example_by_ratio:
                snap_mapping = snap.get("retained_original_positions_or_unit_mapping")
                pyramid_mapping = pyramid.get("retained_original_positions_or_unit_mapping")
                snap_flat = flatten_indices(snap_mapping)
                pyramid_flat = flatten_indices(pyramid_mapping)
                examples.append(
                    {
                        "dataset": label,
                        "sample_id": sample_id,
                        "compression_ratio": compression_ratio,
                        "retained_budget": snap.get("retained_budget"),
                        "SnapKVPress": {
                            "retained_index_sha256": digest_value(snap_mapping),
                            "retained_index_count_with_repetition": len(snap_flat),
                            "retained_index_preview": snap_flat[:8],
                            "retained_index_tail": snap_flat[-8:],
                            "output_sha256": digest_value(snap.get("raw_output")),
                        },
                        "PyramidKVPress": {
                            "retained_index_sha256": digest_value(pyramid_mapping),
                            "retained_index_count_with_repetition": len(pyramid_flat),
                            "retained_index_preview": pyramid_flat[:8],
                            "retained_index_tail": pyramid_flat[-8:],
                            "output_sha256": digest_value(pyramid.get("raw_output")),
                        },
                        "retained_indices_equal": snap_mapping == pyramid_mapping,
                        "outputs_equal": snap.get("raw_output") == pyramid.get("raw_output"),
                    }
                )
                example_by_ratio.add(compression_ratio)

        pair_count = len(indexed)
        datasets[label] = {
            "pair_count": pair_count,
            "pairs_by_compression_ratio": dict(sorted(ratios.items())),
            "equality": {
                field: {"equal_pairs": dataset_counts[field], "rate": dataset_counts[field] / pair_count}
                for field in equality_fields
            },
        }

    result = {
        "schema_version": "kvdiagnosis.pyramidkv_adapter_audit.v1",
        "audit_status": "invalid_adapter_confirmed_rows_excluded",
        "root_cause": (
            "The retained-index tracker replaced PyramidKVPress.compress with a generic scorer "
            "compressor that used one fixed int(k_len * (1 - compression_ratio)) budget. This "
            "bypassed PyramidKVPress.get_layer_budget and made the executed path identical to SnapKV."
        ),
        "pair_count": total_pairs,
        "datasets": datasets,
        "overall_equality": {
            field: {"equal_pairs": overall[field], "rate": overall[field] / total_pairs}
            for field in equality_fields
        },
        "execution_independence": {
            "different_slurm_job_id_pairs": distinct_job_pairs,
            "different_slurm_job_id_rate": distinct_job_pairs / total_pairs,
            "different_node_pairs": distinct_node_pairs,
            "different_node_rate": distinct_node_pairs / total_pairs,
        },
        "provenance": provenance(
            args.adapter_source.resolve() if args.adapter_source else None,
            args.corrected_tracker_source.resolve(),
            args.kvpress_source_commit,
        ),
        "representative_pairs": examples,
        "disposition": (
            "All PyramidKV rows from this adapter are invalid method-level measurements. They are "
            "excluded from the released eight-method matrix and every denominator is recomputed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pair_count": total_pairs, "overall_equality": result["overall_equality"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
