#!/usr/bin/env python3
"""Export FullCache-correct/compressed-wrong rows from paired_metrics.jsonl files.

This script is provided for reproducibility. It expects a working KVbench-style
results directory with paired metrics and writes the compact public diagnostic
artifact used by this repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = {
    "ruler8k": "ruler8k_kvdench_qwen3_8b/paired_metrics.jsonl",
    "ruler16k": "ruler16k_kvdench_qwen3_8b/paired_metrics.jsonl",
    "longbench_v2_proxy90": "longbench_v2_proxy90_kvdench_qwen3_8b/paired_metrics.jsonl",
    "qasper_hotpot": "qasper_hotpot_evidence_kvdench_qwen3_8b/paired_metrics.jsonl",
}

METHOD_DISPLAY = {
    "SnapKVPress": "SnapKV",
    "TOVAPress": "TOVA",
    "KeyDiffPress": "KeyDiff",
    "ThinKPress": "ThinK",
    "ChunkKVPress_Knorm": "ChunkKV",
    "PyramidKVPress": "PyramidKV",
    "AdaKVPress": "AdaKV",
    "StreamingLLMPress": "StreamingLLM",
    "QuantizedCache": "QuantizedCache",
}

KEEP_FIELDS = [
    "experiment_id", "sample_id", "task_key", "official_task_name",
    "context_length_tokens", "model_id", "method_name", "method_family",
    "compression_ratio", "retained_budget", "full_run_id", "compressed_run_id",
    "prompt_hash", "reference_answer", "raw_output_full", "raw_output_compressed",
    "extracted_answer_full", "extracted_answer_compressed", "full_score",
    "compressed_score", "score_drop", "full_correct", "compressed_correct", "CIF",
    "ERR", "ECov", "DRR", "full_gold_NLL", "compressed_gold_NLL",
    "delta_NLL", "GPR", "metric_applicability_notes",
]

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    counts = {}
    for dataset, rel in DATASETS.items():
        rows = []
        for row in iter_jsonl(args.results_root / rel):
            if row.get("full_correct") is True and row.get("compressed_correct") is False:
                out = {field: row.get(field) for field in KEEP_FIELDS if field in row}
                out["dataset"] = dataset
                out["method_display"] = METHOD_DISPLAY.get(str(row.get("method_name")), str(row.get("method_name")))
                rows.append(out)
        counts[dataset] = len(rows)
        with (args.output_dir / f"{dataset}.jsonl").open("w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        all_rows.extend(rows)
    with (args.output_dir / "all_selected_failures.jsonl").open("w", encoding="utf-8") as handle:
        for item in all_rows:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"counts": counts, "total": len(all_rows)}, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
