# Reproduction Notes

## Analysis-Only Reproduction

The public repository can be used without GPUs:

```bash
pip install -e .
kvcachebench validate data/processed/selected_failures/all_selected_failures.jsonl
kvcachebench summarize data/processed/selected_failures/all_selected_failures.jsonl       --group-by dataset,method_name,retained_budget       --output results/selected_failures_summary.csv
```

## Full Inference Reproduction

Full reruns require a GPU environment capable of loading Qwen3-8B and the
pinned compression stack. The paper environment used `kvpress==0.5.3`,
PyTorch/CUDA builds for H200 GPUs, deterministic decoding, and the same prompt
and scoring rules for paired FullCache/compressed-cache runs.

Original prompt construction depends on upstream datasets:

- RULER: generated from NVIDIA/RULER.
- Qasper and HotpotQA: use official evidence/support annotations.
- LongBench V2 proxy90: uses a curated proxy-evidence pilot and should not be
  interpreted as a full LongBench evidence benchmark.

To regenerate the compact public diagnostic corpus from full paired metrics:

```bash
python scripts/export_selected_failures.py       --results-root /path/to/KVbench/results       --output-dir data/processed/selected_failures
```

The script expects the same dataset slugs used in the paper artifacts:
`ruler8k_kvdench_qwen3_8b`, `ruler16k_kvdench_qwen3_8b`,
`longbench_v2_proxy90_kvdench_qwen3_8b`, and
`qasper_hotpot_evidence_kvdench_qwen3_8b`.
