from pathlib import Path

from kvcachebench.io import read_jsonl
from kvcachebench.diagnostics import summarize_rows

rows = read_jsonl(Path(__file__).resolve().parents[1] / "data/processed/selected_failures/all_selected_failures.jsonl")
summary = summarize_rows(rows, ["dataset", "method_name", "retained_budget"])
for row in summary[:10]:
    print(row)
