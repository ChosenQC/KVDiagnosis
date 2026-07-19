from pathlib import Path

from kvcachebench.diagnostics import validate_selected_failure_rows
from kvcachebench.io import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def test_fixture_selected_failures_validate():
    rows = read_jsonl(Path(__file__).parent / "fixtures/selected_failures_sample.jsonl")
    report = validate_selected_failure_rows(rows)
    assert report["ok"], report
    assert report["rows"] == 30


def test_complete_public_corpus_matches_frozen_audit():
    rows = read_jsonl(
        ROOT / "data/processed/selected_failures/all_selected_failures.jsonl"
    )
    report = validate_selected_failure_rows(rows)
    assert report["ok"], report
    assert report["rows"] == 12520
    assert report["datasets"] == {
        "hotpotqa": 827,
        "qasper": 327,
        "ruler16k": 5396,
        "ruler8k": 5970,
    }
    assert report["attention_available"] == 7038
    assert report["topk_available"] == 8400
    assert all(row["method_name"] != "PyramidKVPress" for row in rows)
    assert all(
        not any(field in row for field in ("ERR", "ECov", "DRR"))
        for row in rows
    )
