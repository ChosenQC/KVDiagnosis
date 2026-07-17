from pathlib import Path

from kvcachebench.diagnostics import validate_selected_failure_rows
from kvcachebench.io import read_jsonl


def test_fixture_selected_failures_validate():
    rows = read_jsonl(Path(__file__).parent / "fixtures/selected_failures_sample.jsonl")
    report = validate_selected_failure_rows(rows)
    assert report["ok"], report
    assert report["rows"] > 0
