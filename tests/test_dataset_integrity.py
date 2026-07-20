import json
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
    assert report["coverage_types"] == {
        "measured_token_coverage": 6211,
        "projected_token_coverage": 2224,
        "structural_position_addressability": 4085,
    }
    assert report["failure_signatures"] == {
        "ambiguous": 397,
        "conflicting_diagnostic_signals": 1556,
        "decode_scorer_candidate": 405,
        "high_mapped_coverage_likelihood_drift": 19,
        "low_ear_candidate": 104,
        "low_mapped_coverage": 5047,
        "partial_mapped_coverage": 2866,
        "structural_position_likelihood_drift": 2126,
    }
    assert all(row["method_name"] != "PyramidKVPress" for row in rows)
    assert all(
        not any(field in row for field in ("ERR", "ECov", "DRR"))
        for row in rows
    )
    structural = [
        row
        for row in rows
        if row["coverage_type"] == "structural_position_addressability"
    ]
    assert all(row["ERR_slot"] is None and row["ECov_slot"] is None for row in structural)
    assert all(
        row["ERR_slot_status"] == row["ECov_slot_status"] == "not_applicable"
        for row in structural
    )


def test_pyramidkv_adapter_audit_is_complete_and_excluded():
    audit = json.loads(
        (ROOT / "data/audits/pyramidkv_adapter_audit.json").read_text(encoding="utf-8")
    )
    assert audit["audit_status"] == "invalid_adapter_confirmed_rows_excluded"
    assert audit["pair_count"] == 7800
    assert audit["execution_independence"]["different_slurm_job_id_pairs"] == 7800
    for field in (
        "retained_original_positions_or_unit_mapping",
        "raw_output",
        "extracted_answer",
        "score",
        "gold_NLL",
        "semantic_payload",
    ):
        assert audit["overall_equality"][field] == {
            "equal_pairs": 7800,
            "rate": 1.0,
        }
    assert (
        audit["provenance"]["corrected_public_tracker"]["pyramid_budget_call"]
        == "PyramidKVPress.get_layer_budget(module, k_len)"
    )
