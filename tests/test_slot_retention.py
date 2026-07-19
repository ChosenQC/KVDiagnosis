from kvcachebench.retention import slot_retention_metrics


def test_cross_slot_union_does_not_create_perfect_coverage():
    records = {
        "0": [[0], [1]],
        "1": [[2], [3]],
    }
    metrics = slot_retention_metrics(
        records,
        evidence_spans=[[0, 4]],
        distractor_spans=[],
        prompt_len=8,
        ecov_threshold=0.5,
    )
    assert metrics["ERR_slot"] == 0.25
    assert metrics["ECov_slot"] == 0.0
    assert metrics["retention_slot_count"] == 4


def test_span_coverage_is_distinct_from_token_retention():
    records = {"0": [[0, 1, 4, 5], [0, 4]]}
    metrics = slot_retention_metrics(
        records,
        evidence_spans=[[0, 2], [4, 6]],
        distractor_spans=[],
        prompt_len=8,
        ecov_threshold=1.0,
    )
    assert metrics["ERR_slot"] == 0.75
    assert metrics["ECov_slot"] == 0.5


def test_distractor_ratio_is_bounded():
    metrics = slot_retention_metrics(
        {"0": [[0, 1, 4, 5], [0, 4]]},
        evidence_spans=[[0, 2]],
        distractor_spans=[[4, 6]],
        prompt_len=8,
    )
    assert 0.0 <= metrics["DRR_slot"] <= 1.0
