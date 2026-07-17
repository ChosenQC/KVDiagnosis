from kvcachebench.scoring import score_prediction


def test_exact_string_match():
    result = score_prediction("The code is RLHZW.", ["RLHZW"])
    assert result["correct"] is True
    assert result["score"] == 1.0


def test_multichoice_extraction():
    result = score_prediction("Answer: C", ["C"])
    assert result["correct"] is True
    assert result["extracted_answer"] == "C"
