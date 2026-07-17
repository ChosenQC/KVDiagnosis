from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MULTICHOICE_RE = re.compile(r"(?:answer\s*(?:is|:)?\s*)?\b([ABCD])\b", re.IGNORECASE)


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    return " ".join(text.split())


def token_f1(prediction: str, references: list[str]) -> float:
    pred = TOKEN_RE.findall(normalize(prediction))
    if not pred:
        return 0.0
    best = 0.0
    for ref in references:
        gold = TOKEN_RE.findall(normalize(ref))
        if not gold:
            continue
        common = Counter(pred) & Counter(gold)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred)
        recall = overlap / len(gold)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def string_match_fraction(prediction: str, references: list[str]) -> float:
    pred = normalize(prediction)
    refs = [normalize(ref) for ref in references if normalize(ref)]
    if not refs:
        return 0.0
    return sum(1.0 for ref in refs if ref in pred) / len(refs)


def extract_answer(prediction: str, references: list[str]) -> str:
    text = (prediction or "").strip()
    if not text:
        return ""
    refs = [ref for ref in references if ref]
    for ref in refs:
        if normalize(ref) in normalize(text):
            return ref
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first[:256]


def is_multichoice_references(references: list[str]) -> bool:
    cleaned = [normalize(ref).upper() for ref in references if normalize(ref)]
    return bool(cleaned) and all(ref in {"A", "B", "C", "D"} for ref in cleaned)


def extract_multichoice_answer(prediction: str) -> str:
    text = (prediction or "").strip()
    if not text:
        return ""
    for line in text.splitlines():
        match = MULTICHOICE_RE.search(line.strip())
        if match:
            return match.group(1).upper()
    match = MULTICHOICE_RE.search(text[:256])
    return match.group(1).upper() if match else ""


def score_prediction(prediction: str, references: list[str]) -> dict[str, Any]:
    if is_multichoice_references(references):
        pred_choice = extract_multichoice_answer(prediction)
        gold = {normalize(ref).upper() for ref in references if normalize(ref)}
        correct = bool(pred_choice and pred_choice in gold)
        return {
            "score": 1.0 if correct else 0.0,
            "f1": 1.0 if correct else 0.0,
            "correct": correct,
            "extracted_answer": pred_choice or extract_answer(prediction, references),
        }
    match_fraction = string_match_fraction(prediction, references)
    return {
        "score": match_fraction,
        "f1": token_f1(prediction, references),
        "correct": bool(match_fraction >= 1.0),
        "extracted_answer": extract_answer(prediction, references),
    }
