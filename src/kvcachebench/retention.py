from __future__ import annotations

from collections.abc import Iterable
from typing import Any


EPS = 1e-12


def _na(reason: str) -> dict[str, str]:
    return {"value": "N/A", "reason": reason}


def _valid_span_sets(
    spans: Iterable[Iterable[int]], prompt_len: int
) -> list[set[int]]:
    valid: list[set[int]] = []
    for span in spans or []:
        values = list(span)
        if len(values) != 2:
            continue
        start = max(0, int(values[0]))
        end = min(prompt_len, int(values[1]))
        if end > start:
            valid.append(set(range(start, end)))
    return valid


def slot_retention_metrics(
    records: dict[str, Any],
    evidence_spans: Iterable[Iterable[int]],
    distractor_spans: Iterable[Iterable[int]],
    prompt_len: int,
    ecov_threshold: float = 0.5,
) -> dict[str, Any]:
    """Measure evidence retention per layer/KV-head slot.

    Records map layer identifiers to retained-position lists, one per KV head.
    The function never unions positions across slots.
    """
    if not 0.0 <= ecov_threshold <= 1.0:
        raise ValueError("ecov_threshold must be in [0, 1]")
    if prompt_len <= 0:
        raise ValueError("prompt_len must be positive")

    evidence_sets = _valid_span_sets(evidence_spans, prompt_len)
    distractor_sets = _valid_span_sets(distractor_spans, prompt_len)
    evidence = set().union(*evidence_sets) if evidence_sets else set()
    distractors = set().union(*distractor_sets) if distractor_sets else set()
    slots: list[set[int]] = []
    for heads in records.values():
        if not isinstance(heads, list):
            continue
        for positions in heads:
            if not isinstance(positions, list):
                continue
            slots.append(
                {
                    int(position)
                    for position in positions
                    if 0 <= int(position) < prompt_len
                }
            )

    if not slots:
        unavailable = _na(
            "N/A: no layer/head retained-position mapping was exposed"
        )
        return {
            "ERR_slot": unavailable,
            "ECov_slot": unavailable,
            "distractor_retention_rate_slot": unavailable,
            "DRR_slot": unavailable,
            "retention_slot_count": 0,
        }

    if not evidence:
        unavailable = _na("N/A: no deterministic evidence token span")
        distractor_rate: Any = (
            sum(
                len(slot & distractors) / len(distractors)
                for slot in slots
            )
            / len(slots)
            if distractors
            else _na("N/A: no deterministic distractor token span")
        )
        return {
            "ERR_slot": unavailable,
            "ECov_slot": unavailable,
            "distractor_retention_rate_slot": distractor_rate,
            "DRR_slot": unavailable,
            "retention_slot_count": len(slots),
        }

    evidence_rate = (
        sum(len(slot & evidence) / len(evidence) for slot in slots) / len(slots)
    )
    span_hits = [
        float(len(slot & span) / len(span) >= ecov_threshold)
        for slot in slots
        for span in evidence_sets
    ]
    ecov = sum(span_hits) / len(span_hits)

    if distractors:
        distractor_rate = (
            sum(
                len(slot & distractors) / len(distractors)
                for slot in slots
            )
            / len(slots)
        )
        drr: Any = distractor_rate / (
            distractor_rate + evidence_rate + EPS
        )
    else:
        distractor_rate = _na("N/A: no deterministic distractor token span")
        drr = _na("N/A: no deterministic distractor token span")

    return {
        "ERR_slot": evidence_rate,
        "ECov_slot": ecov,
        "distractor_retention_rate_slot": distractor_rate,
        "DRR_slot": drr,
        "retention_slot_count": len(slots),
    }
