"""Retrieval metrics used by the MMDocIR OpenVINO benchmark."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class LayoutCandidate:
    page_id: int
    bbox: BBox
    doc_name: str = ""


def _dcg(gains: Sequence[float], k: int) -> float:
    return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains[:k]))


def page_metrics(
    ranked_page_ids: Sequence[int], relevant_page_ids: Iterable[int], ks: Sequence[int] = (5, 10)
) -> dict[str, float]:
    """Return macro-ready Recall@K and binary nDCG@K for one query."""
    relevant = set(relevant_page_ids)
    if not relevant:
        raise ValueError("a query must have at least one relevant page")
    gains = [1.0 if page_id in relevant else 0.0 for page_id in ranked_page_ids]
    ideal = [1.0] * len(relevant)
    output: dict[str, float] = {}
    for k in ks:
        if k < 1:
            raise ValueError(f"K must be positive, got {k}")
        output[f"recall@{k}"] = len(set(ranked_page_ids[:k]) & relevant) / len(relevant)
        ideal_dcg = _dcg(ideal, k)
        output[f"ndcg@{k}"] = _dcg(gains, k) / ideal_dcg if ideal_dcg else 0.0
    return output


def intersection_area(left: BBox, right: BBox) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def layout_ground_truth_area(ground_truth: Sequence[LayoutCandidate]) -> float:
    """Return the total evaluable area of a query's ground-truth layouts."""
    return sum(_area(target.bbox) for target in ground_truth)


def _layout_gain(candidate: LayoutCandidate, ground_truth: Sequence[LayoutCandidate]) -> float:
    return sum(
        intersection_area(candidate.bbox, target.bbox)
        for target in ground_truth
        if candidate.doc_name == target.doc_name and candidate.page_id == target.page_id
    )


def layout_metrics(
    ranked_candidates: Sequence[LayoutCandidate],
    all_candidates: Sequence[LayoutCandidate],
    ground_truth: Sequence[LayoutCandidate],
    ks: Sequence[int] = (5, 10),
) -> dict[str, float]:
    """Return area Recall@K and graded nDCG@K for one query.

    Recall follows MMDocIR's published evaluator: intersected ground-truth area
    accumulated over retrieved layouts, divided by total ground-truth area.
    nDCG treats each candidate's intersected area as its graded relevance and
    computes the ideal ranking over the same candidate scope.
    """
    if not ground_truth:
        raise ValueError("a query must have at least one relevant layout")
    ground_truth_area = layout_ground_truth_area(ground_truth)
    if ground_truth_area <= 0:
        raise ValueError("ground-truth layouts must have positive area")

    ranked_gains = [_layout_gain(candidate, ground_truth) for candidate in ranked_candidates]
    ideal_gains = sorted(
        (_layout_gain(candidate, ground_truth) for candidate in all_candidates), reverse=True
    )
    output: dict[str, float] = {}
    for k in ks:
        if k < 1:
            raise ValueError(f"K must be positive, got {k}")
        output[f"recall@{k}"] = sum(ranked_gains[:k]) / ground_truth_area
        ideal_dcg = _dcg(ideal_gains, k)
        output[f"ndcg@{k}"] = _dcg(ranked_gains, k) / ideal_dcg if ideal_dcg else 0.0
    return output