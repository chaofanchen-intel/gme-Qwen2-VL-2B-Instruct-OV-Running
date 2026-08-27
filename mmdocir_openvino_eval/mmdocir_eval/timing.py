"""Stable timing summaries shared by encoding and matching."""
from __future__ import annotations

import statistics
from typing import Sequence

import numpy as np


def summarize_ms(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def throughput(items: int, seconds: float) -> float:
    return items / seconds if seconds > 0 else 0.0