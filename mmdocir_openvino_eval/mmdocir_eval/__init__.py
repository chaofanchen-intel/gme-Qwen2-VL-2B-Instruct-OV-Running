"""Reusable MMDocIR evaluation components."""

from .metrics import BBox, LayoutCandidate, layout_metrics, page_metrics

__all__ = ["BBox", "LayoutCandidate", "layout_metrics", "page_metrics"]