"""End-to-end MMDocIR encoding, caching, matching, and reporting."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import psutil
from PIL import Image

from .backend import EmbeddingBackend
from .data import Document, Query, iter_parquet_rows, selected_row_count
from .metrics import LayoutCandidate, layout_metrics, page_metrics
from .timing import summarize_ms, throughput


@dataclass
class EncodedIndex:
    vectors: np.ndarray
    rows: list[dict[str, Any]]
    performance: dict[str, Any]


class ResourceTracker:
    """Track observable process and plugin memory between inference calls."""

    def __init__(self, backend: EmbeddingBackend) -> None:
        self.backend = backend
        self.samples: list[dict[str, Any]] = []
        self.sample()

    def sample(self) -> None:
        snapshot: dict[str, Any] = {
            "process_rss_bytes": psutil.Process().memory_info().rss,
        }
        resource_snapshot = getattr(self.backend, "resource_snapshot", None)
        if resource_snapshot is not None:
            snapshot.update(resource_snapshot())
        self.samples.append(snapshot)

    def summary(self) -> dict[str, Any]:
        rss = [sample["process_rss_bytes"] for sample in self.samples]
        gpu_keys = sorted(
            {
                key
                for sample in self.samples
                for key in sample.get("gpu_memory_bytes", {})
            }
        )
        gpu_peak = {
            key: max(sample.get("gpu_memory_bytes", {}).get(key, 0) for sample in self.samples)
            for key in gpu_keys
        }
        gpu_totals = [sum(sample.get("gpu_memory_bytes", {}).values()) for sample in self.samples]
        return {
            "sampling": "before/after inference calls; internal instantaneous peaks may be higher",
            "samples": len(self.samples),
            "process_rss_bytes": {"start": rss[0], "end": rss[-1], "observed_peak": max(rss)},
            "gpu_memory_observed_peak_bytes": gpu_peak,
            "gpu_memory_observed_total_peak_bytes": max(gpu_totals, default=0),
        }


def _json_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_image(binary: bytes) -> Image.Image:
    with Image.open(BytesIO(binary)) as image:
        return image.convert("RGB").copy()


def _owner_maps(documents: Sequence[Document]) -> tuple[dict[int, tuple[str, int]], dict[int, str]]:
    pages: dict[int, tuple[str, int]] = {}
    layouts: dict[int, str] = {}
    for document in documents:
        for global_index in range(document.page_rows.start, document.page_rows.end + 1):
            pages[global_index] = (document.doc_name, global_index - document.page_rows.start)
        for global_index in range(document.layout_rows.start, document.layout_rows.end + 1):
            layouts[global_index] = document.doc_name
    return pages, layouts


def _cache_paths(cache_root: Path, kind: str, fingerprint: str) -> tuple[Path, Path, Path]:
    prefix = cache_root / f"{kind}_{fingerprint}"
    return prefix.with_suffix(".npy"), prefix.with_suffix(".rows.json"), prefix.with_suffix(".perf.json")


def _cached_index(
    cache_root: Path,
    kind: str,
    fingerprint: str,
    expected_count: int,
    expected_dim: int,
) -> EncodedIndex | None:
    vectors_path, rows_path, performance_path = _cache_paths(cache_root, kind, fingerprint)
    if not all(path.exists() for path in (vectors_path, rows_path, performance_path)):
        return None
    vectors = np.load(vectors_path, mmap_mode="r")
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    expected_shape = (expected_count, expected_dim)
    if vectors.shape != expected_shape or vectors.dtype != np.float32 or len(rows) != expected_count:
        raise RuntimeError(
            f"invalid cache {vectors_path}: vectors={vectors.shape}/{vectors.dtype}, "
            f"rows={len(rows)}, expected={expected_shape}/float32"
        )
    performance = {**performance, "cache_hit": True}
    return EncodedIndex(vectors=vectors, rows=rows, performance=performance)


def encode_queries(
    backend: EmbeddingBackend, queries: Sequence[Query], batch_size: int
) -> tuple[np.ndarray, dict[str, Any]]:
    vectors: list[np.ndarray] = []
    latencies_ms: list[float] = []
    resources = ResourceTracker(backend)
    started = time.perf_counter()
    for offset in range(0, len(queries), batch_size):
        batch = [query.text for query in queries[offset : offset + batch_size]]
        call_started = time.perf_counter()
        vectors.append(_normalize(backend.encode_queries(batch, batch_size=batch_size)))
        resources.sample()
        latencies_ms.append((time.perf_counter() - call_started) * 1000)
    elapsed = time.perf_counter() - started
    output = np.concatenate(vectors, axis=0)
    steady_items = max(0, len(queries) - min(batch_size, len(queries)))
    steady_seconds = sum(latencies_ms[1:]) / 1000
    return output, {
        "items": len(queries),
        "wall_seconds": elapsed,
        "queries_per_second": throughput(len(queries), elapsed),
        "batch_call_latency_ms": summarize_ms(latencies_ms),
        "average_per_query_ms": elapsed * 1000 / len(queries),
        "steady_state_batch_call_latency_ms": summarize_ms(latencies_ms[1:]),
        "steady_state_queries_per_second": throughput(steady_items, steady_seconds),
        "steady_state_average_per_query_ms": (
            steady_seconds * 1000 / steady_items if steady_items else None
        ),
        "resources": resources.summary(),
    }


def encode_pages(
    backend: EmbeddingBackend,
    parquet_path: Path,
    documents: Sequence[Document],
    cache_root: Path,
    cache_key: dict[str, Any],
    batch_size: int,
    force: bool,
) -> EncodedIndex:
    fingerprint = _fingerprint({**cache_key, "kind": "page"})
    count = selected_row_count(documents, "page")
    if not force and (
        cached := _cached_index(cache_root, "page", fingerprint, count, backend.embedding_dim)
    ) is not None:
        return cached
    vectors_path, rows_path, performance_path = _cache_paths(cache_root, "page", fingerprint)
    temporary_vectors_path = vectors_path.with_suffix(vectors_path.suffix + ".tmp")
    vectors = np.lib.format.open_memmap(
        temporary_vectors_path, mode="w+", dtype=np.float32, shape=(count, backend.embedding_dim)
    )
    page_owners, _ = _owner_maps(documents)
    rows: list[dict[str, Any]] = []
    pending: list[tuple[int, Image.Image]] = []
    encode_ms: list[float] = []
    resources = ResourceTracker(backend)
    preprocessing_ms = 0.0
    started = time.perf_counter()

    def flush() -> None:
        if not pending:
            return
        positions, images = zip(*pending)
        call_started = time.perf_counter()
        batch_vectors = _normalize(backend.encode_page_images(images, batch_size=batch_size))
        resources.sample()
        encode_ms.append((time.perf_counter() - call_started) * 1000)
        vectors[list(positions)] = batch_vectors
        pending.clear()

    for position, (global_index, row) in enumerate(
        iter_parquet_rows(parquet_path, ["doc_name", "passage_id", "image_binary"], documents, "page")
    ):
        prep_started = time.perf_counter()
        image = _load_image(row["image_binary"])
        preprocessing_ms += (time.perf_counter() - prep_started) * 1000
        doc_name, page_id = page_owners[global_index]
        rows.append(
            {
                "global_index": global_index,
                "doc_name": doc_name,
                "page_id": page_id,
                "passage_id": row["passage_id"],
            }
        )
        pending.append((position, image))
        if len(pending) >= batch_size:
            flush()
    flush()
    if len(rows) != count:
        raise RuntimeError(f"page parquet yielded {len(rows)} selected rows; expected {count}")
    vectors.flush()
    vector_bytes = int(vectors.nbytes)
    del vectors
    temporary_vectors_path.replace(vectors_path)
    elapsed = time.perf_counter() - started
    steady_items = max(0, count - min(batch_size, count))
    steady_seconds = sum(encode_ms[1:]) / 1000
    performance = {
        "cache_hit": False,
        "items": count,
        "wall_seconds": elapsed,
        "pages_per_second": throughput(count, elapsed),
        "model_call_seconds": sum(encode_ms) / 1000,
        "preprocessing_seconds": preprocessing_ms / 1000,
        "batch_call_latency_ms": summarize_ms(encode_ms),
        "steady_state_batch_call_latency_ms": summarize_ms(encode_ms[1:]),
        "steady_state_model_only_pages_per_second": throughput(steady_items, steady_seconds),
        "vector_bytes": vector_bytes,
        "resources": resources.summary(),
    }
    _json_dump(rows_path, rows)
    _json_dump(performance_path, performance)
    return EncodedIndex(np.load(vectors_path, mmap_mode="r"), rows, performance)


def encode_layouts(
    backend: EmbeddingBackend,
    parquet_path: Path,
    documents: Sequence[Document],
    cache_root: Path,
    cache_key: dict[str, Any],
    text_batch_size: int,
    fused_batch_size: int,
    force: bool,
) -> EncodedIndex:
    fingerprint = _fingerprint({**cache_key, "kind": "layout", "hybrid": "image/table=fused-vlm"})
    count = selected_row_count(documents, "layout")
    if not force and (
        cached := _cached_index(cache_root, "layout", fingerprint, count, backend.embedding_dim)
    ) is not None:
        return cached
    vectors_path, rows_path, performance_path = _cache_paths(cache_root, "layout", fingerprint)
    temporary_vectors_path = vectors_path.with_suffix(vectors_path.suffix + ".tmp")
    vectors = np.lib.format.open_memmap(
        temporary_vectors_path, mode="w+", dtype=np.float32, shape=(count, backend.embedding_dim)
    )
    _, layout_owners = _owner_maps(documents)
    rows: list[dict[str, Any]] = []
    text_pending: list[tuple[int, str]] = []
    fused_pending: list[tuple[int, str, Image.Image]] = []
    timings: dict[str, list[float]] = {"text": [], "fused": []}
    resources = ResourceTracker(backend)
    counts = {"text": 0, "fused": 0}
    preprocessing_ms = 0.0
    started = time.perf_counter()

    def flush_text() -> None:
        if not text_pending:
            return
        positions, texts = zip(*text_pending)
        call_started = time.perf_counter()
        batch_vectors = _normalize(
            backend.encode_layout_texts(texts, batch_size=text_batch_size)
        )
        resources.sample()
        timings["text"].append((time.perf_counter() - call_started) * 1000)
        vectors[list(positions)] = batch_vectors
        text_pending.clear()

    def flush_fused() -> None:
        if not fused_pending:
            return
        positions, texts, images = zip(*fused_pending)
        call_started = time.perf_counter()
        batch_vectors = _normalize(
            backend.encode_layout_fused(texts, images, batch_size=fused_batch_size)
        )
        resources.sample()
        timings["fused"].append((time.perf_counter() - call_started) * 1000)
        vectors[list(positions)] = batch_vectors
        fused_pending.clear()

    columns = ["doc_name", "type", "layout_id", "page_id", "bbox", "image_binary", "text", "vlm_text"]
    for position, (global_index, row) in enumerate(
        iter_parquet_rows(parquet_path, columns, documents, "layout")
    ):
        doc_name = layout_owners[global_index]
        layout_type = str(row["type"]).lower()
        rows.append(
            {
                "global_index": global_index,
                "doc_name": doc_name,
                "layout_id": int(row["layout_id"]),
                "page_id": int(row["page_id"]),
                "bbox": [float(value) for value in row["bbox"]],
                "type": layout_type,
            }
        )
        if layout_type in {"image", "table"}:
            prep_started = time.perf_counter()
            image = _load_image(row["image_binary"])
            preprocessing_ms += (time.perf_counter() - prep_started) * 1000
            fused_pending.append((position, row["vlm_text"] or "", image))
            counts["fused"] += 1
            if len(fused_pending) >= fused_batch_size:
                flush_fused()
        else:
            text_pending.append((position, row["text"] or ""))
            counts["text"] += 1
            if len(text_pending) >= text_batch_size:
                flush_text()
    flush_text()
    flush_fused()
    if len(rows) != count:
        raise RuntimeError(f"layout parquet yielded {len(rows)} selected rows; expected {count}")
    vectors.flush()
    vector_bytes = int(vectors.nbytes)
    del vectors
    temporary_vectors_path.replace(vectors_path)
    elapsed = time.perf_counter() - started
    performance: dict[str, Any] = {
        "cache_hit": False,
        "items": count,
        "wall_seconds": elapsed,
        "layouts_per_second": throughput(count, elapsed),
        "preprocessing_seconds": preprocessing_ms / 1000,
        "vector_bytes": vector_bytes,
        "resources": resources.summary(),
        "modalities": {},
    }
    for modality in ("text", "fused"):
        seconds = sum(timings[modality]) / 1000
        modality_batch_size = text_batch_size if modality == "text" else fused_batch_size
        steady_items = max(0, counts[modality] - min(modality_batch_size, counts[modality]))
        steady_seconds = sum(timings[modality][1:]) / 1000
        performance["modalities"][modality] = {
            "items": counts[modality],
            "batch_size": modality_batch_size,
            "model_call_seconds": seconds,
            "model_only_layouts_per_second": throughput(counts[modality], seconds),
            "batch_call_latency_ms": summarize_ms(timings[modality]),
            "steady_state_batch_call_latency_ms": summarize_ms(timings[modality][1:]),
            "steady_state_model_only_layouts_per_second": throughput(
                steady_items, steady_seconds
            ),
        }
    _json_dump(rows_path, rows)
    _json_dump(performance_path, performance)
    return EncodedIndex(np.load(vectors_path, mmap_mode="r"), rows, performance)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _aggregate(per_query: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    names = sorted(per_query[0][key]) if per_query else []
    return {name: float(np.mean([row[key][name] for row in per_query])) for name in names}


def evaluate_scope(
    scope: str,
    queries: Sequence[Query],
    query_vectors: np.ndarray,
    pages: EncodedIndex,
    layouts: EncodedIndex,
    ks: Sequence[int] = (5, 10),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    page_by_doc: dict[str, list[int]] = defaultdict(list)
    layout_by_doc: dict[str, list[int]] = defaultdict(list)
    layout_by_doc_page: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(pages.rows):
        page_by_doc[row["doc_name"]].append(index)
    layout_candidates = [
        LayoutCandidate(row["page_id"], tuple(row["bbox"]), row["doc_name"]) for row in layouts.rows
    ]
    for index, candidate in enumerate(layout_candidates):
        layout_by_doc[candidate.doc_name].append(index)
        layout_by_doc_page[(candidate.doc_name, candidate.page_id)].append(index)

    page_score_ms: list[float] = []
    page_rank_ms: list[float] = []
    layout_score_ms: list[float] = []
    layout_rank_ms: list[float] = []
    details: list[dict[str, Any]] = []
    max_k = max(ks)
    warm_query = query_vectors[0]
    if scope == "close":
        warm_page_indices = page_by_doc[queries[0].doc_name]
        warm_layout_indices = layout_by_doc[queries[0].doc_name]
        warm_page_scores = np.asarray(pages.vectors[warm_page_indices]) @ warm_query
        warm_layout_scores = np.asarray(layouts.vectors[warm_layout_indices]) @ warm_query
    else:
        warm_page_scores = pages.vectors @ warm_query
        warm_layout_scores = layouts.vectors @ warm_query
    np.argsort(-warm_page_scores, kind="stable")[:max_k]
    np.argsort(-warm_layout_scores, kind="stable")[:max_k]
    for query_offset, query in enumerate(queries):
        if scope == "close":
            page_indices = page_by_doc[query.doc_name]
            layout_indices = layout_by_doc[query.doc_name]
        elif scope == "open":
            page_indices = None
            layout_indices = None
        else:
            raise ValueError(f"unknown scope {scope!r}")
        query_vector = query_vectors[query_offset]

        started = time.perf_counter()
        page_matrix = pages.vectors if page_indices is None else np.asarray(pages.vectors[page_indices])
        page_scores = page_matrix @ query_vector
        page_score_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        page_order = np.argsort(-page_scores, kind="stable")[:max_k]
        ranked_page_rows = [
            pages.rows[int(position) if page_indices is None else page_indices[position]]
            for position in page_order
        ]
        page_rank_ms.append((time.perf_counter() - started) * 1000)
        if scope == "close":
            ranked_page_ids = [row["page_id"] for row in ranked_page_rows]
            relevant_pages = query.relevant_page_ids
        else:
            ranked_page_ids = [row["global_index"] for row in ranked_page_rows]
            relevant_pages = [query.page_rows.start + page_id for page_id in query.relevant_page_ids]
        query_page_metrics = page_metrics(ranked_page_ids, relevant_pages, ks)

        started = time.perf_counter()
        layout_matrix = layouts.vectors if layout_indices is None else np.asarray(layouts.vectors[layout_indices])
        layout_scores = layout_matrix @ query_vector
        layout_score_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        layout_order = np.argsort(-layout_scores, kind="stable")[:max_k]
        ranked_layouts = [
            layout_candidates[int(position) if layout_indices is None else layout_indices[position]]
            for position in layout_order
        ]
        layout_rank_ms.append((time.perf_counter() - started) * 1000)
        ground_truth = [
            LayoutCandidate(layout.page_id, layout.bbox, query.doc_name)
            for layout in query.relevant_layouts
        ]
        ideal_indices = {
            index
            for target in ground_truth
            for index in layout_by_doc_page[(target.doc_name, target.page_id)]
        }
        ideal_candidates = [layout_candidates[index] for index in ideal_indices]
        query_layout_metrics = layout_metrics(
            ranked_layouts, ideal_candidates, ground_truth, ks
        )
        details.append(
            {
                "query_id": query.query_id,
                "doc_name": query.doc_name,
                "domain": query.domain,
                "question_type": query.question_type,
                "query": query.text,
                "page": query_page_metrics,
                "layout": query_layout_metrics,
                "top_pages": ranked_page_rows,
                "top_layouts": [asdict(candidate) for candidate in ranked_layouts],
            }
        )

    summary = {
        "scope": scope,
        "queries": len(queries),
        "page": _aggregate(details, "page"),
        "layout": _aggregate(details, "layout"),
        "match_latency_ms": {
            "cpu_page_score": summarize_ms(page_score_ms),
            "cpu_page_topk": summarize_ms(page_rank_ms),
            "cpu_page_score_and_topk": summarize_ms(
                [score + rank for score, rank in zip(page_score_ms, page_rank_ms)]
            ),
            "cpu_layout_score": summarize_ms(layout_score_ms),
            "cpu_layout_topk": summarize_ms(layout_rank_ms),
            "cpu_layout_score_and_topk": summarize_ms(
                [score + rank for score, rank in zip(layout_score_ms, layout_rank_ms)]
            ),
            "gpu": {"available": False, "reason": "torch.xpu is unavailable in the selected environment"},
        },
    }
    return summary, details


def system_metadata() -> dict[str, Any]:
    try:
        import openvino

        openvino_version = openvino.__version__
        openvino_path = openvino.__file__
    except Exception as error:
        openvino_version = f"unavailable: {error}"
        openvino_path = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "openvino_version": openvino_version,
        "openvino_module": openvino_path,
    }