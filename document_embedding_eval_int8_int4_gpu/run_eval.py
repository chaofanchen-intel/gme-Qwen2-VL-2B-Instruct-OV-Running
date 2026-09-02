#!/usr/bin/env python3
"""Benchmark GME PDF embeddings on GPU with final causal and visual optimizations."""
from __future__ import annotations

import gc
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parent
RUNNING = ROOT.parent
PDF_PATH = RUNNING / "060_OT_Downlook_PR_Error.pdf"
PACKAGE = RUNNING / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"
RESULTS = ROOT / "results"
PAGES = ROOT / "pages"
BATCHES = (1, 4, 8, 14)
REPEATS = 3
CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
DPI = 150

QUALITY_QUERIES = [
    {"id": "halo", "text": "What causes the halo effect on the edge bump?", "pages": [2]},
    {"id": "pr_error", "text": "What should be checked when Downlook PR fails?", "pages": [2, 3, 4, 5]},
    {"id": "flux", "text": "How should flux coverage be checked?", "pages": [3]},
    {"id": "substrate", "text": "What substrate should be used for PR learning?", "pages": [3]},
    {"id": "threshold", "text": "How should the PR threshold and area be adjusted?", "pages": [4]},
    {"id": "lighting", "text": "How should the PR lighting be adjusted?", "pages": [4]},
    {"id": "inspection_log", "text": "How do I turn on the Vision inspection log?", "pages": [5, 6]},
    {"id": "dump_files", "text": "How do I save Vision Dump files and system files?", "pages": [11, 12, 13]},
]


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def token_stats(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile([float(x) for x in values], 50),
        "mean": float(statistics.mean(values)),
        "max": max(values),
    }


def extract_pdf() -> tuple[list[str], dict[str, Any]]:
    start = time.perf_counter()
    document = fitz.open(PDF_PATH)
    open_ms = (time.perf_counter() - start) * 1000
    texts: list[str] = []
    extract_ms: list[float] = []
    for page in document:
        t0 = time.perf_counter()
        texts.append(page.get_text("text", sort=True).strip())
        extract_ms.append((time.perf_counter() - t0) * 1000)
    document.close()
    return texts, {
        "path": str(PDF_PATH),
        "pages": len(texts),
        "file_bytes": PDF_PATH.stat().st_size,
        "open_ms": open_ms,
        "text_extract_ms": stats(extract_ms),
        "text_chars": sum(len(text) for text in texts),
    }


def render_pages() -> tuple[list[str], dict[str, Any]]:
    PAGES.mkdir(parents=True, exist_ok=True)
    document = fitz.open(PDF_PATH)
    paths: list[str] = []
    render_ms: list[float] = []
    scale = DPI / 72.0
    matrix = fitz.Matrix(scale, scale)
    for index, page in enumerate(document, 1):
        path = PAGES / f"page_{index:02d}_{DPI}dpi.png"
        t0 = time.perf_counter()
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(path)
        render_ms.append((time.perf_counter() - t0) * 1000)
        paths.append(str(path))
    document.close()
    return paths, {
        "dpi": DPI,
        "render_ms": stats(render_ms),
        "image_width": Image.open(paths[0]).width,
        "image_height": Image.open(paths[0]).height,
        "total_bytes": sum(Path(path).stat().st_size for path in paths),
    }


def tokenize_texts(encoder: Any, texts: list[str]) -> list[int]:
    tokenizer = encoder.processor.tokenizer
    result = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False)
        result.append(len(encoded["input_ids"]))
    return result


def make_chunks(page_texts: list[str], encoder: Any) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    tokenizer = encoder.processor.tokenizer
    for page_index, text in enumerate(page_texts, 1):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            chunks.append({"id": f"page{page_index}_chunk0", "page": page_index, "text": "", "tokens": 0})
            continue
        start = 0
        chunk_index = 0
        while start < len(ids):
            end = min(start + CHUNK_SIZE, len(ids))
            chunk_ids = ids[start:end]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append({
                "id": f"page{page_index}_chunk{chunk_index}",
                "page": page_index,
                "text": chunk_text,
                "tokens": len(chunk_ids),
            })
            if end == len(ids):
                break
            start = end - CHUNK_OVERLAP
            chunk_index += 1
    return chunks


def make_inputs(page_texts: list[str], page_images: list[str], encoder: Any) -> dict[str, Any]:
    chunks = make_chunks(page_texts, encoder)
    page_token_counts = tokenize_texts(encoder, page_texts)
    chunk_token_counts = [chunk["tokens"] for chunk in chunks]
    return {
        "page_text": [{"id": f"page{index}", "page": index, "text": text, "tokens": page_token_counts[index - 1]} for index, text in enumerate(page_texts, 1)],
        "text_chunk_256": chunks,
        "page_image_150": [{"id": f"page{index}", "page": index, "image": image} for index, image in enumerate(page_images, 1)],
        "page_fused_150": [{"id": f"page{index}", "page": index, "text": text, "image": image, "tokens": page_token_counts[index - 1]} for index, (text, image) in enumerate(zip(page_texts, page_images), 1)],
        "token_stats": {
            "page_text": token_stats(page_token_counts),
            "text_chunk_256": token_stats(chunk_token_counts),
        },
    }


def encode_once(encoder: Any, kind: str, items: list[dict[str, Any]], batch_size: int) -> np.ndarray:
    show = False
    if kind == "page_text" or kind == "text_chunk_256":
        return encoder.get_text_embeddings([item["text"] for item in items], is_query=False, batch_size=batch_size, show_progress_bar=show)
    if kind == "page_image_150":
        return encoder.get_image_embeddings([item["image"] for item in items], is_query=False, batch_size=batch_size, show_progress_bar=show)
    if kind == "page_fused_150":
        return encoder.get_fused_embeddings([item["text"] for item in items], [item["image"] for item in items], is_query=False, batch_size=batch_size, show_progress_bar=show)
    raise ValueError(kind)


def benchmark_representation(encoder: Any, kind: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for batch_size in BATCHES:
        # The API internally splits larger collections; batch_size is the API batch.
        encode_once(encoder, kind, items, batch_size)
        times = []
        rss_peak = rss_mb()
        vectors = None
        for _ in range(REPEATS):
            gc.collect()
            rss_before = rss_mb()
            t0 = time.perf_counter()
            vectors = encode_once(encoder, kind, items, batch_size)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
            rss_peak = max(rss_peak, rss_before, rss_mb())
        rows.append({
            "batch_size": batch_size,
            "items": len(items),
            "embedding_dim": int(vectors.shape[1]),
            "time_ms": stats(times),
            "items_per_sec": len(items) / (statistics.mean(times) / 1000),
            "tokens": sum(item.get("tokens", 0) for item in items),
            "tokens_per_sec": (sum(item.get("tokens", 0) for item in items) / (statistics.mean(times) / 1000)) if kind in ("page_text", "text_chunk_256") else None,
            "rss_peak_mb": rss_peak,
        })
    return {"kind": kind, "items": len(items), "rows": rows}


def cosine_matrix(queries: np.ndarray, documents: np.ndarray) -> np.ndarray:
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    d = documents / np.linalg.norm(documents, axis=1, keepdims=True)
    return q @ d.T


def quality_metrics(encoder: Any, inputs: dict[str, Any], embeddings: dict[str, np.ndarray]) -> dict[str, Any]:
    queries = [item["text"] for item in QUALITY_QUERIES]
    query_vectors = encoder.get_text_embeddings(queries, is_query=True, batch_size=8, show_progress_bar=False)
    output: dict[str, Any] = {"queries": QUALITY_QUERIES}
    for kind in ("page_text", "text_chunk_256", "page_image_150", "page_fused_150"):
        items = inputs[kind]
        scores = cosine_matrix(query_vectors, embeddings[kind])
        hits = []
        reciprocal = []
        for query_index, query in enumerate(QUALITY_QUERIES):
            relevant = set(query["pages"])
            order = np.argsort(-scores[query_index])
            ranked_pages = [items[index]["page"] for index in order]
            rank = next((i + 1 for i, page in enumerate(ranked_pages) if page in relevant), len(ranked_pages) + 1)
            reciprocal.append(1.0 / rank if rank <= len(ranked_pages) else 0.0)
            hits.append({
                "id": query["id"],
                "top_pages": ranked_pages[:5],
                "top_scores": [float(scores[query_index][index]) for index in order[:5]],
                "first_relevant_rank": rank,
            })
        output[kind] = {
            "recall_at_1": float(np.mean([hit["first_relevant_rank"] <= 1 for hit in hits])),
            "recall_at_3": float(np.mean([hit["first_relevant_rank"] <= 3 for hit in hits])),
            "recall_at_5": float(np.mean([hit["first_relevant_rank"] <= 5 for hit in hits])),
            "mrr": float(np.mean(reciprocal)),
            "per_query": hits,
        }
    # Cross-representation agreement for corresponding pages.
    cross = {}
    for left, right in (("page_text", "page_image_150"), ("page_text", "page_fused_150"), ("page_image_150", "page_fused_150")):
        values = np.diag(cosine_matrix(embeddings[left], embeddings[right]))
        cross[f"{left}_vs_{right}"] = {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}
    output["cross_representation_cosine"] = cross
    return output


def main() -> None:
    sys.path.insert(0, str(RUNNING))
    from gme_openvino import GmeOpenVINO

    # Default scope is the three fifth-round final precisions (see README.md).
    # PRECISIONS and OUTPUT remain available for explicit comparison runs.
    precisions_env = os.environ.get("PRECISIONS")
    precisions = tuple(p.strip() for p in precisions_env.split(",")) if precisions_env else (
        "int8_causal_vopt", "int4_causal_vopt", "int4_gs64_causal_vopt"
    )
    visual_dq_env = os.environ.get("VISUAL_DQ_THRESHOLD")
    visual_dq_threshold = int(visual_dq_env) if visual_dq_env else None
    default_output = precisions == (
        "int8_causal_vopt", "int4_causal_vopt", "int4_gs64_causal_vopt"
    )
    default_filename = (
        "benchmark_gme_round9_vopt_vdq0_20260829.json"
        if visual_dq_threshold == 0
        else "benchmark_gme_round5_final_vopt_20260827.json"
    )
    output_path = Path(os.environ.get("OUTPUT", RESULTS / default_filename)) if default_output else \
        Path(os.environ.get("OUTPUT", RESULTS / f"benchmark_{'_'.join(precisions)}.json"))

    RESULTS.mkdir(parents=True, exist_ok=True)
    text_start = time.perf_counter()
    page_texts, pdf_stats = extract_pdf()
    page_images, render_stats = render_pages()
    parse_ms = (time.perf_counter() - text_start) * 1000
    all_results: dict[str, Any] = {
        "pdf": pdf_stats,
        "render": render_stats,
        "parse_and_render_ms": parse_ms,
        "visual_dynamic_quantization_threshold": visual_dq_threshold,
        "precisions": {},
    }

    for precision in precisions:
        print(f"Loading {precision} on GPU", flush=True)
        t0 = time.perf_counter()
        encoder = GmeOpenVINO(
            PACKAGE,
            device="GPU",
            precision=precision,
            batch_mode="auto",
            visual_dynamic_quantization_threshold=visual_dq_threshold,
        )
        load_ms = (time.perf_counter() - t0) * 1000
        inputs = make_inputs(page_texts, page_images, encoder)
        representations = {}
        embedding_vectors = {}
        for kind in ("page_text", "text_chunk_256", "page_image_150", "page_fused_150"):
            print(f"Benchmark {precision}/{kind}: {len(inputs[kind])} items", flush=True)
            representations[kind] = benchmark_representation(encoder, kind, inputs[kind])
            # Use the largest practical API batch once for quality evaluation.
            embedding_vectors[kind] = encode_once(encoder, kind, inputs[kind], 14)
        quality = quality_metrics(encoder, inputs, embedding_vectors)
        all_results["precisions"][precision] = {
            "device": "GPU",
            "model_package": str(PACKAGE),
            "visual_dynamic_quantization_threshold": visual_dq_threshold,
            "load_ms": load_ms,
            "embedding_dim": 1536,
            "token_stats": inputs["token_stats"],
            "representations": representations,
            "quality": quality,
        }
        del encoder
        gc.collect()
    output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
