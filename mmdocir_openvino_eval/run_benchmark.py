#!/usr/bin/env python3
"""Benchmark OpenVINO embedding models on MMDocIR page/layout retrieval."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

from mmdocir_eval.backend import create_backend
from mmdocir_eval.data import load_annotations, selected_row_count
from mmdocir_eval.runner import (
    encode_layouts,
    encode_pages,
    encode_queries,
    evaluate_scope,
    system_metadata,
)


HERE = Path(__file__).resolve().parent
RUNNING = HERE.parent
DEFAULT_DATASET = Path("/home/chaofan/datasets/MMDocIR_Evaluation_Dataset")
DEFAULT_PACKAGE = RUNNING / "ov_models" / "gme-qwen2-vl-2b-instruct-openvino"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--backend", default="gme-openvino")
    parser.add_argument("--precisions", default="fp32,int8_causal_vopt,int4_causal_vopt")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--batch-profile", choices=("recommended", "uniform"), default="recommended")
    parser.add_argument("--batch-size", type=int, default=1, help="fallback and uniform-profile batch")
    parser.add_argument("--query-batch-size", type=int)
    parser.add_argument("--page-batch-size", type=int)
    parser.add_argument("--text-layout-batch-size", type=int)
    parser.add_argument("--fused-layout-batch-size", type=int)
    parser.add_argument("--batch-mode", choices=("auto", "per_sample", "padded"), default="auto")
    parser.add_argument("--max-image-tokens", type=int)
    parser.add_argument("--query-instruction")
    parser.add_argument("--scopes", default="close,open")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--max-queries", "--max-samples", dest="max_queries", type=int)
    parser.add_argument("--subset-file", type=Path, help="JSON manifest containing fixed query_ids")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    parser.add_argument("--ov-cache-dir", type=Path, default=HERE / ".ov_cache")
    parser.add_argument("--force", action="store_true", help="ignore complete embedding caches")
    parser.add_argument("--dry-run", action="store_true", help="print selection and exit before loading a model")
    return parser.parse_args()


def require_files(dataset_dir: Path) -> tuple[Path, Path, Path]:
    files = tuple(
        dataset_dir / name
        for name in ("MMDocIR_annotations.jsonl", "MMDocIR_pages.parquet", "MMDocIR_layouts.parquet")
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise SystemExit(f"dataset files missing: {', '.join(missing)}; run download_dataset.py")
    return files


def quality_deltas(runs: dict, baseline_name: str = "fp32") -> dict:
    if baseline_name not in runs:
        return {"available": False, "reason": f"baseline {baseline_name!r} was not run"}
    baseline = runs[baseline_name]["evaluation"]
    output = {"available": True, "baseline": baseline_name, "precisions": {}}
    for precision, run in runs.items():
        if precision == baseline_name:
            continue
        precision_output = {}
        for scope, scope_result in run["evaluation"].items():
            if scope not in baseline:
                continue
            precision_output[scope] = {}
            for level in ("page", "layout"):
                precision_output[scope][level] = {}
                for metric, value in scope_result[level].items():
                    reference = baseline[scope][level][metric]
                    precision_output[scope][level][metric] = {
                        "value": value,
                        "fp32": reference,
                        "absolute": value - reference,
                        "percentage_points": (value - reference) * 100,
                        "relative_percent": ((value / reference) - 1) * 100 if reference else None,
                    }
        output["precisions"][precision] = precision_output
    return output


def package_identity(package: Path) -> dict:
    identity = {"path": str(package.resolve()), "configs": {}}
    if not package.is_dir():
        return identity
    for path in sorted(package.glob("*.json")):
        payload = path.read_bytes()
        identity["configs"][path.name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return identity


def load_subset(path: Path) -> tuple[dict, set[int]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    query_ids = [
        int(query_id)
        for domain in manifest.get("domains", [])
        for query_id in domain.get("query_ids", [])
    ]
    if not query_ids:
        raise ValueError(f"subset manifest contains no query IDs: {path}")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError(f"subset manifest contains duplicate query IDs: {path}")
    expected = manifest.get("query_count")
    if expected is not None and int(expected) != len(query_ids):
        raise ValueError(
            f"subset manifest declares {expected} queries but contains {len(query_ids)}: {path}"
        )
    payload = path.read_bytes()
    provenance = {
        "path": str(path.resolve()),
        "name": manifest.get("name", path.stem),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "manifest": manifest,
    }
    return provenance, set(query_ids)


def resolve_batches(args: argparse.Namespace, precision: str) -> dict[str, int]:
    if args.batch_profile == "recommended":
        recommended = {
            "query": 8,
            "page": 1,
            "text_layout": 1 if precision == "fp32" else 4,
            "fused_layout": 1,
        }
    else:
        recommended = {name: args.batch_size for name in ("query", "page", "text_layout", "fused_layout")}
    explicit = {
        "query": args.query_batch_size,
        "page": args.page_batch_size,
        "text_layout": args.text_layout_batch_size,
        "fused_layout": args.fused_layout_batch_size,
    }
    return {
        name: explicit[name] if explicit[name] is not None else recommended[name]
        for name in recommended
    }


def main() -> None:
    args = parse_args()
    explicit_batches = (
        args.batch_size,
        args.query_batch_size,
        args.page_batch_size,
        args.text_layout_batch_size,
        args.fused_layout_batch_size,
    )
    if any(value is not None and value < 1 for value in explicit_batches):
        raise SystemExit("all batch sizes must be positive")
    if args.max_documents is not None and args.max_documents < 1:
        raise SystemExit("--max-documents must be positive")
    if args.max_queries is not None and args.max_queries < 1:
        raise SystemExit("--max-samples must be positive")
    if args.subset_file is not None and (
        args.max_documents is not None or args.max_queries is not None
    ):
        raise SystemExit("--subset-file cannot be combined with --max-documents/--max-samples")
    annotations_path, pages_path, layouts_path = require_files(args.dataset_dir)
    subset, selected_query_ids = (
        load_subset(args.subset_file) if args.subset_file is not None else (None, None)
    )
    documents = load_annotations(
        annotations_path,
        args.max_documents,
        args.max_queries,
        selected_query_ids,
    )
    queries = tuple(query for document in documents for query in document.queries)
    if not queries:
        raise SystemExit("selection contains no queries")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.ov_cache_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "documents": len(documents),
        "queries": len(queries),
        "pages": selected_row_count(documents, "page"),
        "layouts": selected_row_count(documents, "layout"),
        "doc_names": [document.doc_name for document in documents],
        "query_ids": [query.query_id for query in queries],
        "domain_query_counts": {
            domain: sum(query.domain == domain for query in queries)
            for domain in sorted({query.domain for query in queries})
        },
        "subset": subset,
    }
    if args.dry_run:
        precisions = [value.strip() for value in args.precisions.split(",") if value.strip()]
        print(
            json.dumps(
                {
                    "dataset_dir": str(args.dataset_dir.resolve()),
                    "selection": selection,
                    "batch_profile": args.batch_profile,
                    "execution_batches": {
                        precision: resolve_batches(args, precision) for precision in precisions
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    manifest_path = args.dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    all_results = {
        "format": 1,
        "system": system_metadata(),
        "dataset": {
            "name": "MMDocIR/MMDocIR_Evaluation_Dataset",
            "directory": str(args.dataset_dir.resolve()),
            "manifest": manifest,
            "selection": selection,
            "page_input": "image_binary",
            "layout_input": "image/table -> fused(vlm_text,image_binary); all other types -> text",
            "matching": "cosine similarity (L2-normalized dot product)",
        },
        "runs": {},
    }
    for precision in [value.strip() for value in args.precisions.split(",") if value.strip()]:
        batches = resolve_batches(args, precision)
        profile = (
            f"b{args.batch_size}"
            if args.batch_profile == "uniform" and len(set(batches.values())) == 1
            else f"q{batches['query']}_p{batches['page']}_t{batches['text_layout']}_f{batches['fused_layout']}"
        )
        print(f"[{precision}] loading backend", flush=True)
        backend = create_backend(
            args.backend,
            package=args.package,
            precision=precision,
            device=args.device,
            batch_size=max(batches.values()),
            running_dir=RUNNING,
            cache_dir=args.ov_cache_dir,
            batch_mode=args.batch_mode,
            max_image_tokens=args.max_image_tokens,
            query_instruction=args.query_instruction,
        )
        backend_metadata = backend.metadata()
        stable_backend_metadata = {
            key: value for key, value in backend_metadata.items() if key != "load_seconds"
        }
        cache_key = {
            "format": 1,
            "backend": stable_backend_metadata,
            "selection": selection,
            "pages_size": pages_path.stat().st_size,
            "layouts_size": layouts_path.stat().st_size,
            "dataset_revision": manifest.get("revision") if manifest else None,
            "package": package_identity(args.package),
            "execution_batches": batches,
        }
        print(f"[{precision}] encoding {len(queries)} queries", flush=True)
        query_vectors, query_performance = encode_queries(backend, queries, batches["query"])
        print(f"[{precision}] encoding {selection['pages']} pages", flush=True)
        pages = encode_pages(
            backend, pages_path, documents, args.cache_dir, cache_key, batches["page"], args.force
        )
        print(f"[{precision}] encoding {selection['layouts']} layouts", flush=True)
        layouts = encode_layouts(
            backend,
            layouts_path,
            documents,
            args.cache_dir,
            cache_key,
            batches["text_layout"],
            batches["fused_layout"],
            args.force,
        )
        run = {
            "backend": backend_metadata,
            "execution_batches": batches,
            "encoding": {"query": query_performance, "page": pages.performance, "layout": layouts.performance},
            "evaluation": {},
        }
        for scope in [value.strip() for value in args.scopes.split(",") if value.strip()]:
            print(f"[{precision}] evaluating {scope}-domain", flush=True)
            summary, details = evaluate_scope(scope, queries, query_vectors, pages, layouts)
            run["evaluation"][scope] = summary
            detail_path = args.output_dir / f"per_query_{precision}_{profile}_{scope}.jsonl"
            with detail_path.open("w", encoding="utf-8") as output:
                for row in details:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
        all_results["runs"][precision] = run
        output_path = args.output_dir / "summary.json"
        output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{precision}] checkpointed {output_path}", flush=True)
        del backend, query_vectors, pages, layouts
        gc.collect()
    all_results["quality_delta_vs_fp32"] = quality_deltas(all_results["runs"])
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()