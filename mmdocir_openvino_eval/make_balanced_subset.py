#!/usr/bin/env python3
"""Build a deterministic, domain-balanced MMDocIR screening manifest."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet


DEFAULT_DATASET = Path("/home/chaofan/datasets/MMDocIR_Evaluation_Dataset")
FP32_RATES = {"page": 0.159, "text": 3.532, "fused": 0.112}


def question_category(value: Any) -> str:
    text = str(value).lower()
    if "layout" in text or "meta" in text:
        return "layout_meta"
    if "table" in text:
        return "table"
    if any(name in text for name in ("chart", "figure", "image", "map")):
        return "visual"
    if "pure-text" in text or "plain-text" in text:
        return "text"
    return "other"


def read_layout_types(path: Path) -> list[str]:
    values: list[str] = []
    for batch in parquet.ParquetFile(path).iter_batches(columns=["type"]):
        values.extend(str(row["type"]) for row in batch.to_pylist())
    return values


def choose_documents(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    states: dict[int, tuple[float, list[dict[str, Any]]]] = {0: (0.0, [])}
    for row in rows:
        for count, (cost, selected) in list(states.items()):
            next_count = min(target, count + len(row["queries"]))
            next_cost = cost + row["estimated_fp32_seconds"]
            if next_count not in states or next_cost < states[next_count][0]:
                states[next_count] = (next_cost, selected + [row])
    if target not in states:
        raise ValueError(f"domain has fewer than {target} queries")
    return states[target][1]


def allocate_categories(
    full_counts: Counter[str], available_counts: Counter[str], target: int
) -> Counter[str]:
    total = sum(full_counts.values())
    quotas = Counter(
        {
            category: min(available_counts[category], math.floor(target * count / total))
            for category, count in full_counts.items()
        }
    )
    while sum(quotas.values()) < target:
        choices = [
            category
            for category in full_counts
            if quotas[category] < available_counts[category]
        ]
        if not choices:
            raise ValueError("selected documents cannot supply the requested category allocation")
        category = max(
            choices,
            key=lambda value: (
                target * full_counts[value] / total - quotas[value],
                full_counts[value],
                value,
            ),
        )
        quotas[category] += 1
    return quotas


def evenly_spaced(pool: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return [pool[min(len(pool) - 1, math.floor((index + 0.5) * len(pool) / count))] for index in range(count)]


def build_manifest(dataset_dir: Path, queries_per_domain: int) -> dict[str, Any]:
    annotations_path = dataset_dir / "MMDocIR_annotations.jsonl"
    layout_types = read_layout_types(dataset_dir / "MMDocIR_layouts.parquet")
    documents: list[dict[str, Any]] = []
    full_categories: dict[str, Counter[str]] = defaultdict(Counter)
    query_id = 0
    with annotations_path.open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            queries = []
            for question in item["questions"]:
                category = question_category(question["type"])
                queries.append({"query_id": query_id, "category": category})
                full_categories[item["domain"]][category] += 1
                query_id += 1
            layout_start, layout_end = item["layout_indices"]
            type_counts = Counter(layout_types[layout_start : layout_end + 1])
            pages = item["page_indices"][1] - item["page_indices"][0] + 1
            text_layouts = type_counts["text"] + type_counts["equation"]
            fused_layouts = type_counts["image"] + type_counts["table"]
            cost = (
                pages / FP32_RATES["page"]
                + text_layouts / FP32_RATES["text"]
                + fused_layouts / FP32_RATES["fused"]
            )
            documents.append(
                {
                    "doc_name": item["doc_name"],
                    "domain": item["domain"],
                    "queries": queries,
                    "pages": pages,
                    "text_layouts": text_layouts,
                    "fused_layouts": fused_layouts,
                    "estimated_fp32_seconds": cost,
                }
            )

    domains = []
    selected_documents: dict[str, dict[str, Any]] = {}
    overall_categories: Counter[str] = Counter()
    for domain in sorted(full_categories):
        candidates = choose_documents(
            [document for document in documents if document["domain"] == domain],
            queries_per_domain,
        )
        candidate_queries = [
            {**query, "doc_name": document["doc_name"]}
            for document in candidates
            for query in document["queries"]
        ]
        available = Counter(query["category"] for query in candidate_queries)
        quotas = allocate_categories(full_categories[domain], available, queries_per_domain)
        selected_queries = []
        for category, count in sorted(quotas.items()):
            pool = [query for query in candidate_queries if query["category"] == category]
            selected_queries.extend(evenly_spaced(pool, count))
        used_names = {query["doc_name"] for query in selected_queries}
        used_documents = [document for document in candidates if document["doc_name"] in used_names]
        selected_documents.update({document["doc_name"]: document for document in used_documents})
        category_counts = Counter(query["category"] for query in selected_queries)
        overall_categories.update(category_counts)
        domains.append(
            {
                "domain": domain,
                "query_ids": sorted(query["query_id"] for query in selected_queries),
                "category_counts": dict(sorted(category_counts.items())),
                "documents": [document["doc_name"] for document in used_documents],
            }
        )

    manifest_path = dataset_dir / "manifest.json"
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    selected = list(selected_documents.values())
    query_count = queries_per_domain * len(domains)
    return {
        "format": 1,
        "name": f"balanced_{query_count}",
        "description": (
            f"Deterministic {query_count}-query screening subset: {queries_per_domain} queries "
            "from each MMDocIR domain, stratified by question modality in a measured-cost document set."
        ),
        "dataset_revision": dataset_manifest.get("revision"),
        "query_count": query_count,
        "selection_policy": {
            "queries_per_domain": queries_per_domain,
            "document_policy": (
                "Minimum estimated FP32 batch=1 page/text/fused encoding cost among document "
                "sets supplying the target query count in each domain."
            ),
            "query_policy": (
                "Within selected documents, deterministic stratification approximating each "
                "domain's full question-modality distribution."
            ),
            "fp32_screening_rates": FP32_RATES,
            "intended_use": "FP32/INT8/INT4 screening; not a replacement for the full benchmark.",
        },
        "workload": {
            "documents": len(selected),
            "pages": sum(document["pages"] for document in selected),
            "text_equation_layouts": sum(document["text_layouts"] for document in selected),
            "image_table_layouts": sum(document["fused_layouts"] for document in selected),
            "layouts": sum(
                document["text_layouts"] + document["fused_layouts"] for document in selected
            ),
            "estimated_fp32_batch1_hours": sum(
                document["estimated_fp32_seconds"] for document in selected
            ) / 3600,
        },
        "domains": domains,
        "overall_category_counts": dict(sorted(overall_categories.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--queries-per-domain", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.queries_per_domain < 1:
        raise SystemExit("--queries-per-domain must be positive")
    manifest = build_manifest(args.dataset_dir, args.queries_per_domain)
    output = args.output or Path("subsets") / f"{manifest['name']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}: {manifest['workload']}")


if __name__ == "__main__":
    main()
