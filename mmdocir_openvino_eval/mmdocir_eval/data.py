"""MMDocIR annotations and memory-bounded parquet access."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


@dataclass(frozen=True)
class RowRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid inclusive row range: {self.start}..{self.end}")

    def contains(self, row_index: int) -> bool:
        return self.start <= row_index <= self.end


@dataclass(frozen=True)
class GroundTruthLayout:
    page_id: int
    page_size: tuple[float, float]
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Query:
    query_id: int
    text: str
    answer: str
    question_type: str
    relevant_page_ids: tuple[int, ...]
    relevant_layouts: tuple[GroundTruthLayout, ...]
    doc_name: str
    domain: str
    page_rows: RowRange
    layout_rows: RowRange


@dataclass(frozen=True)
class Document:
    doc_name: str
    domain: str
    page_rows: RowRange
    layout_rows: RowRange
    queries: tuple[Query, ...]


def _pair(value: Sequence[Any]) -> RowRange:
    if len(value) != 2:
        raise ValueError(f"expected [start, end], got {value!r}")
    return RowRange(int(value[0]), int(value[1]))


def load_annotations(
    path: Path,
    max_documents: int | None = None,
    max_queries: int | None = None,
    selected_query_ids: set[int] | None = None,
) -> tuple[Document, ...]:
    """Load annotations while preserving official global query ordering."""
    if selected_query_ids is not None and (max_documents is not None or max_queries is not None):
        raise ValueError("selected_query_ids cannot be combined with max_documents/max_queries")
    documents: list[Document] = []
    found_query_ids: set[int] = set()
    query_id = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            if max_documents is not None and len(documents) >= max_documents:
                break
            item = json.loads(line)
            page_rows = _pair(item["page_indices"])
            layout_rows = _pair(item.get("layout_indices", item.get("layout_indinces")))
            queries: list[Query] = []
            for qa in item["questions"]:
                if max_queries is not None and query_id >= max_queries:
                    break
                include_query = selected_query_ids is None or query_id in selected_query_ids
                if not include_query:
                    query_id += 1
                    continue
                layouts = tuple(
                    GroundTruthLayout(
                        page_id=int(layout["page"]),
                        page_size=tuple(float(value) for value in layout["page_size"]),
                        bbox=tuple(float(value) for value in layout["bbox"]),
                    )
                    for layout in qa["layout_mapping"]
                )
                queries.append(
                    Query(
                        query_id=query_id,
                        text=qa["Q"],
                        answer=qa["A"],
                        question_type=str(qa["type"]),
                        relevant_page_ids=tuple(int(value) for value in qa["page_id"]),
                        relevant_layouts=layouts,
                        doc_name=item["doc_name"],
                        domain=item["domain"],
                        page_rows=page_rows,
                        layout_rows=layout_rows,
                    )
                )
                found_query_ids.add(query_id)
                query_id += 1
            if queries:
                documents.append(
                    Document(
                        doc_name=item["doc_name"],
                        domain=item["domain"],
                        page_rows=page_rows,
                        layout_rows=layout_rows,
                        queries=tuple(queries),
                    )
                )
            if max_queries is not None and query_id >= max_queries:
                break
    if selected_query_ids is not None and found_query_ids != selected_query_ids:
        missing = sorted(selected_query_ids - found_query_ids)
        raise ValueError(f"query IDs not found in annotations: {missing}")
    return tuple(documents)


def selected_row_count(documents: Sequence[Document], kind: str) -> int:
    ranges = [document.page_rows if kind == "page" else document.layout_rows for document in documents]
    return sum(row_range.end - row_range.start + 1 for row_range in ranges)


def iter_parquet_rows(
    path: Path,
    columns: Sequence[str],
    documents: Sequence[Document],
    kind: str,
    batch_size: int = 256,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield selected global rows from parquet without loading all image binaries."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required; install requirements.txt") from error

    ranges = sorted(
        (document.page_rows if kind == "page" else document.layout_rows for document in documents),
        key=lambda value: value.start,
    )
    if not ranges:
        return
    range_index = 0
    row_group_offset = 0
    parquet_file = parquet.ParquetFile(path)
    for row_group_index in range(parquet_file.num_row_groups):
        row_group_rows = parquet_file.metadata.row_group(row_group_index).num_rows
        row_group_end = row_group_offset + row_group_rows - 1
        while range_index < len(ranges) and row_group_offset > ranges[range_index].end:
            range_index += 1
        if range_index >= len(ranges):
            return
        intersects = ranges[range_index].start <= row_group_end
        if intersects:
            batch_offset = 0
            for record_batch in parquet_file.iter_batches(
                batch_size=batch_size,
                row_groups=[row_group_index],
                columns=list(columns),
            ):
                rows = record_batch.to_pylist()
                for local_index, row in enumerate(rows):
                    global_index = row_group_offset + batch_offset + local_index
                    while range_index < len(ranges) and global_index > ranges[range_index].end:
                        range_index += 1
                    if range_index >= len(ranges):
                        return
                    if ranges[range_index].contains(global_index):
                        yield global_index, row
                batch_offset += len(rows)
        row_group_offset += row_group_rows