import json
import tempfile
import unittest
from pathlib import Path

from mmdocir_eval.data import RowRange, load_annotations, selected_row_count


class AnnotationTest(unittest.TestCase):
    def test_loads_ranges_and_global_query_ids(self) -> None:
        items = [
            {
                "doc_name": "a.pdf",
                "domain": "test",
                "page_indices": [10, 12],
                "layout_indices": [30, 33],
                "questions": [
                    {
                        "Q": "where",
                        "A": "there",
                        "type": "text",
                        "page_id": [1],
                        "layout_mapping": [
                            {"page": 1, "page_size": [100, 200], "bbox": [1, 2, 3, 4]}
                        ],
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in items), encoding="utf-8")
            documents = load_annotations(path)

        self.assertEqual(documents[0].page_rows, RowRange(10, 12))
        self.assertEqual(documents[0].queries[0].query_id, 0)
        self.assertEqual(documents[0].queries[0].relevant_layouts[0].bbox, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(selected_row_count(documents, "page"), 3)
        self.assertEqual(selected_row_count(documents, "layout"), 4)

    def test_selects_global_query_ids_and_keeps_owning_documents(self) -> None:
        items = []
        for index in range(3):
            items.append(
                {
                    "doc_name": f"{index}.pdf",
                    "domain": "test",
                    "page_indices": [index, index],
                    "layout_indices": [index, index],
                    "questions": [
                        {
                            "Q": f"q{index}", "A": "a", "type": "text", "page_id": [0],
                            "layout_mapping": [
                                {"page": 0, "page_size": [1, 1], "bbox": [0, 0, 1, 1]}
                            ],
                        }
                    ],
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in items), encoding="utf-8")
            documents = load_annotations(path, selected_query_ids={0, 2})

        self.assertEqual([document.doc_name for document in documents], ["0.pdf", "2.pdf"])
        self.assertEqual(
            [query.query_id for document in documents for query in document.queries], [0, 2]
        )

    def test_rejects_unknown_selected_query_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "query IDs not found"):
                load_annotations(path, selected_query_ids={99})


if __name__ == "__main__":
    unittest.main()