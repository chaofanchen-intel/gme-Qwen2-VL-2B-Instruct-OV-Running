import unittest
import tempfile
from argparse import Namespace
from pathlib import Path

from run_benchmark import load_subset, package_identity, quality_deltas, resolve_batches


class SummaryTest(unittest.TestCase):
    def test_quality_delta_uses_fp32_as_baseline(self) -> None:
        runs = {
            "fp32": {"evaluation": {"close": {"page": {"recall@5": 0.8}, "layout": {}}}},
            "int8": {"evaluation": {"close": {"page": {"recall@5": 0.78}, "layout": {}}}},
        }

        delta = quality_deltas(runs)["precisions"]["int8"]["close"]["page"]["recall@5"]

        self.assertAlmostEqual(delta["absolute"], -0.02)
        self.assertAlmostEqual(delta["percentage_points"], -2.0)
        self.assertAlmostEqual(delta["relative_percent"], -2.5)

    def test_package_identity_accepts_non_gme_config_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "qwen3_config.json").write_text('{"model": "qwen3"}', encoding="utf-8")

            identity = package_identity(package)

        self.assertIn("qwen3_config.json", identity["configs"])
        self.assertEqual(identity["configs"]["qwen3_config.json"]["bytes"], 18)

    def test_load_subset_validates_declared_query_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subset.json"
            path.write_text(
                '{"name":"test","query_count":2,"domains":[{"query_ids":[3,7]}]}',
                encoding="utf-8",
            )

            provenance, query_ids = load_subset(path)

        self.assertEqual(provenance["name"], "test")
        self.assertEqual(query_ids, {3, 7})

    def test_recommended_batches_depend_on_precision(self) -> None:
        args = Namespace(
            batch_profile="recommended", batch_size=1, query_batch_size=None,
            page_batch_size=None, text_layout_batch_size=None, fused_layout_batch_size=None,
        )

        self.assertEqual(
            resolve_batches(args, "fp32"),
            {"query": 8, "page": 1, "text_layout": 1, "fused_layout": 1},
        )
        self.assertEqual(
            resolve_batches(args, "int8_causal_vopt"),
            {"query": 8, "page": 1, "text_layout": 4, "fused_layout": 1},
        )

    def test_explicit_batch_overrides_recommended_profile(self) -> None:
        args = Namespace(
            batch_profile="recommended", batch_size=1, query_batch_size=2,
            page_batch_size=None, text_layout_batch_size=None, fused_layout_batch_size=3,
        )

        self.assertEqual(
            resolve_batches(args, "int4_causal_vopt"),
            {"query": 2, "page": 1, "text_layout": 4, "fused_layout": 3},
        )


if __name__ == "__main__":
    unittest.main()