import math
import unittest

from mmdocir_eval.metrics import LayoutCandidate, intersection_area, layout_metrics, page_metrics


class PageMetricsTest(unittest.TestCase):
    def test_recall_and_binary_ndcg(self) -> None:
        result = page_metrics([9, 2, 7, 4], {2, 4}, ks=(1, 2, 4))

        self.assertEqual(result["recall@1"], 0.0)
        self.assertEqual(result["recall@2"], 0.5)
        self.assertEqual(result["recall@4"], 1.0)
        expected_dcg = 1 / math.log2(3) + 1 / math.log2(5)
        ideal_dcg = 1 + 1 / math.log2(3)
        self.assertAlmostEqual(result["ndcg@4"], expected_dcg / ideal_dcg)


class LayoutMetricsTest(unittest.TestCase):
    def test_intersection_area(self) -> None:
        self.assertEqual(intersection_area((0, 0, 4, 4), (2, 1, 5, 3)), 4.0)
        self.assertEqual(intersection_area((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)

    def test_area_recall_and_graded_ndcg(self) -> None:
        target = LayoutCandidate(3, (0, 0, 10, 10))
        irrelevant = LayoutCandidate(2, (0, 0, 10, 10))
        quarter = LayoutCandidate(3, (0, 0, 5, 5))
        full = LayoutCandidate(3, (0, 0, 10, 10))

        result = layout_metrics(
            [irrelevant, quarter, full], [irrelevant, quarter, full], [target], ks=(1, 2, 3)
        )

        self.assertEqual(result["recall@1"], 0.0)
        self.assertEqual(result["recall@2"], 0.25)
        self.assertEqual(result["recall@3"], 1.25)
        expected_dcg = 25 / math.log2(3) + 100 / math.log2(4)
        ideal_dcg = 100 + 25 / math.log2(3)
        self.assertAlmostEqual(result["ndcg@3"], expected_dcg / ideal_dcg)

    def test_open_domain_layout_identity_includes_document(self) -> None:
        target = LayoutCandidate(3, (0, 0, 10, 10), "relevant.pdf")
        wrong_document = LayoutCandidate(3, (0, 0, 10, 10), "other.pdf")

        result = layout_metrics([wrong_document], [wrong_document], [target], ks=(1,))

        self.assertEqual(result["recall@1"], 0.0)
        self.assertEqual(result["ndcg@1"], 0.0)


if __name__ == "__main__":
    unittest.main()