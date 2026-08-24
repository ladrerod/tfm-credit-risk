from __future__ import annotations

import unittest

from src.metrics import calibration_table, classification_metrics, cohort_metrics


class MetricTests(unittest.TestCase):
    def test_metrics_reconcile_counts_and_probabilities(self) -> None:
        target = [0, 0, 1, 1]
        probability = [0.05, 0.20, 0.70, 0.90]
        result = classification_metrics(target, probability, threshold=0.5)
        self.assertEqual(4, sum(result["confusion"].values()))
        self.assertEqual(1.0, result["roc_auc"])
        self.assertGreater(result["pr_auc"], 0.9)
        table = calibration_table(target, probability, bins=2)
        self.assertEqual(4, sum(row["count"] for row in table))
        self.assertEqual(2, sum(row["events"] for row in table))

    def test_cohort_metrics_uses_selected_threshold(self) -> None:
        result = cohort_metrics([2022, 2022], [0, 1], [0.4, 0.6], threshold=0.7)

        self.assertEqual(1, result[0]["confusion"]["fn"])
        self.assertEqual(0.7, result[0]["threshold"])


if __name__ == "__main__":
    unittest.main()
