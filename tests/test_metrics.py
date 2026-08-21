from __future__ import annotations

import unittest

from src.metrics import calibration_table, classification_metrics


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


if __name__ == "__main__":
    unittest.main()
