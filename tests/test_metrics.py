from __future__ import annotations

import math
import unittest

import numpy as np

from src.metrics import (
    binned_distribution,
    classification_metrics,
    material_winner,
    population_stability_index,
    quantile_breaks,
    risk_band_cutoffs,
    risk_bands,
    summarize_year_metrics,
)


class MetricTests(unittest.TestCase):
    def test_classification_metrics_has_only_probability_metrics(self) -> None:
        result = classification_metrics([0, 0, 1, 1], [0.05, 0.2, 0.7, 0.9])

        self.assertEqual(
            {
                "n", "events", "prevalence", "roc_auc", "pr_auc", "ks", "brier", "log_loss",
                "calibration_intercept", "calibration_slope",
            },
            set(result),
        )
        self.assertNotIn("accuracy", result)
        self.assertNotIn("f1", result)
        self.assertNotIn("confusion", result)
        self.assertNotIn("threshold", result)

    def test_single_class_returns_na_only_where_metric_is_undefined(self) -> None:
        result = classification_metrics([0, 0], [0.2, 0.4])

        for name in ("roc_auc", "pr_auc", "ks", "calibration_intercept", "calibration_slope"):
            self.assertIsNone(result[name])
        self.assertEqual(0.1, result["brier"])
        self.assertGreater(result["log_loss"], 0)

    def test_risk_bands_use_frozen_p50_and_p90(self) -> None:
        cutoffs = risk_band_cutoffs([0.1, 0.3, 0.5, 0.7, 0.9])

        self.assertAlmostEqual(0.5, cutoffs[0])
        self.assertAlmostEqual(0.82, cutoffs[1])
        self.assertEqual(
            ["low", "medium", "medium", "medium", "high", "high"],
            risk_bands(
                [cutoffs[0] - 0.01, cutoffs[0], cutoffs[0] + 0.01, cutoffs[1] - 0.01, cutoffs[1], cutoffs[1] + 0.01],
                cutoffs,
            ).tolist(),
        )

    def test_psi_uses_frozen_quantiles_missing_bin_and_log_floor_only(self) -> None:
        breaks = quantile_breaks([1.0, 1.0, 1.0, 2.0, 2.0, np.nan], bins=4)
        distribution = binned_distribution([1.0, 2.0, np.nan], breaks)

        self.assertEqual([1.0, 2.0], breaks.tolist())
        self.assertAlmostEqual(1 / 3, distribution[-1])
        self.assertAlmostEqual(1.0, distribution.sum())
        self.assertAlmostEqual(2 * math.log(1_000_000), population_stability_index([1.0, 0.0], [0.0, 1.0]))

    def test_year_metrics_macro_summary_ignores_undefined_years(self) -> None:
        summary = summarize_year_metrics([
            {
                "cohort_year": 2018,
                "roc_auc": 0.9,
                "pr_auc": 0.8,
                "ks": 0.7,
                "brier": 0.1,
                "log_loss": 0.2,
                "calibration_intercept": -0.4,
                "calibration_slope": 0.9,
            },
            {
                "cohort_year": 2019,
                "roc_auc": 0.8,
                "pr_auc": 0.7,
                "ks": 0.6,
                "brier": 0.3,
                "log_loss": 0.4,
                "calibration_intercept": 0.8,
                "calibration_slope": 1.3,
            },
            {"cohort_year": 2020, "roc_auc": None, "brier": None},
        ])

        self.assertAlmostEqual(0.85, summary["roc_auc"]["macro_mean"])
        for metric in ("roc_auc", "pr_auc", "ks", "brier", "log_loss", "calibration_intercept", "calibration_slope"):
            self.assertEqual(2019, summary[metric]["worst_year"])

    def test_material_winner_preserves_the_frozen_paired_rule(self) -> None:
        baseline = [{"roc_auc": 0.70, "brier": 0.20, "log_loss": 0.30}] * 3
        challenger = [
            {"roc_auc": 0.71, "brier": 0.18, "log_loss": 0.30, "calibration_intercept": 0.0, "calibration_slope": 1.0},
            {"roc_auc": 0.72, "brier": 0.20, "log_loss": 0.28, "calibration_intercept": 0.2, "calibration_slope": 0.9},
            {"roc_auc": 0.70, "brier": 0.20, "log_loss": 0.30, "calibration_intercept": 0.4, "calibration_slope": 1.3},
        ]

        self.assertTrue(material_winner(challenger, baseline, require_calibration=True))
        one_win = [
            challenger[0],
            {**challenger[1], "roc_auc": 0.70, "log_loss": 0.30},
            challenger[2],
        ]
        self.assertFalse(material_winner(one_win, baseline, require_calibration=True))


if __name__ == "__main__":
    unittest.main()
