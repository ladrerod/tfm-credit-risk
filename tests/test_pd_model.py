from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pd_model import PDConfig, train_and_select


class PDModelTests(unittest.TestCase):
    def _sample(self, seed: int, rows: int, year: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        score = rng.normal(710, 45, rows)
        cltv = rng.uniform(45, 105, rows)
        occupancy = rng.choice(["principal", "investor"], rows, p=[0.8, 0.2])
        logit = -3.2 - 0.018 * (score - 700) + 0.035 * (cltv - 80) + 0.7 * (occupancy == "investor")
        target = rng.binomial(1, 1 / (1 + np.exp(-logit)))
        return pd.DataFrame(
            {
                "credit_score": score,
                "cltv": cltv,
                "occupancy": occupancy,
                "cohort_year": year,
                "default_24m": target,
            }
        )

    def test_trains_calibrates_and_selects_without_test_input(self) -> None:
        config = PDConfig(
            numeric_features=("credit_score", "cltv"),
            categorical_features=("occupancy",),
            seed=42,
        )
        result = train_and_select(
            self._sample(1, 500, 2018),
            self._sample(2, 250, 2019),
            self._sample(3, 250, 2020),
            config,
        )
        self.assertIn(result["selected_name"], {"logistic", "hist_gradient_boosting"})
        self.assertEqual({"logistic", "hist_gradient_boosting"}, set(result["metrics"]))
        self.assertEqual(10, len(result["calibration_table"]))
        probability = result["selected_model"].predict_proba(self._sample(4, 10, 2021))[:, 1]
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())

    def test_rejects_outcome_leakage_features(self) -> None:
        config = PDConfig(
            numeric_features=("credit_score", "current_delinquency_status"),
            categorical_features=(),
            seed=42,
        )
        with self.assertRaisesRegex(ValueError, "leakage"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
