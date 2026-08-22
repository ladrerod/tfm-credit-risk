from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.loss_models import HurdleLGD, build_huber_regressor, regression_metrics, train_loss_models


class LossModelTests(unittest.TestCase):
    def _sample(self, seed: int, rows: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        cltv = rng.uniform(45, 120, rows)
        score = rng.normal(700, 50, rows)
        occupancy = rng.choice(["principal", "investor"], rows, p=[0.8, 0.2])
        positive_probability = 1 / (1 + np.exp(-(-4.0 + 0.055 * (cltv - 80) + 0.5 * (occupancy == "investor"))))
        positive = rng.binomial(1, positive_probability)
        lgd = positive * np.clip(0.35 + 0.006 * (cltv - 80) + rng.normal(0, 0.08, rows), 0.05, 1.4)
        ead_ratio = np.clip(1 + rng.normal(0, 0.02, rows), 0.8, 1.2)
        return pd.DataFrame(
            {
                "original_cltv": cltv,
                "credit_score": score,
                "occupancy": occupancy,
                "lgd": lgd,
                "ead_ratio": ead_ratio,
            }
        )

    def test_hurdle_recovers_zero_inflated_portfolio_loss(self) -> None:
        development = self._sample(1, 4000)
        validation = self._sample(2, 2000)
        features = ["original_cltv", "credit_score", "occupancy"]
        direct = build_huber_regressor(
            numeric=["original_cltv", "credit_score"], categorical=["occupancy"]
        ).fit(development[features], development["lgd"])
        hurdle = HurdleLGD(
            numeric=["original_cltv", "credit_score"], categorical=["occupancy"], seed=42
        ).fit(development[features], development["lgd"])
        direct_error = regression_metrics(
            validation["lgd"], np.clip(direct.predict(validation[features]), 0, 2)
        )["portfolio_relative_error"]
        hurdle_error = regression_metrics(validation["lgd"], hurdle.predict(validation[features]))[
            "portfolio_relative_error"
        ]
        self.assertLess(hurdle_error, direct_error)

    def test_selection_reports_ead_lgd_and_readiness(self) -> None:
        result = train_loss_models(
            self._sample(3, 2000),
            self._sample(4, 1000),
            numeric=["original_cltv", "credit_score"],
            categorical=["occupancy"],
            seed=42,
        )
        self.assertIn(result["ead"]["selected_name"], {"constant_1", "hist_gradient_boosting"})
        self.assertIn(result["lgd"]["selected_name"], {"segment_mean", "direct_huber", "hurdle"})
        self.assertIsInstance(result["decision_grade"], bool)

    def test_ead_and_lgd_can_use_different_maturity_populations(self) -> None:
        ead_development = self._sample(5, 2000).drop(columns="lgd")
        ead_validation = self._sample(6, 1000).drop(columns="lgd")
        lgd_development = self._sample(7, 1800).drop(columns="ead_ratio")
        lgd_validation = self._sample(8, 900).drop(columns="ead_ratio")

        result = train_loss_models(
            ead_development,
            ead_validation,
            lgd_development=lgd_development,
            lgd_validation=lgd_validation,
            numeric=["original_cltv", "credit_score"],
            categorical=["occupancy"],
            seed=42,
        )

        self.assertEqual(1000, result["ead"]["metrics"][result["ead"]["selected_name"]]["n"])
        self.assertEqual(900, result["lgd"]["metrics"][result["lgd"]["selected_name"]]["n"])

    def test_selection_omits_hurdle_when_all_losses_are_positive(self) -> None:
        development = self._sample(9, 1000)
        validation = self._sample(10, 500)
        development["lgd"] = development["lgd"].clip(lower=0.1)
        validation["lgd"] = validation["lgd"].clip(lower=0.1)

        result = train_loss_models(
            development,
            validation,
            numeric=["original_cltv", "credit_score"],
            categorical=["occupancy"],
            seed=42,
        )

        self.assertNotIn("hurdle", result["lgd"]["metrics"])


if __name__ == "__main__":
    unittest.main()
