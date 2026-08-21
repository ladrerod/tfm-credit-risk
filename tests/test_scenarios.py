from __future__ import annotations

import unittest

import pandas as pd

from src.scenarios import MacroShock, Policy, evaluate_scenario, stress_probability


class ScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "pd": [0.01, 0.04, 0.07, 0.10],
                "ead_ratio": [1.0, 1.0, 1.0, 1.0],
                "lgd": [0.2, 0.3, 0.4, 0.5],
                "original_upb": [100000.0, 200000.0, 150000.0, 250000.0],
                "cltv": [70.0, 85.0, 95.0, 80.0],
                "dti": [30.0, 40.0, 45.0, 55.0],
            }
        )

    def test_conservative_policy_never_increases_retention(self) -> None:
        base = evaluate_scenario(
            self.frame,
            Policy("base", max_cltv=97, max_dti=50, max_pd=0.08),
            MacroShock("observed", 0, 0),
        )
        conservative = evaluate_scenario(
            self.frame,
            Policy("conservative", max_cltv=90, max_dti=43, max_pd=0.05),
            MacroShock("observed", 0, 0),
        )
        self.assertLessEqual(conservative["retained"], base["retained"])
        self.assertLessEqual(conservative["retained_exposure"], base["retained_exposure"])
        self.assertNotIn("approval_rate", base)

    def test_adverse_macro_shock_increases_probability(self) -> None:
        observed = stress_probability(self.frame["pd"], MacroShock("observed", 0, 0))
        stressed = stress_probability(self.frame["pd"], MacroShock("stress", 3, -15))
        self.assertTrue((stressed > observed).all())
        result = evaluate_scenario(
            self.frame,
            Policy("base", max_cltv=97, max_dti=50, max_pd=0.08),
            MacroShock("stress", 3, -15),
        )
        self.assertFalse(result["is_forecast"])
        self.assertFalse(result["contains_row_data"])


if __name__ == "__main__":
    unittest.main()
