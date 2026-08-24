from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.monthly_model import (
    CANONICAL_STATES,
    MultiStateConfig,
    cumulative_incidence,
    multistate_metrics,
    train_and_compare_multistate,
)


def _cohort(prefix: str, offset: int) -> pd.DataFrame:
    rows = []
    for repeat in range(5):
        for index, state in enumerate(CANONICAL_STATES):
            rows.append(
                {
                    "loan_key": f"{prefix}-{repeat}-{state}",
                    "current_state": "current" if state in {"default", "prepay"} else state,
                    "next_state": state,
                    "loan_age_months": repeat + 1,
                    "months_since_first_payment": repeat,
                    "monthly_ead_ratio": 1 - index / 100 + offset / 10_000,
                    "current_interest_rate": 3 + index / 10,
                    "original_cltv": 70 + index,
                    "original_dti": 30 + index,
                    "origination_fico": 700 + index,
                    "occupancy_status": "P" if index % 2 else "I",
                    "property_state": "TX" if index % 2 else "CA",
                }
            )
    return pd.DataFrame(rows)


class MonthlyModelTests(unittest.TestCase):
    def test_cumulative_incidence_reconciles_known_hazards(self) -> None:
        result = cumulative_incidence(np.array([0.2, 0.2]), np.array([0.1, 0.1]))

        np.testing.assert_allclose(result["survival"], [0.7, 0.49])
        np.testing.assert_allclose(result["default"], [0.2, 0.34])
        np.testing.assert_allclose(result["prepay"], [0.1, 0.17])

    def test_trains_calibrated_champion_and_challenger(self) -> None:
        config = MultiStateConfig(seed=7)

        result = train_and_compare_multistate(
            _cohort("development", 0), _cohort("calibration", 10), _cohort("validation", 20), config
        )

        self.assertEqual("multinomial_logistic", result["champion_name"])
        self.assertEqual({"multinomial_logistic", "hist_gradient_boosting"}, set(result["validation_metrics"]))
        self.assertEqual(set(CANONICAL_STATES), set(result["validation_metrics"]["multinomial_logistic"]["observed_state_counts"]))
        self.assertFalse(result["calibration_adequacy"]["suitable"])
        self.assertEqual(5, result["class_counts"]["calibration"]["default"])

    def test_metrics_are_aggregate_and_include_cure_and_incidence(self) -> None:
        frame = _cohort("validation", 0)
        probability = np.full((len(frame), len(CANONICAL_STATES)), 1 / len(CANONICAL_STATES))

        result = multistate_metrics(frame, probability, CANONICAL_STATES)

        self.assertEqual(len(frame), sum(result["observed_state_counts"].values()))
        self.assertIn("default", result["one_vs_rest"])
        self.assertIn("prepay", result["one_vs_rest"])
        self.assertIn("cures", result)
        self.assertIn("cumulative_incidence", result)
        self.assertTrue(result["cumulative_incidence"]["available"])
        self.assertIn("excludes 90_plus", result["cumulative_incidence"]["default_definition"])
        self.assertNotIn("loan_key", str(result))

    def test_rejects_missing_states_and_overlapping_cohorts(self) -> None:
        config = MultiStateConfig()
        development = _cohort("shared", 0)
        calibration = _cohort("calibration", 0)
        validation = _cohort("validation", 0)

        with self.assertRaisesRegex(ValueError, "overlap"):
            train_and_compare_multistate(development, development.copy(), validation, config)
        with self.assertRaisesRegex(ValueError, "missing canonical next states"):
            train_and_compare_multistate(
                development.loc[development["next_state"].ne("prepay")], calibration, validation, config
            )

        result = train_and_compare_multistate(
            development, calibration, validation.loc[validation["next_state"].ne("prepay")], config
        )
        self.assertEqual("multinomial_logistic", result["champion_name"])

    def test_incidence_rejects_gapped_monthly_horizons(self) -> None:
        frame = _cohort("validation", 0)
        frame.loc[frame["months_since_first_payment"].eq(1), "months_since_first_payment"] = 3
        probability = np.full((len(frame), len(CANONICAL_STATES)), 1 / len(CANONICAL_STATES))

        result = multistate_metrics(frame, probability, CANONICAL_STATES)

        self.assertFalse(result["cumulative_incidence"]["available"])

    def test_metrics_reject_invalid_current_state(self) -> None:
        frame = _cohort("validation", 0)
        frame.loc[0, "current_state"] = "unknown"
        probability = np.full((len(frame), len(CANONICAL_STATES)), 1 / len(CANONICAL_STATES))

        with self.assertRaisesRegex(ValueError, "invalid current states"):
            multistate_metrics(frame, probability, CANONICAL_STATES)


if __name__ == "__main__":
    unittest.main()
