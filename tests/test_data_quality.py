from __future__ import annotations

import unittest

import pandas as pd

from src.data_quality import mature_at_horizon, summarize_eda, validate_frame


class DataQualityTests(unittest.TestCase):
    def test_maturity_requires_complete_horizon(self) -> None:
        frame = pd.DataFrame(
            {
                "origination_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
                "performance_end_date": pd.to_datetime(["2022-01-01", "2021-12-01"]),
            }
        )
        self.assertEqual([True, False], mature_at_horizon(frame, 24).tolist())

    def test_validation_rejects_duplicate_keys_and_ranges(self) -> None:
        duplicate = pd.DataFrame({"record_key": [1, 1], "cltv": [80.0, 101.0]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_frame(duplicate, required=("record_key", "cltv"), ranges={"cltv": (0, 200)})
        outside = pd.DataFrame({"record_key": [1], "cltv": [201.0]})
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_frame(outside, required=("record_key", "cltv"), ranges={"cltv": (0, 200)})

    def test_eda_contains_only_aggregates(self) -> None:
        frame = pd.DataFrame(
            {
                "cohort_year": [2020, 2020, 2021],
                "default_24m": [0, 1, 0],
                "credit_score": [700.0, None, 760.0],
                "cltv": [80.0, 90.0, 70.0],
            }
        )
        result = summarize_eda(frame, target="default_24m", cohort="cohort_year")
        self.assertEqual(3, result["rows"])
        self.assertAlmostEqual(1 / 3, result["event_rate"])
        self.assertAlmostEqual(1 / 3, result["missingness"]["credit_score"])
        self.assertEqual({2020, 2021}, {row["cohort"] for row in result["cohorts"]})
        self.assertNotIn("records", result)


if __name__ == "__main__":
    unittest.main()
