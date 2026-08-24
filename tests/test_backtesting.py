from __future__ import annotations

import json
import unittest

import pandas as pd

from src.backtesting import run_walk_forward
from src.pd_model import PDConfig


FOLD = {
    "name": "oot_2020",
    "as_of_date": "2020-01-01",
    "development_years": [2015],
    "calibration_year": 2016,
    "validation_year": 2017,
    "test_year": 2020,
}


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in (2015, 2016, 2017, 2020):
        for position in range(4):
            event = position % 2
            available = pd.Timestamp(year + 2, 1 + position, 1)
            event_date = pd.Timestamp(year + 1, 1 + position, 1) if event else pd.NaT
            rows.append(
                {
                    "cohort_year": year,
                    "x": float(position + event + (year - 2015) / 10),
                    "original_cltv": 70.0 + 5 * position,
                    "original_upb": 100_000.0,
                    "default_24m": event,
                    "ead_ratio": 0.95 + 0.02 * position if event else float("nan"),
                    "lgd": 0.20 + 0.03 * position if event else float("nan"),
                    "lgd_eligible": bool(event),
                    "source_cutoff_date": pd.Timestamp("2026-03-01"),
                    "pd_label_available_date": available,
                    "ead_label_available_date": event_date,
                    "lgd_label_available_date": (
                        event_date + pd.DateOffset(months=4) if event else pd.NaT
                    ),
                }
            )
    rows.append(
        {
            **rows[0],
            "x": 999.0,
            "default_24m": 1,
            "ead_ratio": 1.1,
            "lgd": 0.8,
            "lgd_eligible": True,
            "pd_label_available_date": pd.Timestamp("2020-02-01"),
            "ead_label_available_date": pd.Timestamp("2020-02-01"),
            "lgd_label_available_date": pd.Timestamp("2020-06-01"),
        }
    )
    return pd.DataFrame(rows)


class WalkForwardTests(unittest.TestCase):
    def _run(
        self, frame: pd.DataFrame | None = None, *, minimum_lgd_rows: int = 100
    ) -> dict[str, object]:
        return run_walk_forward(
            _frame() if frame is None else frame,
            [FOLD],
            PDConfig(("x",), (), 7),
            numeric=["x", "original_cltv"],
            categorical=[],
            seed=7,
            minimum_ead_rows=2,
            minimum_lgd_rows=minimum_lgd_rows,
        )

    def test_future_labels_are_excluded_from_model_roles(self) -> None:
        original = _frame()
        changed = original.copy()
        changed.loc[changed["x"].eq(999), ["x", "default_24m", "ead_ratio", "lgd"]] = [
            -999.0,
            0,
            0.1,
            1.9,
        ]

        before = self._run(original)
        after = self._run(changed)

        self.assertEqual(4, before["folds"][0]["pd"]["development_rows"])
        self.assertEqual(before, after)

    def test_invalid_temporal_roles_are_rejected(self) -> None:
        bad = {**FOLD, "validation_year": FOLD["test_year"]}
        with self.assertRaisesRegex(ValueError, "before test"):
            run_walk_forward(
                _frame(),
                [bad],
                PDConfig(("x",), (), 7),
                numeric=["x"],
                categorical=[],
                seed=7,
                minimum_ead_rows=2,
                minimum_lgd_rows=100,
            )

    def test_as_of_cannot_follow_test_start(self) -> None:
        bad = {**FOLD, "as_of_date": "2020-02-01"}
        with self.assertRaisesRegex(ValueError, "test start"):
            run_walk_forward(
                _frame(),
                [bad],
                PDConfig(("x",), (), 7),
                numeric=["x"],
                categorical=[],
                seed=7,
                minimum_ead_rows=2,
                minimum_lgd_rows=100,
            )

    def test_pd_and_ead_are_evaluated_without_claiming_thin_lgd(self) -> None:
        fold = self._run()["folds"][0]

        self.assertEqual("evaluated", fold["pd"]["status"])
        self.assertEqual("evaluated", fold["ead"]["status"])
        self.assertEqual("insufficient_evidence", fold["lgd"]["status"])
        self.assertEqual("unavailable", fold["expected_loss"]["status"])
        self.assertEqual(2, fold["ead"]["test_rows"])
        self.assertIn("test_metrics", fold["pd"])
        self.assertIn("test_metrics", fold["ead"])

    def test_one_class_returns_insufficient_evidence(self) -> None:
        frame = _frame()
        frame.loc[frame["cohort_year"].eq(2016), "default_24m"] = 0

        fold = self._run(frame)["folds"][0]

        self.assertEqual("insufficient_evidence", fold["pd"]["status"])
        self.assertIn("both outcomes", fold["pd"]["reason"])

    def test_output_is_aggregate_and_has_no_global_approval(self) -> None:
        result = self._run()
        encoded = json.dumps(result)

        self.assertFalse(result["contains_row_data"])
        self.assertNotIn("decision_grade", encoded)
        self.assertNotIn("loan_key", encoded)
        self.assertNotIn("probabilities", encoded)
        self.assertNotIn("labels", encoded)

    def test_contract_requires_temporal_lineage(self) -> None:
        with self.assertRaisesRegex(ValueError, "temporal columns"):
            self._run(_frame().drop(columns="ead_label_available_date"))

    def test_test_is_not_evaluated_before_source_maturity(self) -> None:
        frame = _frame()
        frame.loc[frame["cohort_year"].eq(2020), "source_cutoff_date"] = pd.Timestamp(
            "2021-01-01"
        )

        fold = self._run(frame)["folds"][0]

        self.assertEqual("insufficient_evidence", fold["pd"]["status"])
        self.assertEqual(0, fold["pd"]["test_rows"])

    def test_expected_loss_requires_complete_eligible_recoveries(self) -> None:
        complete = self._run(minimum_lgd_rows=2)["folds"][0]["expected_loss"]
        frame = _frame()
        test_defaults = frame["cohort_year"].eq(2020) & frame["default_24m"].eq(1)
        frame.loc[frame.index[test_defaults][0], "lgd_eligible"] = False
        incomplete = self._run(frame, minimum_lgd_rows=1)["folds"][0]["expected_loss"]

        self.assertEqual("evaluated", complete["status"])
        self.assertEqual("unavailable", incomplete["status"])
        self.assertEqual(0.5, incomplete["realized_loss_coverage"])


if __name__ == "__main__":
    unittest.main()
