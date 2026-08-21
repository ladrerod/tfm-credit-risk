from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.expected_loss import compose_expected_loss, summarize_expected_loss


class ExpectedLossTests(unittest.TestCase):
    def test_composition_and_cohorts_reconcile(self) -> None:
        frame = pd.DataFrame({"original_upb": [100.0, 200.0], "cohort_year": [2021, 2022]})
        losses = compose_expected_loss([0.1, 0.2], [100.0, 200.0], [0.5, 0.25])
        self.assertTrue(np.allclose([5.0, 10.0], losses))
        result = summarize_expected_loss(
            frame,
            probability_default=[0.1, 0.2],
            ead_ratio=[1.0, 1.0],
            loss_given_default=[0.5, 0.25],
        )
        self.assertAlmostEqual(15.0, result["total_expected_loss"])
        self.assertAlmostEqual(15.0, sum(row["total_expected_loss"] for row in result["cohorts"]))
        self.assertFalse(result["contains_row_data"])

    def test_rejects_invalid_components(self) -> None:
        with self.assertRaises(ValueError):
            compose_expected_loss([1.1], [100.0], [0.5])


if __name__ == "__main__":
    unittest.main()
