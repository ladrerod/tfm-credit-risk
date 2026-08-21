from __future__ import annotations

import unittest

import pandas as pd

from src.governance import audit_numeric_associations, explain_linear


class GovernanceTests(unittest.TestCase):
    def test_linear_contributions_reconcile_score(self) -> None:
        result = explain_linear([2.0, -1.0], [3.0, 4.0], intercept=0.5, names=["a", "b"])
        self.assertAlmostEqual(2.5, result["score"])
        self.assertAlmostEqual(2.0, sum(row["contribution"] for row in result["contributions"]))

    def test_association_output_is_aggregate_only(self) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [4, 1, 3, 2]})
        result = audit_numeric_associations(frame, ["a", "b", "c"], strong_threshold=0.8)
        self.assertEqual(3, len(result["pairs"]))
        self.assertEqual(1, len(result["strong_pairs"]))
        self.assertFalse(result["contains_row_data"])


if __name__ == "__main__":
    unittest.main()
