from __future__ import annotations

import unittest

import pandas as pd

from src.governance import audit_numeric_associations


class GovernanceTests(unittest.TestCase):
    def test_association_output_is_aggregate_only(self) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [4, 1, 3, 2]})
        result = audit_numeric_associations(frame, ["a", "b", "c"], strong_threshold=0.8)
        self.assertEqual(3, len(result["pairs"]))
        self.assertEqual(1, len(result["strong_pairs"]))
        self.assertFalse(result["contains_row_data"])


if __name__ == "__main__":
    unittest.main()
