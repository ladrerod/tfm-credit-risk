from __future__ import annotations

import unittest

import pandas as pd

from src.macro import build_state_month_features, merge_macro


class MacroTests(unittest.TestCase):
    def test_features_precede_origination(self) -> None:
        dates = pd.date_range("2019-01-01", "2021-12-01", freq="MS")
        laus = pd.DataFrame({"state": "CA", "date": dates, "unemployment": range(len(dates))})
        hpi = pd.DataFrame(
            {
                "state": "CA",
                "year": [2019, 2019, 2019, 2019, 2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021],
                "quarter": [1, 2, 3, 4] * 3,
                "hpi": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            }
        )
        macro = build_state_month_features(laus, hpi, 2021, 2021)
        loans = pd.DataFrame({"state": ["CA"], "origination_date": pd.to_datetime(["2021-05-01"])})
        merged = merge_macro(loans, macro)
        self.assertLess(merged.loc[0, "macro_asof_date"], merged.loc[0, "origination_date"])
        self.assertAlmostEqual(107 / 103 - 1, merged.loc[0, "hpi_yoy"])

    def test_merge_rejects_future_information(self) -> None:
        loans = pd.DataFrame({"state": ["CA"], "origination_date": pd.to_datetime(["2021-05-01"])})
        macro = pd.DataFrame(
            {
                "state": ["CA"], "year": [2021], "month": [5],
                "unemployment_3m": [5.0], "unemployment_change_12m": [1.0],
                "hpi_yoy": [0.1], "macro_asof_date": pd.to_datetime(["2021-05-15"]),
            }
        )
        with self.assertRaisesRegex(ValueError, "future"):
            merge_macro(loans, macro)


if __name__ == "__main__":
    unittest.main()
