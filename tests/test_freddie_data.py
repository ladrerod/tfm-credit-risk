from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from src.data_access import read_csv_zst
from src.freddie_data import prepare_dataset, prepare_quarter


class FreddieDataTests(unittest.TestCase):
    @staticmethod
    def _origin(
        loan: str,
        upb: str = "100000",
        *,
        width: int = 31,
        maturity: str = "204412",
        high_balance: str = "N",
    ) -> str:
        row = [""] * width
        for position, value in {
            0: "720", 1: "201501", 3: maturity, 5: "0", 6: "1", 7: "P", 8: "80", 9: "35",
            10: upb, 11: "80", 12: "4", 15: "FRM", 16: "CA", 17: "SF", 19: loan,
            20: "P", 21: "360", 22: "1", (24 if width == 31 else 25): high_balance,
        }.items():
            row[position] = value
        return "|".join(row)

    @staticmethod
    def _performance(
        loan: str,
        period: str,
        *,
        width: int = 35,
        upb: str = "100000",
        delinquency: str = "00",
        zero_balance_code: str = "",
        actual_loss: str = "",
        defect_settlement: str = "",
        components: tuple[str, str, str, str, str, str] | None = None,
    ) -> str:
        row = [""] * width
        row[:6] = [loan, period, upb, delinquency, "1", "359"]
        row[8] = zero_balance_code
        row[9] = period if zero_balance_code else ""
        row[6] = defect_settlement
        if components:
            mi, sale, non_mi, expenses, removal, interest = components
            row[13:17] = [mi, sale, non_mi, expenses]
            row[26:28] = [removal, interest]
        row[21] = actual_loss
        return "|".join(row)

    def _archive(self, path: Path, origin: list[str], performance: list[str], *, legacy: bool = False) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            if legacy:
                archive.writestr("historical_data1_Q12015.txt", "\n".join(origin) + "\n")
                archive.writestr("historical_data1_time_Q12015.txt", "\n".join(performance) + "\n")
            else:
                archive.writestr("orig_2015Q1.txt", "\n".join(origin) + "\n")
                archive.writestr("perf_2015Q1.txt", "\n".join(performance) + "\n")

    def test_prepares_states_cures_monthly_ead_and_resolved_lgd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, panel = root / "historical_data_2015Q1.zip", root / "monthly.csv.zst"
            rows = [
                self._performance("L1", "201501", upb="100000", delinquency="00"),
                self._performance("L1", "201502", upb="95000", delinquency="01"),
                self._performance("L1", "201503", upb="90000", delinquency="02"),
                self._performance("L1", "201504", upb="90000", delinquency="03"),
                self._performance("L1", "201505", upb="90000", delinquency="00"),
                self._performance("L1", "201506", upb="90000", delinquency="RA"),
                self._performance("L1", "201801", upb="0", delinquency="00", zero_balance_code="09", actual_loss="35000", components=("-10000", "-40000", "-5000", "5000", "80000", "5000")),
                self._performance("L2", "202603", upb="50000"),
            ]
            self._archive(archive, [self._origin("L1"), self._origin("L2")], rows)

            loans, metadata = prepare_quarter(archive, panel, sample_size=10, seed=7, compression_level=3)
            monthly = pd.concat(read_csv_zst(panel, chunksize=10), ignore_index=True)

            self.assertEqual(["current", "30", "60", "90_plus", "current"], monthly["current_state"].tolist()[:5])
            self.assertTrue(monthly["is_cure"].any())
            self.assertEqual(0.9, loans.loc[loans["default_24m"].eq(1), "ead_ratio"].iloc[0])
            self.assertAlmostEqual(35000 / 90000, loans.loc[loans["default_24m"].eq(1), "lgd"].iloc[0])
            self.assertEqual(0.9, monthly.loc[monthly["current_state"].eq("default"), "monthly_ead_ratio"].iloc[0])
            self.assertNotIn("loan_identifier", monthly.columns)
            self.assertIn("loan_key", monthly.columns)
            self.assertEqual(set(loans["loan_key"]), set(monthly["loan_key"]))
            self.assertTrue({
                "cohort_year", "origination_date", "performance_date", "loan_age_months",
                "current_upb", "monthly_ead_ratio", "current_interest_rate", "remaining_months",
                "modification_flag", "current_state", "next_state", "is_cure", "is_rollback",
                "consecutive_month", "original_interest_rate", "property_state", "zero_balance_code",
                "actual_loss",
            }.issubset(monthly.columns))
            self.assertEqual("2026-03", metadata["performance_cutoff"])

    def test_supports_legacy_34_fields_latest_loss_and_signed_lgd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, panel = root / "historical_data_2015Q1.zip", root / "monthly.csv.zst"
            rows = [
                self._performance("L1", "201503", width=34, upb="90000", delinquency="03"),
                self._performance("L1", "201504", width=34, upb="0", zero_balance_code="09", actual_loss="999", components=("-10000", "-40000", "-5000", "5000", "80000", "5000")),
                self._performance("L1", "201801", width=34, upb="0", zero_balance_code="09", actual_loss="35000", components=("-10000", "-40000", "-5000", "5000", "80000", "5000")),
                self._performance("L2", "202603", width=34),
            ]
            self._archive(
                archive,
                [self._origin("L1", width=32, high_balance="Y"), self._origin("L2", width=32)],
                rows,
                legacy=True,
            )

            loans, _ = prepare_quarter(archive, panel, sample_size=10, seed=7, compression_level=3)

            defaulted = loans.loc[loans["default_24m"].eq(1)].iloc[0]
            self.assertTrue(defaulted["lgd_eligible"])
            self.assertAlmostEqual(35000 / 90000, defaulted["lgd"])
            self.assertEqual("Y", defaulted["high_balance_loan"])

    def test_legacy_positive_components_subtract_expenses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, panel = root / "historical_data_2015Q1.zip", root / "monthly.csv.zst"
            rows = [
                self._performance("L1", "201503", width=32, upb="100000", delinquency="03"),
                self._performance(
                    "L1",
                    "201504",
                    width=32,
                    upb="0",
                    zero_balance_code="09",
                    actual_loss="71000",
                    components=("10000", "10000", "5000", "5000", "100000", "1000"),
                ),
                self._performance("L2", "202603", width=32),
            ]
            self._archive(
                archive,
                [self._origin("L1", width=32), self._origin("L2", width=32)],
                rows,
                legacy=True,
            )

            loans, _ = prepare_quarter(archive, panel, sample_size=10, seed=7, compression_level=3)

            defaulted = loans.loc[loans["default_24m"].eq(1)].iloc[0]
            self.assertTrue(defaulted["lgd_eligible"])
            self.assertAlmostEqual(0.71, defaulted["lgd"])

    def test_rejects_noncontiguous_loan_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, panel = root / "historical_data_2015Q1.zip", root / "monthly.csv.zst"
            rows = [
                self._performance("L1", "201501"),
                self._performance("L2", "201501"),
                self._performance("L1", "201502"),
            ]
            self._archive(archive, [self._origin("L1"), self._origin("L2")], rows)

            with self.assertRaisesRegex(ValueError, "contiguous"):
                prepare_quarter(archive, panel, sample_size=10, seed=7, compression_level=3)

    def test_rejects_unordered_periods_within_a_loan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, panel = root / "historical_data_2015Q1.zip", root / "monthly.csv.zst"
            rows = [
                self._performance("L1", "201502"),
                self._performance("L1", "201501"),
            ]
            self._archive(archive, [self._origin("L1")], rows)

            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                prepare_quarter(archive, panel, sample_size=10, seed=7, compression_level=3)

    def test_pre_first_payment_rows_do_not_create_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, panel = root / "historical_data_2015Q1.zip", root / "monthly.csv.zst"
            rows = [
                self._performance("L1", "201412"),
                self._performance("L1", "201501", delinquency="01"),
                self._performance("L1", "201502", delinquency="00"),
            ]
            self._archive(archive, [self._origin("L1")], rows)

            prepare_quarter(archive, panel, sample_size=10, seed=7, compression_level=3)
            monthly = pd.concat(read_csv_zst(panel, chunksize=10), ignore_index=True)

            before_first_payment = monthly.loc[monthly["months_since_first_payment"].lt(0)].iloc[0]
            self.assertFalse(before_first_payment["consecutive_month"])
            self.assertTrue(pd.isna(before_first_payment["next_state"]))
            self.assertFalse(before_first_payment["is_cure"])
            self.assertFalse(before_first_payment["is_rollback"])

    def test_terminal_state_absorbs_later_corrections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, panel = root / "historical_data_2015Q1.zip", root / "monthly.csv.zst"
            rows = [
                self._performance("L1", "201501", upb="100000", delinquency="00"),
                self._performance("L1", "201502", upb="0", zero_balance_code="01"),
                self._performance("L1", "201503", upb="0", delinquency="00"),
                self._performance("L1", "201504", upb="0", delinquency="01"),
                self._performance("L2", "201501", upb="100000", delinquency="00"),
                self._performance("L2", "201502", upb="0", zero_balance_code="16"),
                self._performance("L2", "201503", upb="0", delinquency="00"),
                self._performance("L2", "201504", upb="0", delinquency="01"),
            ]
            self._archive(archive, [self._origin("L1"), self._origin("L2")], rows)

            prepare_quarter(archive, panel, sample_size=10, seed=7, compression_level=3)
            monthly = pd.concat(read_csv_zst(panel, chunksize=10), ignore_index=True)

            for terminal in ("prepay", "other_exit"):
                loan = next(
                    group.reset_index(drop=True)
                    for _, group in monthly.groupby("loan_key")
                    if terminal in set(group["current_state"])
                )
                self.assertEqual(terminal, loan.iloc[0]["next_state"])
                self.assertFalse(loan.iloc[2]["consecutive_month"])
                self.assertTrue(pd.isna(loan.iloc[2]["next_state"]))

    def test_code_01_at_contractual_maturity_is_not_prepay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, panel = root / "historical_data_2015Q1.zip", root / "monthly.csv.zst"
            rows = [
                self._performance("L1", "201501"),
                self._performance("L1", "201502", upb="0", zero_balance_code="01"),
            ]
            self._archive(archive, [self._origin("L1", maturity="201502")], rows)

            prepare_quarter(archive, panel, sample_size=10, seed=7, compression_level=3)
            monthly = pd.concat(read_csv_zst(panel, chunksize=10), ignore_index=True)

            self.assertEqual(["current", "other_exit"], monthly["current_state"].tolist())
            self.assertNotIn("prepay", set(monthly["current_state"]))

    def test_other_exit_before_horizon_is_censored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, panel = root / "historical_data_2015Q1.zip", root / "monthly.csv.zst"
            rows = [
                self._performance("L1", "201501"),
                self._performance("L1", "201502", upb="0", zero_balance_code="16"),
                self._performance("L1", "201503", upb="0", zero_balance_code="09"),
                self._performance("L2", "202603"),
            ]
            self._archive(archive, [self._origin("L1"), self._origin("L2")], rows)

            loans, metadata = prepare_quarter(
                archive, panel, sample_size=10, seed=7, compression_level=3
            )

            self.assertEqual(1, len(loans))
            self.assertEqual(1, metadata["censored"])

    def test_prepay_before_later_default_correction_remains_nondefault(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, panel = root / "historical_data_2015Q1.zip", root / "monthly.csv.zst"
            rows = [
                self._performance("L1", "201501"),
                self._performance("L1", "201502", upb="0", zero_balance_code="01"),
                self._performance("L1", "201503", upb="0", zero_balance_code="09"),
            ]
            self._archive(archive, [self._origin("L1")], rows)

            loans, _ = prepare_quarter(
                archive, panel, sample_size=10, seed=7, compression_level=3
            )

            self.assertEqual(1, len(loans))
            self.assertEqual(0, loans.iloc[0]["default_24m"])

    def test_trailing_defect_invalidates_lgd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, panel = root / "historical_data_2015Q1.zip", root / "monthly.csv.zst"
            rows = [
                self._performance("L1", "201503", upb="90000", delinquency="03"),
                self._performance(
                    "L1",
                    "201504",
                    upb="0",
                    zero_balance_code="09",
                    actual_loss="35000",
                    components=("-10000", "-40000", "-5000", "5000", "80000", "5000"),
                ),
                self._performance("L1", "201505", upb="0", zero_balance_code="96", defect_settlement="201505"),
                self._performance("L2", "202603"),
            ]
            self._archive(archive, [self._origin("L1"), self._origin("L2")], rows)

            loans, _ = prepare_quarter(
                archive, panel, sample_size=10, seed=7, compression_level=3
            )

            defaulted = loans.loc[loans["default_24m"].eq(1)].iloc[0]
            self.assertFalse(defaulted["lgd_eligible"])
            self.assertTrue(pd.isna(defaulted["lgd"]))

    def test_unknown_net_sale_proceeds_makes_lgd_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, panel = root / "historical_data_2015Q1.zip", root / "monthly.csv.zst"
            rows = [
                self._performance("L1", "201503", upb="90000", delinquency="03"),
                self._performance(
                    "L1",
                    "201504",
                    upb="0",
                    zero_balance_code="09",
                    actual_loss="35000",
                    components=("-10000", "U", "-5000", "5000", "80000", "5000"),
                ),
                self._performance("L2", "202603"),
            ]
            self._archive(archive, [self._origin("L1"), self._origin("L2")], rows)

            loans, _ = prepare_quarter(archive, panel, sample_size=10, seed=7, compression_level=3)

            defaulted = loans.loc[loans["default_24m"].eq(1)].iloc[0]
            self.assertFalse(defaulted["lgd_eligible"])
            self.assertTrue(pd.isna(defaulted["lgd"]))

    def test_ead_falls_back_to_later_removal_upb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, panel = root / "historical_data_2015Q1.zip", root / "monthly.csv.zst"
            rows = [
                self._performance("L1", "201503", upb="0", delinquency="03"),
                self._performance(
                    "L1",
                    "201504",
                    upb="0",
                    zero_balance_code="09",
                    actual_loss="35000",
                    components=("-10000", "-40000", "-5000", "5000", "80000", "5000"),
                ),
                self._performance("L2", "202603"),
            ]
            self._archive(archive, [self._origin("L1"), self._origin("L2")], rows)

            loans, _ = prepare_quarter(archive, panel, sample_size=10, seed=7, compression_level=3)

            defaulted = loans.loc[loans["default_24m"].eq(1)].iloc[0]
            self.assertEqual(0.8, defaulted["ead_ratio"])
            self.assertAlmostEqual(35000 / 80000, defaulted["lgd"])

    def test_prepares_dataset_outputs_and_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "historical_data_2015Q1.zip"
            rows = [
                self._performance("L1", "201503", upb="90000", delinquency="03"),
                self._performance("L2", "202603", upb="50000"),
            ]
            self._archive(archive, [self._origin("L1"), self._origin("L2")], rows)
            output = root / "prepared" / "freddie-analysis.csv.zst"
            panels = root / "prepared" / "freddie-monthly"
            manifest_path = root / "prepared" / "manifest.json"

            manifest = prepare_dataset(
                root,
                output,
                panels,
                manifest_path,
                years=[2015],
                quarters=[1],
                sample_size=10,
                seed=7,
                compression_level=3,
            )

            analysis = pd.concat(read_csv_zst(output, chunksize=10), ignore_index=True)
            monthly = pd.concat(read_csv_zst(panels / "2015Q1.csv.zst", chunksize=10), ignore_index=True)
            self.assertEqual(2, manifest["eligible_rows"])
            self.assertEqual(set(analysis["loan_key"]), set(monthly["loan_key"]))
            self.assertTrue(manifest_path.is_file())


if __name__ == "__main__":
    unittest.main()
