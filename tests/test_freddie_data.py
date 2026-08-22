from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from src.data_access import load_manifest, read_csv_zst
from src.freddie_data import _selected_performance_chunks, prepare_dataset, prepare_quarter


class FreddieDataTests(unittest.TestCase):
    @staticmethod
    def _origin(
        loan: str,
        *,
        first_payment: str = "201501",
        upb: str = "100000",
        high_balance: str = "N",
    ) -> str:
        row = [""] * 31
        values = {
            0: "720",
            1: first_payment,
            2: "N",
            3: "204412",
            5: "0",
            6: "1",
            7: "P",
            8: "80",
            9: "35",
            10: upb,
            11: "80",
            12: "4.0",
            13: "R",
            14: "N",
            15: "FRM",
            16: "CA",
            17: "SF",
            19: loan,
            20: "P",
            21: "360",
            22: "1",
            24: high_balance,
            28: "N",
            29: "2",
            30: "N",
        }
        for position, value in values.items():
            row[position] = value
        return "|".join(row)

    @staticmethod
    def _performance(
        loan: str,
        period: str,
        *,
        upb: str = "100000",
        delinquency: str = "00",
        zero_balance_code: str = "",
        zero_balance_date: str = "",
        components: tuple[str, str, str, str, str, str, str] | None = None,
    ) -> str:
        row = [""] * 35
        row[0:6] = [loan, period, upb, delinquency, "1", "359"]
        row[10] = "4.0"
        if zero_balance_code:
            row[8] = zero_balance_code
            row[9] = zero_balance_date or period
        if components:
            mi, sale, non_mi, expenses, actual_loss, removal_upb, interest = components
            row[13:17] = [mi, sale, non_mi, expenses]
            row[21] = actual_loss
            row[26] = removal_upb
            row[27] = interest
        return "|".join(row)

    def _write_zip(self, path: Path, origin: list[str], performance: list[str]) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("orig_2015Q1.txt", "\n".join(origin) + "\n")
            archive.writestr("perf_2015Q1.txt", "\n".join(performance) + "\n")

    def test_filters_performance_rows_before_column_conversion(self) -> None:
        rows = (
            self._performance("SKIP", "202001") + "\n" + self._performance("KEEP", "202002") + "\n"
        ).encode()

        chunks = list(_selected_performance_chunks(io.BytesIO(rows), {"KEEP"}, chunksize=1))

        self.assertEqual(["KEEP"], chunks[0]["loan_identifier"].tolist())
        self.assertEqual(["202002"], chunks[0]["period"].tolist())

    def test_rejects_wrong_field_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical_data_2015Q1.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("orig_2015Q1.txt", "|".join(["x"] * 30) + "\n")
                archive.writestr("perf_2015Q1.txt", "|".join([""] * 35) + "\n")

            with self.assertRaisesRegex(ValueError, "31 fields"):
                prepare_quarter(path, sample_size=1, seed=1)

    def test_builds_default_ead_and_reconciled_lgd_without_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical_data_2015Q1.zip"
            origins = [self._origin("D1", high_balance="Y"), self._origin("A1")]
            performance = [
                self._performance("D1", "201501"),
                self._performance("D1", "201504", upb="90000", delinquency="03"),
                self._performance(
                    "D1",
                    "201801",
                    upb="0",
                    zero_balance_code="09",
                    components=("-10000", "-40000", "-5000", "5000", "35000", "80000", "5000"),
                ),
                self._performance("A1", "202603", upb="50000"),
            ]
            self._write_zip(path, origins, performance)

            frame, metadata = prepare_quarter(path, sample_size=10, seed=1)

            defaulted = frame.loc[frame["default_24m"].eq(1)].iloc[0]
            self.assertEqual(pd.Timestamp("2015-04-01"), defaulted["default_event_date"])
            self.assertAlmostEqual(0.9, defaulted["ead_ratio"])
            self.assertAlmostEqual(35000 / 90000, defaulted["lgd"])
            self.assertTrue(defaulted["lgd_eligible"])
            self.assertEqual("Y", defaulted["high_balance_loan"])
            self.assertNotIn("loan_identifier", frame)
            self.assertEqual(1, metadata["defaults"])

    def test_handles_prepayment_censoring_and_zero_balance_ead_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical_data_2015Q1.zip"
            origins = [self._origin("P1"), self._origin("C1"), self._origin("Z1")]
            performance = [
                self._performance("P1", "201501"),
                self._performance("P1", "201506", upb="0", zero_balance_code="01"),
                self._performance("C1", "201501"),
                self._performance("C1", "201506"),
                self._performance(
                    "Z1",
                    "201510",
                    upb="",
                    zero_balance_code="03",
                    components=("0", "-20000", "0", "2000", "52000", "70000", "0"),
                ),
            ]
            self._write_zip(path, origins, performance)

            frame, metadata = prepare_quarter(path, sample_size=10, seed=1)

            self.assertEqual(2, len(frame))
            self.assertEqual(0, int(frame.loc[frame["default_24m"].eq(0), "default_24m"].iloc[0]))
            fallback = frame.loc[frame["default_24m"].eq(1)].iloc[0]
            self.assertAlmostEqual(0.7, fallback["ead_ratio"])
            self.assertEqual(1, metadata["censored"])

    def test_excludes_incomplete_and_recent_losses_from_lgd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical_data_2015Q1.zip"
            origins = [self._origin("I1"), self._origin("R1"), self._origin("A1")]
            performance = [
                self._performance("I1", "201503", upb="90000", delinquency="03"),
                self._performance(
                    "I1",
                    "201801",
                    upb="0",
                    zero_balance_code="09",
                    components=("", "-40000", "0", "5000", "45000", "80000", "0"),
                ),
                self._performance("R1", "201503", upb="90000", delinquency="03"),
                self._performance(
                    "R1",
                    "202602",
                    upb="0",
                    zero_balance_code="09",
                    components=("0", "-40000", "0", "5000", "45000", "80000", "0"),
                ),
                self._performance("A1", "202603", upb="50000"),
            ]
            self._write_zip(path, origins, performance)

            frame, _ = prepare_quarter(path, sample_size=10, seed=1)

            defaults = frame.loc[frame["default_24m"].eq(1)]
            self.assertEqual(2, len(defaults))
            self.assertFalse(defaults["lgd_eligible"].any())

    def test_prepares_one_compact_file_and_reproducibility_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw" / "historical_data_2015"
            source.mkdir(parents=True)
            archive = source / "historical_data_2015Q1.zip"
            self._write_zip(
                archive,
                [self._origin("D1"), self._origin("A1")],
                [
                    self._performance("D1", "201504", upb="90000", delinquency="03"),
                    self._performance("A1", "202603", upb="50000"),
                ],
            )
            output = root / "freddie-analysis.csv.zst"
            manifest_path = root / "manifest.json"

            manifest = prepare_dataset(
                root / "raw",
                output,
                manifest_path,
                years=[2015],
                quarters=[1],
                sample_size=10,
                seed=7,
                compression_level=3,
            )

            validated = load_manifest(manifest_path)
            frames = list(
                read_csv_zst(output, columns=validated["files"][0]["columns"], chunksize=10)
            )
            self.assertEqual(2, len(pd.concat(frames)))
            self.assertEqual("Freddie Mac Single-Family Loan-Level Dataset", manifest["source"])
            self.assertEqual(1, len(manifest["source_files"]))
            self.assertNotIn("loan_identifier", manifest["files"][0]["columns"])


if __name__ == "__main__":
    unittest.main()
