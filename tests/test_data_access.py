from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import zstandard as zstd

from src.data_access import (
    COMPACT_COLUMNS,
    SOURCE_CUTOFF,
    file_sha256,
    load_compact,
    read_csv_zst,
)


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cohort_year": [2010, 2010, 2010, 2011],
            "cohort_quarter": [1, 2, 3, 4],
            "first_payment_date": [
                "2010-01-01",
                "2010-04-01",
                "2010-07-01",
                "2011-10-01",
            ],
            "original_interest_rate": [4.0, 4.1, 4.2, 4.3],
            "original_upb": [100_000, 110_000, 120_000, 130_000],
            "original_loan_term": [360] * 4,
            "original_ltv": [75, 76, 77, 78],
            "original_cltv": [80, 81, 82, 83],
            "number_of_borrowers": [1, 2, 1, 2],
            "original_dti": [30, 31, 32, 33],
            "origination_fico": [700, 710, 720, 730],
            "mortgage_insurance_percentage": [0, 10, 20, 30],
            "first_time_home_buyer": ["N", "Y", "N", "Y"],
            "loan_purpose": ["P", "C", "R", "P"],
            "property_type": ["SF"] * 4,
            "number_of_units": ["1"] * 4,
            "occupancy_status": ["P"] * 4,
            "property_state": ["VA", "CA", "TX", "NY"],
            "amortization_type": ["FRM"] * 4,
            "mortgage_insurance_type": ["none", "1", "2", "1"],
            "high_balance_loan": ["N"] * 4,
            "pd_label_available_date": [
                "2011-12-01",
                "2012-03-01",
                "2012-06-01",
                "2013-09-01",
            ],
            "default_24m": [0, 1, 0, 1],
        }
    )[list(COMPACT_COLUMNS)]


class DataAccessTests(unittest.TestCase):
    def write_compact(self, frame: pd.DataFrame) -> Path:
        path = Path(self.temporary.name) / "compact.csv.zst"
        path.write_bytes(
            zstd.ZstdCompressor(level=3).compress(frame.to_csv(index=False).encode())
        )
        return path

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_load_compact_requires_exact_hash_schema_binary_target_and_maturity(
        self,
    ) -> None:
        path = self.write_compact(valid_frame())
        digest = file_sha256(path)

        frame, observed = load_compact(path, digest, years=(2010, 2011), chunksize=3)

        self.assertEqual(list(COMPACT_COLUMNS), list(frame.columns))
        self.assertEqual(digest, observed)
        self.assertEqual(
            [3, 1], [len(chunk) for chunk in read_csv_zst(path, chunksize=3)]
        )
        self.assertEqual(SOURCE_CUTOFF, pd.Timestamp("2026-03-01"))

    def test_load_compact_rejects_incorrect_hash_before_reading(self) -> None:
        path = self.write_compact(valid_frame().assign(default_24m=[0, 2, 0, 1]))

        with self.assertRaisesRegex(ValueError, "sha256"):
            load_compact(path, "0" * 64, years=(2010,), chunksize=2)

    def test_read_csv_zst_rejects_extra_missing_or_private_columns(self) -> None:
        cases = {
            "extra": valid_frame().assign(unexpected=1),
            "missing": valid_frame().drop(columns="default_24m"),
            "private": valid_frame().assign(loan_identifier="secret"),
        }
        for name, frame in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "header"):
                list(read_csv_zst(self.write_compact(frame), chunksize=2))

    def test_load_compact_rejects_selected_nonbinary_target_or_immaturity(self) -> None:
        cases = {
            "target": valid_frame().assign(default_24m=[0, 2, 0, 1]),
            "immature": valid_frame().assign(
                pd_label_available_date=["2026-03-02"] * 4
            ),
        }
        for name, frame in cases.items():
            with self.subTest(name=name):
                path = self.write_compact(frame)
                with self.assertRaises(ValueError):
                    load_compact(
                        path, file_sha256(path), years=(2010, 2011), chunksize=2
                    )

    def test_load_compact_rejects_invalid_cohort_quarter_cap_and_ranges(self) -> None:
        oversized = pd.concat([valid_frame().iloc[[0]]] * 12_501, ignore_index=True)
        cases = {
            "year": valid_frame().assign(cohort_year=[2003, 2010, 2010, 2011]),
            "quarter": valid_frame().assign(cohort_quarter=[5, 2, 3, 4]),
            "cap": oversized,
            "range": valid_frame().assign(original_cltv=[1_001, 81, 82, 83]),
        }
        for name, frame in cases.items():
            with self.subTest(name=name):
                path = self.write_compact(frame)
                with self.assertRaises(ValueError):
                    load_compact(
                        path, file_sha256(path), years=(2010, 2011), chunksize=2_000
                    )

    def test_load_compact_does_not_convert_or_validate_unselected_2023_target(
        self,
    ) -> None:
        frame = pd.concat(
            [
                valid_frame(),
                valid_frame()
                .iloc[[0]]
                .assign(cohort_year=2023, default_24m="sentinel"),
            ],
            ignore_index=True,
        )
        path = self.write_compact(frame)

        loaded, _ = load_compact(
            path, file_sha256(path), years=(2010, 2011), chunksize=2
        )

        self.assertEqual({2010, 2011}, set(loaded["cohort_year"]))


if __name__ == "__main__":
    unittest.main()
