from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import zstandard as zstd

from src.data_access import read_csv_zst, write_csv_zst_parts


class DataAccessTests(unittest.TestCase):
    def _fixture(self, root: Path, *, private: bool = False) -> Path:
        frame = pd.DataFrame(
            {
                "cohort": [202001, 202001, 202002],
                "credit_score": [720, 680, 760],
                "default_24m": [0, 1, 0],
            }
        )
        if private:
            frame[" Loan-Identifier "] = ["A", "B", "C"]
        path = root / "loans-2020.csv.zst"
        compressed = zstd.ZstdCompressor(level=3).compress(frame.to_csv(index=False).encode())
        path.write_bytes(compressed)
        return path

    def test_streams_standalone_prepared_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._fixture(Path(directory))
            chunks = list(read_csv_zst(path, chunksize=2))
            self.assertEqual([2, 1], [len(frame) for frame in chunks])
            self.assertEqual(1, int(chunks[0]["default_24m"].sum()))

    def test_rejects_private_columns_in_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._fixture(Path(directory), private=True)
            with self.assertRaisesRegex(ValueError, "private columns"):
                list(read_csv_zst(path, chunksize=2))

    def test_rejects_freddie_sequence_number_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.csv.zst"
            frame = pd.DataFrame({"Loan Sequence Number": ["F123"]})
            path.write_bytes(zstd.ZstdCompressor(level=3).compress(frame.to_csv(index=False).encode()))

            with self.assertRaisesRegex(ValueError, "private columns"):
                list(read_csv_zst(path, chunksize=1))

    def test_writer_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parts.csv.zst"
            write_csv_zst_parts([pd.DataFrame({"value": [1, 2]})], path, level=3)
            self.assertEqual([1, 2], pd.concat(read_csv_zst(path, chunksize=1))["value"].tolist())


if __name__ == "__main__":
    unittest.main()
