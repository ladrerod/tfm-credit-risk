from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import zstandard as zstd

from src.data_access import load_manifest, read_csv_zst, verify_file
from src.integrity import file_sha256


class DataAccessTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, dict[str, object]]:
        frame = pd.DataFrame(
            {
                "cohort": [202001, 202001, 202002],
                "credit_score": [720, 680, 760],
                "default_24m": [0, 1, 0],
            }
        )
        path = root / "loans-2020.csv.zst"
        compressed = zstd.ZstdCompressor(level=3).compress(frame.to_csv(index=False).encode())
        path.write_bytes(compressed)
        item = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "rows": len(frame),
            "sha256": file_sha256(path),
            "columns": list(frame.columns),
        }
        return path, item

    def test_streams_verified_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, item = self._fixture(Path(directory))
            verify_file(path, item)
            chunks = list(read_csv_zst(path, columns=item["columns"], chunksize=2))
            self.assertEqual([2, 1], [len(frame) for frame in chunks])
            self.assertEqual(1, int(chunks[0]["default_24m"].sum()))

    def test_rejects_changed_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, item = self._fixture(Path(directory))
            item["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_file(path, item)

    def test_rejects_private_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, item = self._fixture(Path(directory))
            item["columns"] = [*item["columns"], "loan_identifier"]
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"version": 1, "files": [item]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "private columns"):
                load_manifest(manifest)

    def test_rejects_schema_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, item = self._fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                list(read_csv_zst(path, columns=["wrong"], chunksize=2))


if __name__ == "__main__":
    unittest.main()
