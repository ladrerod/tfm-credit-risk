from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import zstandard as zstd

from src.product import (
    BUNDLE_VERSION,
    PRODUCT_FEATURES,
    _implementation_sha256,
    load_bundle,
    save_bundle,
    train_product,
)


class ProductTests(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _prepared_file(self, directory: Path) -> Path:
        rows: list[dict[str, float | int | str]] = []
        for year in range(2015, 2021):
            for index in range(48):
                rows.append(
                    {
                        "cohort_year": year,
                        "origination_fico": 620 + index * 4,
                        "original_dti": 15 + index % 25,
                        "original_cltv": 250 if index == 0 else 60 + index % 70,
                        "original_interest_rate": 2.5 + index % 8 / 10,
                        "number_of_borrowers": 1 + index % 2,
                        "default_24m": index % 2,
                    }
                )
        for year in (2021, 2022):
            poison = -999_999 if year == 2021 else 999_999
            rows.extend(
                {
                    "cohort_year": year,
                    "origination_fico": poison,
                    "original_dti": poison,
                    "original_cltv": poison,
                    "original_interest_rate": poison,
                    "number_of_borrowers": poison,
                    "default_24m": 2,
                }
                for _ in range(48)
            )
        target = directory / "freddie-analysis.csv.zst"
        compressed = zstd.ZstdCompressor(level=3).compress(pd.DataFrame(rows).to_csv(index=False).encode())
        target.write_bytes(compressed)
        return target

    def test_rejects_an_unexpected_data_hash_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))
            with patch("src.product.read_csv_zst", side_effect=AssertionError("reader called")):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    train_product(source, expected_sha256="0" * 64)

    def test_implementation_hash_changes_when_metrics_change(self) -> None:
        def digest(metrics_hash: str) -> str:
            with patch(
                "src.product.file_sha256",
                side_effect=lambda path: metrics_hash if path.name == "metrics.py" else "0" * 64,
            ):
                return _implementation_sha256()

        self.assertNotEqual(digest("0" * 64), digest("f" * 64))

    def test_loader_rejects_boolean_or_float_bundle_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._prepared_file(root)
            bundle = train_product(source, chunksize=48, expected_sha256=self._sha256(source))
            output = root / "pd-model.joblib"
            for version in (True, 1.0):
                invalid = dict(bundle, bundle_version=version)
                joblib.dump(invalid, output)
                with self.assertRaisesRegex(ValueError, "version"):
                    load_bundle(output)

    def test_cli_writes_a_bundle_and_prints_only_aggregate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._prepared_file(root)
            output = root / "models" / "pd-model.joblib"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.train_model",
                    "--data",
                    str(source),
                    "--output",
                    str(output),
                    "--expected-sha256",
                    self._sha256(source),
                ],
                cwd=Path(__file__).parents[1],
                capture_output=True,
                text=True,
            )

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(
                {"model_version", "selected_model_name", "data_sha256", "validation_metrics"},
                set(ast.literal_eval(result.stdout)),
            )
            self.assertFalse(load_bundle(output)["test_evaluated"])

    def test_trains_only_through_2020_and_builds_the_five_field_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))

            bundle = train_product(source, expected_sha256=self._sha256(source))

        self.assertEqual(BUNDLE_VERSION, bundle["bundle_version"])
        self.assertEqual(list(PRODUCT_FEATURES), bundle["features"])
        self.assertEqual(
            {
                "origination_fico": (620.0, 808.0),
                "original_dti": (15.0, 39.0),
                "original_cltv": (61.0, 200.0),
                "original_interest_rate": (2.5, 3.2),
                "number_of_borrowers": (1.0, 2.0),
            },
            {
                name: (schema["minimum"], schema["maximum"])
                for name, schema in bundle["input_schema"].items()
            },
        )
        self.assertFalse(bundle["test_evaluated"])
        self.assertEqual({"logistic", "hist_gradient_boosting"}, set(bundle["validation_metrics"]))
        self.assertTrue(
            np.isfinite(bundle["model"].predict_proba(pd.DataFrame([{feature: 1 for feature in PRODUCT_FEATURES}]))).all()
        )

    def test_joblib_round_trip_rejects_a_changed_feature_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._prepared_file(root)
            bundle = train_product(source, chunksize=48, expected_sha256=self._sha256(source))
            output = root / "models" / "pd-model.joblib"
            save_bundle(bundle, output)

            loaded = load_bundle(output)
            self.assertEqual(bundle["data_sha256"], loaded["data_sha256"])
            loaded["features"] = list(reversed(PRODUCT_FEATURES))
            joblib.dump(loaded, output)

            with self.assertRaisesRegex(ValueError, "feature order"):
                load_bundle(output)

    def test_loader_rejects_missing_keys_wrong_version_and_test_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._prepared_file(root)
            bundle = train_product(source, chunksize=48, expected_sha256=self._sha256(source))
            output = root / "pd-model.joblib"
            for change, message in (
                (lambda value: value.pop("model"), "missing keys"),
                (lambda value: value.__setitem__("bundle_version", BUNDLE_VERSION + 1), "version"),
                (lambda value: value.__setitem__("test_evaluated", True), "test_evaluated"),
            ):
                invalid = dict(bundle)
                change(invalid)
                joblib.dump(invalid, output)
                with self.assertRaisesRegex(ValueError, message):
                    load_bundle(output)

    def test_atomic_save_removes_a_partial_file_when_joblib_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "models" / "pd-model.joblib"

            def fail_dump(_: object, temporary: Path) -> None:
                temporary.write_bytes(b"partial")
                raise OSError("disk full")

            with patch("src.product.joblib.dump", side_effect=fail_dump):
                with self.assertRaisesRegex(OSError, "disk full"):
                    save_bundle({"bundle_version": BUNDLE_VERSION}, output)

            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".joblib.partial").exists())


if __name__ == "__main__":
    unittest.main()
