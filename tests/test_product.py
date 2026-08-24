from __future__ import annotations

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
    load_bundle,
    save_bundle,
    train_product,
)


class ProductTests(unittest.TestCase):
    def _prepared_file(self, directory: Path) -> Path:
        rows: list[dict[str, float | int]] = []
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
            rows.extend(
                {
                    "cohort_year": year,
                    "origination_fico": "poisoned",
                    "original_dti": "poisoned",
                    "original_cltv": "poisoned",
                    "original_interest_rate": "poisoned",
                    "number_of_borrowers": "poisoned",
                    "default_24m": 2,
                }
                for _ in range(48)
            )
        target = directory / "freddie-analysis.csv.zst"
        compressed = zstd.ZstdCompressor(level=3).compress(pd.DataFrame(rows).to_csv(index=False).encode())
        target.write_bytes(compressed)
        return target

    def test_trains_only_through_2020_and_builds_the_five_field_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))

            bundle = train_product(source)

        self.assertEqual(BUNDLE_VERSION, bundle["bundle_version"])
        self.assertEqual(list(PRODUCT_FEATURES), bundle["features"])
        self.assertEqual(200.0, bundle["input_schema"]["original_cltv"]["maximum"])
        self.assertFalse(bundle["test_evaluated"])
        self.assertEqual({"logistic", "hist_gradient_boosting"}, set(bundle["validation_metrics"]))
        self.assertTrue(
            np.isfinite(bundle["model"].predict_proba(pd.DataFrame([{feature: 1 for feature in PRODUCT_FEATURES}]))).all()
        )

    def test_joblib_round_trip_rejects_a_changed_feature_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = train_product(self._prepared_file(root), chunksize=48)
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
            bundle = train_product(self._prepared_file(root), chunksize=48)
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
