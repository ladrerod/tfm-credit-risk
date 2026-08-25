from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import zstandard as zstd

from src.data_access import load_compact
from src.product import (
    BUNDLE_VERSION,
    FIT_LABEL_CUTOFF,
    PRODUCT_FEATURES,
    REFERENCE_CALIBRATION_YEARS,
    REFERENCE_DEVELOPMENT_YEARS,
    _implementation_sha256,
    evaluate_product,
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
        for year in range(2010, 2024):
            for index in range(24):
                poison = 2018 <= year <= 2022
                rows.append(
                    {
                        "cohort_year": year,
                        "cohort_quarter": index % 4 + 1,
                        "origination_fico": -999_999 if poison else 620 + index * 5,
                        "original_dti": -999_999 if poison else 15 + index % 25,
                        "original_cltv": -999_999 if poison else 60 + index % 70,
                        "original_interest_rate": -999_999 if poison else 2.5 + index % 8 / 10,
                        "number_of_borrowers": -999_999 if poison else 1 + index % 2,
                        "pd_label_available_date": "2020-03-01" if year == 2015 and index == 0 else "2019-02-28",
                        "default_24m": index % 2,
                    }
                )
        target = directory / "freddie-pd24.csv.zst"
        target.write_bytes(zstd.ZstdCompressor(level=3).compress(pd.DataFrame(rows).to_csv(index=False).encode()))
        return target

    def _train(self, source: Path, **kwargs: object) -> dict[str, object]:
        with patch("src.product.EXPECTED_DATA_SHA256", self._sha256(source)):
            return train_product(source, **kwargs)

    def _load(self, path: Path, source: Path) -> dict[str, object]:
        with patch("src.product.EXPECTED_DATA_SHA256", self._sha256(source)):
            return load_bundle(path)

    def test_train_product_uses_2010_2015_development_and_2016_2017_calibration_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))
            captured: dict[str, set[int]] = {}
            from src.product import fit_calibrated_model as real_fit

            def fit(development: pd.DataFrame, calibration: pd.DataFrame, **kwargs: object) -> object:
                captured["development"] = set(development["cohort_year"])
                captured["calibration"] = set(calibration["cohort_year"])
                return real_fit(development, calibration, **kwargs)

            with patch("src.product.fit_calibrated_model", side_effect=fit):
                self._train(source)

        self.assertEqual(set(REFERENCE_DEVELOPMENT_YEARS), captured["development"])
        self.assertEqual(set(REFERENCE_CALIBRATION_YEARS), captured["calibration"])

    def test_all_fit_rows_have_labels_available_strictly_before_2020_03_01(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))
            captured: list[pd.Timestamp] = []
            from src.product import fit_calibrated_model as real_fit

            def fit(development: pd.DataFrame, calibration: pd.DataFrame, **kwargs: object) -> object:
                captured.extend(pd.concat([development, calibration])["pd_label_available_date"])
                return real_fit(development, calibration, **kwargs)

            with patch("src.product.fit_calibrated_model", side_effect=fit):
                self._train(source)

        self.assertTrue(captured)
        self.assertTrue(all(value < FIT_LABEL_CUTOFF for value in captured))

    def test_train_product_never_reads_2018_2023_as_fit_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))
            with patch("src.product.load_compact", wraps=load_compact) as loader:
                self._train(source)

        self.assertEqual(
            REFERENCE_DEVELOPMENT_YEARS + REFERENCE_CALIBRATION_YEARS,
            loader.call_args.kwargs["years"],
        )

    def test_bundle_freezes_calibration_p50_p90_data_and_implementation_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))
            bundle = self._train(source)
            calibration, digest = load_compact(source, self._sha256(source), years=REFERENCE_CALIBRATION_YEARS)
            calibration = calibration.loc[calibration["pd_label_available_date"] < FIT_LABEL_CUTOFF]
            probability = bundle["model"].predict_proba(calibration[list(PRODUCT_FEATURES)])[:, 1]

        self.assertEqual(BUNDLE_VERSION, bundle["bundle_version"])
        self.assertEqual(list(PRODUCT_FEATURES), bundle["features"])
        self.assertEqual([float(value) for value in np.quantile(probability, [0.5, 0.9])], bundle["risk_band_cutoffs"])
        self.assertEqual(digest, bundle["data_sha256"])
        self.assertEqual(_implementation_sha256(), bundle["implementation_sha256"])

    def test_load_bundle_rejects_missing_stale_or_incompatible_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._prepared_file(root)
            bundle = self._train(source)
            output = root / "pd24-model.joblib"
            for change, message in (
                (lambda item: item.pop("model"), "missing keys"),
                (lambda item: item.__setitem__("bundle_version", BUNDLE_VERSION + 1), "version"),
                (lambda item: item.__setitem__("features", list(reversed(PRODUCT_FEATURES))), "feature order"),
                (lambda item: item.__setitem__("risk_band_cutoffs", [0.9, 0.1]), "risk_band_cutoffs"),
                (lambda item: item.__setitem__("calibration_metrics", {"n": "bad"}), "calibration_metrics"),
                (lambda item: item.__setitem__("data_sha256", "0" * 64), "data_sha256"),
                (lambda item: item.__setitem__("implementation_sha256", "0" * 64), "implementation_sha256"),
            ):
                invalid = copy.deepcopy(bundle)
                change(invalid)
                joblib.dump(invalid, output)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    self._load(output, source)

    def test_evaluate_product_rejects_mixing_2023_with_other_years(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._prepared_file(root)
            output = root / "pd24-model.joblib"
            save_bundle(self._train(source), output)
            with patch("src.product.EXPECTED_DATA_SHA256", self._sha256(source)):
                result = evaluate_product(source, output, (2023,))
                with self.assertRaisesRegex(ValueError, "2023"):
                    evaluate_product(source, output, (2022, 2023))

        self.assertEqual([2023], result["years"])
        self.assertEqual(2023, result["annual"]["cohort_year"])
        self.assertEqual([1, 2, 3, 4], [row["cohort_quarter"] for row in result["quarters"]])

    def test_synthetic_training_is_deterministic_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))
            first = self._train(source)
            second = self._train(source)
            probe = pd.DataFrame([{feature: 1 for feature in PRODUCT_FEATURES}])

        self.assertEqual(
            {key: value for key, value in first.items() if key != "model"},
            {key: value for key, value in second.items() if key != "model"},
        )
        np.testing.assert_allclose(first["model"].predict_proba(probe), second["model"].predict_proba(probe))


if __name__ == "__main__":
    unittest.main()
