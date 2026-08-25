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

import src.product as product
from src.data_access import load_compact
from src.pd_model import SigmoidCalibratedModel
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

    def _prepared_file(self, directory: Path, *, missing_2023_quarter: int | None = None) -> Path:
        rows: list[dict[str, float | int | str]] = []
        for year in range(2010, 2024):
            for index in range(24):
                poison = 2018 <= year <= 2022
                quarter = index % 4 + 1
                if year == 2023 and quarter == missing_2023_quarter:
                    continue
                rows.append(
                    {
                        "cohort_year": year,
                        "cohort_quarter": quarter,
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
        self.assertEqual(
            (
                "src/api.py", "src/data_access.py", "src/integrity.py", "src/metrics.py",
                "src/pd_model.py", "src/product.py", "scripts/train_model.py", "scripts/serve_model.py",
            ),
            product._IMPLEMENTATION_FILES,
        )
        with patch("src.product.Path.read_bytes", return_value=b"line one\r\nline two\r\n") as read_bytes:
            crlf = _implementation_sha256()
        with patch("src.product.Path.read_bytes", return_value=b"line one\nline two\n"):
            lf = _implementation_sha256()
        self.assertEqual(crlf, lf)
        self.assertEqual(8, read_bytes.call_count)

    def test_load_bundle_rejects_missing_stale_or_incompatible_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._prepared_file(root)
            bundle = self._train(source)
            output = root / "pd24-model.joblib"
            self.assertIsInstance(bundle["model"], SigmoidCalibratedModel)
            for change, message in (
                (lambda item: item.pop("model"), "missing keys"),
                (lambda item: item.__setitem__("bundle_version", BUNDLE_VERSION + 1), "version"),
                (lambda item: item.__setitem__("model", item["model"].estimator), "model"),
                (lambda item: item.__setitem__("features", list(reversed(PRODUCT_FEATURES))), "feature order"),
                (lambda item: item.__setitem__("risk_band_cutoffs", [0.9, 0.1]), "risk_band_cutoffs"),
                (lambda item: item.__setitem__("calibration_metrics", {"n": "bad"}), "calibration_metrics"),
                (lambda item: item["calibration_metrics"].__setitem__("prevalence", -1.0), "calibration_metrics"),
                (lambda item: item["calibration_metrics"].__setitem__("roc_auc", 1.1), "calibration_metrics"),
                (lambda item: item["calibration_metrics"].__setitem__("brier", None), "calibration_metrics"),
                (lambda item: item["score_bins"].__setitem__(0, -0.1), "score PSI"),
                (lambda item: item["score_bins"].__setitem__(-1, 1.1), "score PSI"),
                (lambda item: item.__setitem__("score_distribution", [1.0]), "score PSI"),
                (lambda item: item.__setitem__("score_distribution", [0.0] * len(item["score_distribution"])), "score PSI"),
                (lambda item: item.__setitem__("score_distribution", [-0.1, 1.1] + [0.0] * (len(item["score_distribution"]) - 2)), "score PSI"),
                (lambda item: item.__setitem__("data_sha256", "0" * 64), "data_sha256"),
                (lambda item: item.__setitem__("implementation_sha256", "0" * 64), "implementation_sha256"),
            ):
                invalid = copy.deepcopy(bundle)
                change(invalid)
                joblib.dump(invalid, output)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    self._load(output, source)
            save_bundle(bundle, output)
            previous = output.read_bytes()

            def fail_dump(_: object, temporary: Path) -> None:
                temporary.write_bytes(b"partial")
                raise OSError("disk full")

            with patch("src.product.joblib.dump", side_effect=fail_dump):
                with self.assertRaisesRegex(OSError, "disk full"):
                    save_bundle(bundle, output)
            self.assertTrue(output.exists())
            self.assertEqual(previous, output.read_bytes())
            self.assertFalse(output.with_suffix(".joblib.partial").exists())
            self.assertEqual(bundle["data_sha256"], self._load(output, source)["data_sha256"])

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
                with self.assertRaisesRegex(ValueError, "years"):
                    evaluate_product(source, output, (2018, 2018))
                for invalid_years in ([2018], (2019, 2018), (2017,)):
                    with self.subTest(years=invalid_years), self.assertRaisesRegex(ValueError, "years"):
                        evaluate_product(source, output, invalid_years)  # type: ignore[arg-type]
                incomplete = self._prepared_file(root, missing_2023_quarter=4)
                with patch("src.product.EXPECTED_DATA_SHA256", self._sha256(incomplete)):
                    incomplete_bundle = train_product(incomplete)
                    incomplete_output = root / "incomplete-pd24-model.joblib"
                    save_bundle(incomplete_bundle, incomplete_output)
                    with self.assertRaisesRegex(ValueError, "quarters"):
                        evaluate_product(incomplete, incomplete_output, (2023,))

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
