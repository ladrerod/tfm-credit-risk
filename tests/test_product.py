from __future__ import annotations

import ast
import copy
import hashlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import zstandard as zstd

from src.product import (
    AUTHORIZED_DATA_SHA256,
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

    def _train_product(self, source: Path, **kwargs: object) -> dict[str, object]:
        with patch("src.product.AUTHORIZED_DATA_SHA256", self._sha256(source)):
            return train_product(source, **kwargs)

    def _load_bundle(self, path: Path, source: Path) -> dict[str, object]:
        with patch("src.product.AUTHORIZED_DATA_SHA256", self._sha256(source)):
            return load_bundle(path)

    def test_rejects_an_unexpected_data_hash_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))
            with patch("src.product.read_csv_zst", side_effect=AssertionError("reader called")):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    train_product(source)

    def test_implementation_hash_changes_with_every_prediction_dependency(self) -> None:
        def digest(changed_name: str | None) -> str:
            with patch(
                "src.product._source_sha256",
                side_effect=lambda path: "f" * 64 if path.name == changed_name else "0" * 64,
            ):
                return _implementation_sha256()

        baseline = digest(None)
        for name in ("product.py", "pd_model.py", "metrics.py", "data_access.py", "integrity.py", "api.py"):
            with self.subTest(name=name):
                self.assertNotEqual(baseline, digest(name))

    def test_implementation_hash_canonicalizes_source_line_endings(self) -> None:
        names = ("product.py", "pd_model.py", "metrics.py", "data_access.py", "integrity.py", "api.py")
        source = "value = 1\nresult = value\n"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in names:
                (root / name).write_bytes(source.encode())

            def digest(contents: bytes) -> str:
                (root / "product.py").write_bytes(contents)
                with patch("src.product.__file__", str(root / "product.py")):
                    return _implementation_sha256()

            expected = digest(source.encode())
            self.assertEqual(expected, digest(source.replace("\n", "\r\n").encode()))
            self.assertEqual(expected, digest(source.replace("\n", "\r").encode()))
            self.assertNotEqual(expected, digest(b"value = 2\nresult = value\n"))

    def test_training_rejects_an_implementation_changed_after_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))
            with (
                patch("src.product.AUTHORIZED_DATA_SHA256", self._sha256(source)),
                patch("src.product._implementation_sha256", return_value="f" * 64),
            ):
                with self.assertRaisesRegex(RuntimeError, "implementation changed"):
                    train_product(source)

    def test_training_rejects_an_implementation_change_during_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))
            snapshot = _implementation_sha256()
            with (
                patch("src.product.AUTHORIZED_DATA_SHA256", self._sha256(source)),
                patch("src.product._implementation_sha256", side_effect=[snapshot, "f" * 64]),
            ):
                with self.assertRaisesRegex(RuntimeError, "implementation changed"):
                    train_product(source)

    def test_loader_rejects_boolean_or_float_bundle_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._prepared_file(root)
            bundle = self._train_product(source, chunksize=48)
            output = root / "pd-model.joblib"
            for version in (True, 1.0):
                invalid = dict(bundle, bundle_version=version)
                joblib.dump(invalid, output)
                with self.assertRaisesRegex(ValueError, "version"):
                    self._load_bundle(output, source)

    def test_cli_writes_a_bundle_and_prints_only_aggregate_identity(self) -> None:
        from scripts import train_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._prepared_file(root)
            output = root / "models" / "pd-model.joblib"
            stdout = io.StringIO()
            with (
                patch("src.product.AUTHORIZED_DATA_SHA256", self._sha256(source)),
                patch.object(sys, "argv", ["train_model", "--data", str(source), "--output", str(output)]),
                redirect_stdout(stdout),
            ):
                train_model.main()

            self.assertEqual(
                {"model_version", "selected_model_name", "data_sha256", "validation_metrics"},
                set(ast.literal_eval(stdout.getvalue())),
            )
            self.assertFalse(self._load_bundle(output, source)["test_evaluated"])

    def test_cli_rejects_an_expected_hash_override(self) -> None:
        from scripts import train_model

        with patch.object(
            sys,
            "argv",
            ["train_model", "--data", "input.zst", "--output", "model.joblib", "--expected-sha256", "0" * 64],
        ):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    train_model.main()

    def test_trains_only_through_2020_and_builds_the_five_field_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))

            bundle = self._train_product(source)

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
            bundle = self._train_product(source, chunksize=48)
            output = root / "models" / "pd-model.joblib"
            save_bundle(bundle, output)

            loaded = self._load_bundle(output, source)
            self.assertEqual(bundle["data_sha256"], loaded["data_sha256"])
            loaded["features"] = list(reversed(PRODUCT_FEATURES))
            joblib.dump(loaded, output)

            with self.assertRaisesRegex(ValueError, "feature order"):
                self._load_bundle(output, source)

    def test_save_rejects_an_implementation_change_after_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._prepared_file(root)
            bundle = self._train_product(source, chunksize=48)
            output = root / "pd-model.joblib"

            with patch("src.product._implementation_sha256", return_value="f" * 64):
                with self.assertRaisesRegex(RuntimeError, "implementation changed"):
                    save_bundle(bundle, output)

            self.assertFalse(output.exists())

    def test_loader_rejects_missing_keys_wrong_version_and_test_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._prepared_file(root)
            bundle = self._train_product(source, chunksize=48)
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
                    self._load_bundle(output, source)

    def test_loader_rejects_incoherent_consumed_bundle_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._prepared_file(root)
            bundle = self._train_product(source, chunksize=48)
            output = root / "pd-model.joblib"
            bad_values = (
                (lambda value: value.__setitem__("model_version", "other"), "model_version"),
                (lambda value: value.__setitem__("model", object()), "predict_proba"),
                (lambda value: value.__setitem__("selected_model_name", 1), "selected_model_name"),
                (lambda value: value.__setitem__("input_schema", {}), "input_schema"),
                (lambda value: value["input_schema"]["origination_fico"].__setitem__("type", "integer"), "input_schema"),
                (lambda value: value["input_schema"]["original_dti"].__setitem__("minimum", float("nan")), "input_schema"),
                (lambda value: value["input_schema"]["original_dti"].__setitem__("minimum", 100), "input_schema"),
                (lambda value: value["input_schema"]["original_cltv"].__setitem__("maximum", 201), "original_cltv"),
                (lambda value: value.__setitem__("validation_threshold", float("nan")), "validation_threshold"),
                (lambda value: value.__setitem__("validation_threshold", -0.01), "validation_threshold"),
                (lambda value: value.__setitem__("validation_metrics", []), "validation_metrics"),
                (lambda value: value.__setitem__("target", "other"), "target"),
                (lambda value: value.__setitem__("horizon_months", True), "horizon_months"),
                (lambda value: value.__setitem__("event_definition", "other"), "event_definition"),
                (lambda value: value.__setitem__("development_years", [2015]), "development_years"),
                (lambda value: value.__setitem__("calibration_year", 2018), "calibration_year"),
                (lambda value: value.__setitem__("validation_year", 2021), "validation_year"),
                (lambda value: value.__setitem__("data_source", "other"), "data_source"),
                (lambda value: value.__setitem__("data_sha256", "g" * 64), "data_sha256"),
                (lambda value: value.__setitem__("data_sha256", AUTHORIZED_DATA_SHA256), "data_sha256"),
                (lambda value: value.__setitem__("implementation_sha256", "0" * 64), "implementation_sha256"),
            )
            for change, message in bad_values:
                with self.subTest(message=message):
                    invalid = copy.deepcopy(bundle)
                    change(invalid)
                    joblib.dump(invalid, output)
                    with self.assertRaisesRegex(ValueError, message):
                        self._load_bundle(output, source)

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
