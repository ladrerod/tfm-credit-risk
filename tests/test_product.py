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
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_access import COMPACT_COLUMNS, load_compact
from src.pd_model import NUMERIC_FEATURES
from src.product import (
    BUNDLE_VERSION,
    FIT_LABEL_CUTOFF,
    PRODUCT_FEATURES,
    REFERENCE_BAND_YEARS,
    REFERENCE_DEVELOPMENT_YEARS,
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
        for year in range(2013, 2024):
            for index in range(24):
                poison = year == 2020
                quarter = index % 4 + 1
                first_payment = pd.Timestamp(year, quarter * 3 - 2, 1)
                rows.append(
                    {
                        "cohort_year": year,
                        "cohort_quarter": quarter,
                        "first_payment_date": first_payment.date().isoformat(),
                        "original_interest_rate": -999_999
                        if poison
                        else 2.5 + index % 8 / 10,
                        "original_upb": (
                            np.nan
                            if year == 2013 and index == 0
                            else 100_000 + index * 1_000
                        ),
                        "original_loan_term": 360,
                        "original_ltv": 60 + index % 30,
                        "original_cltv": -999_999 if poison else 65 + index % 30,
                        "number_of_borrowers": -999_999 if poison else 1 + index % 2,
                        "original_dti": -999_999 if poison else 15 + index % 25,
                        "origination_fico": -999_999 if poison else 620 + index * 5,
                        "mortgage_insurance_percentage": index % 4 * 10,
                        "first_time_home_buyer": "Y" if index % 2 else "N",
                        "loan_purpose": ("P", "C", "R")[index % 3],
                        "property_type": "SF",
                        "number_of_units": "1",
                        "occupancy_status": "P",
                        "property_state": ("VA", "CA", "TX")[index % 3],
                        "amortization_type": "FRM",
                        "mortgage_insurance_type": "none" if index % 4 == 0 else "1",
                        "high_balance_loan": "N",
                        "pd_label_available_date": (
                            first_payment + pd.DateOffset(months=23)
                        )
                        .date()
                        .isoformat(),
                        "default_24m": index % 2,
                    }
                )
        target = directory / "freddie-pd24-wide.csv.zst"
        frame = pd.DataFrame(rows)[list(COMPACT_COLUMNS)]
        target.write_bytes(
            zstd.ZstdCompressor(level=3).compress(frame.to_csv(index=False).encode())
        )
        return target

    def _train(self, source: Path, **kwargs: object) -> dict[str, object]:
        with patch("src.product.EXPECTED_DATA_SHA256", self._sha256(source)):
            return train_product(source, **kwargs)

    def _load(self, path: Path, source: Path) -> dict[str, object]:
        with patch("src.product.EXPECTED_DATA_SHA256", self._sha256(source)):
            return load_bundle(path)

    def test_train_product_uses_selected_development_and_band_reference_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))
            captured: dict[str, object] = {}
            from src.product import fit_model as real_fit

            def fit(development: pd.DataFrame, **kwargs: object) -> object:
                captured["development"] = set(development["cohort_year"])
                captured["has_unlinked"] = development["original_upb"].isna().any()
                return real_fit(development, **kwargs)

            with patch("src.product.fit_model", side_effect=fit):
                bundle = self._train(source)

        self.assertEqual(set(REFERENCE_DEVELOPMENT_YEARS), captured["development"])
        self.assertFalse(captured["has_unlinked"])
        self.assertEqual(
            list(REFERENCE_BAND_YEARS),
            bundle["periods"]["band_reference_years"],
        )

    def test_all_fit_rows_have_labels_available_before_the_fit_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))
            captured: list[pd.Timestamp] = []
            from src.product import fit_model as real_fit

            def fit(development: pd.DataFrame, **kwargs: object) -> object:
                captured.extend(development["pd_label_available_date"])
                return real_fit(development, **kwargs)

            with patch("src.product.fit_model", side_effect=fit):
                self._train(source)

        self.assertTrue(captured)
        self.assertTrue(all(value < FIT_LABEL_CUTOFF for value in captured))

    def test_train_product_reads_only_refit_years(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))
            with patch("src.product.load_compact", wraps=load_compact) as loader:
                self._train(source)

        self.assertEqual(
            REFERENCE_DEVELOPMENT_YEARS + REFERENCE_BAND_YEARS,
            loader.call_args.kwargs["years"],
        )

    def test_bundle_freezes_reference_p50_p90_and_data_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))
            bundle = self._train(source)
            reference, digest = load_compact(
                source, self._sha256(source), years=REFERENCE_BAND_YEARS
            )
            reference = reference.loc[
                reference["pd_label_available_date"] < FIT_LABEL_CUTOFF
            ]
            probability = bundle["model"].predict_proba(
                reference[list(PRODUCT_FEATURES)]
            )[:, 1]

        self.assertEqual(BUNDLE_VERSION, bundle["bundle_version"])
        self.assertEqual(list(PRODUCT_FEATURES), bundle["features"])
        self.assertEqual(
            [float(value) for value in np.quantile(probability, [0.5, 0.9])],
            bundle["risk_band_cutoffs"],
        )
        self.assertEqual(digest, bundle["data_sha256"])
        self.assertEqual(
            {
                "bundle_version",
                "model_version",
                "model",
                "family",
                "features",
                "input_schema",
                "target",
                "horizon_months",
                "event_definition",
                "periods",
                "risk_band_cutoffs",
                "data_sha256",
            },
            set(bundle),
        )

    def test_load_bundle_rejects_missing_stale_or_incompatible_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._prepared_file(root)
            bundle = self._train(source)
            output = root / "pd24-model.joblib"
            self.assertIsInstance(bundle["model"], Pipeline)
            self.assertIsInstance(
                bundle["model"].named_steps["classifier"], XGBClassifier
            )
            for change, message in (
                (lambda item: item.pop("model"), "missing keys"),
                (
                    lambda item: item.__setitem__("bundle_version", BUNDLE_VERSION + 1),
                    "version",
                ),
                (
                    lambda item: item.__setitem__(
                        "model", item["model"].named_steps["classifier"]
                    ),
                    "model",
                ),
                (
                    lambda item: (
                        item["model"].named_steps["classifier"].set_params(max_depth=9)
                    ),
                    "parameters",
                ),
                (
                    lambda item: (
                        item["model"]
                        .named_steps["preprocessor"]
                        .transformers[0][1]
                        .set_params(strategy="mean")
                    ),
                    "preprocessor",
                ),
                (
                    lambda item: item.__setitem__(
                        "features", list(reversed(PRODUCT_FEATURES))
                    ),
                    "feature order",
                ),
                (
                    lambda item: item.__setitem__("risk_band_cutoffs", [0.9, 0.1]),
                    "risk_band_cutoffs",
                ),
                (
                    lambda item: item.__setitem__("risk_band_cutoffs", [0.5, 0.5]),
                    "risk_band_cutoffs",
                ),
                (lambda item: item.__setitem__("data_sha256", "0" * 64), "data_sha256"),
            ):
                invalid = copy.deepcopy(bundle)
                change(invalid)
                joblib.dump(invalid, output)
                with (
                    self.subTest(message=message),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    self._load(output, source)
            save_bundle(bundle, output)
            previous = output.read_bytes()

            def fail_dump(_: object, temporary: Path) -> None:
                temporary.write_bytes(b"partial")
                raise OSError("disk full")

            with (
                patch("src.product.joblib.dump", side_effect=fail_dump),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                save_bundle(bundle, output)
            self.assertTrue(output.exists())
            self.assertEqual(previous, output.read_bytes())
            self.assertFalse(output.with_suffix(".joblib.partial").exists())
            self.assertEqual(
                bundle["data_sha256"], self._load(output, source)["data_sha256"]
            )

    def test_synthetic_training_is_deterministic_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._prepared_file(Path(directory))
            first = self._train(source)
            second = self._train(source)
            probe = pd.DataFrame([{feature: 1 for feature in NUMERIC_FEATURES}])

        self.assertEqual(
            {key: value for key, value in first.items() if key != "model"},
            {key: value for key, value in second.items() if key != "model"},
        )
        np.testing.assert_allclose(
            first["model"].predict_proba(probe), second["model"].predict_proba(probe)
        )


if __name__ == "__main__":
    unittest.main()
