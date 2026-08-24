from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.product import (
    AUTHORIZED_DATA_SHA256,
    BUNDLE_VERSION,
    EVENT_DEFINITION,
    MODEL_VERSION,
    PRODUCT_FEATURES,
    _implementation_sha256,
)


class ApiTests(unittest.TestCase):
    payload = {
        "origination_fico": 700,
        "original_dti": 30,
        "original_cltv": 80,
        "original_interest_rate": 4.5,
        "number_of_borrowers": 2,
    }

    def _bundle_path(self, directory: Path, *, model: object | None = None, threshold: float = 0.5) -> Path:
        features = pd.DataFrame(
            [
                [620, 20, 70, 3.0, 1],
                [680, 28, 75, 4.0, 2],
                [720, 32, 85, 4.5, 1],
                [760, 38, 95, 5.0, 2],
            ],
            columns=PRODUCT_FEATURES,
        )
        model = model or LogisticRegression(random_state=0).fit(features, [1, 1, 0, 0])
        bundle = {
            "bundle_version": BUNDLE_VERSION,
            "model_version": MODEL_VERSION,
            "model": model,
            "selected_model_name": "logistic",
            "features": list(PRODUCT_FEATURES),
            "input_schema": {
                name: {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 200.0 if name == "original_cltv" else 1000.0,
                }
                for name in PRODUCT_FEATURES
            },
            "validation_threshold": threshold,
            "validation_metrics": {},
            "target": "default_24m",
            "horizon_months": 24,
            "event_definition": EVENT_DEFINITION,
            "development_years": [2015, 2016, 2017, 2018],
            "calibration_year": 2019,
            "validation_year": 2020,
            "test_evaluated": False,
            "data_source": "prepared_freddie_dataset",
            "data_sha256": AUTHORIZED_DATA_SHA256,
            "implementation_sha256": _implementation_sha256(),
        }
        path = directory / "pd-model.joblib"
        joblib.dump(bundle, path)
        return path

    def _client(self, directory: Path):
        from src.api import create_app

        return create_app(self._bundle_path(directory)).test_client()

    def test_health_exposes_only_public_bundle_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            response = self._client(Path(directory)).get("/health")

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "status": "ok",
                "model_version": MODEL_VERSION,
                "model_name": "logistic",
                "target": "default_24m",
                "horizon_months": 24,
                "source": "prepared_freddie_dataset",
            },
            response.get_json(),
        )

    def test_predict_scores_one_valid_loan_from_the_loaded_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._bundle_path(root)
            from src.api import create_app

            client = create_app(path).test_client()
            path.unlink()
            response = client.post("/predict", json=self.payload)

        body = response.get_json()
        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {"risk_score", "risk_level", "model_version", "horizon_months", "warning"}, set(body)
        )
        self.assertTrue(math.isfinite(body["risk_score"]))
        self.assertEqual("elevated", body["risk_level"])
        self.assertEqual(MODEL_VERSION, body["model_version"])
        self.assertEqual(24, body["horizon_months"])
        self.assertEqual("Experimental academic risk score; not a credit decision.", body["warning"])

    def test_predict_rejects_every_invalid_client_payload_with_one_stable_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = self._client(Path(directory))
            invalid_payloads = [
                {"content_type": "text/plain"},
                {"data": "{", "content_type": "application/json"},
                {"json": []},
                {"json": {"origination_fico": 700}},
                {"json": {**self.payload, "unexpected": 1}},
                {"json": {**self.payload, "origination_fico": True}},
                {"json": {**self.payload, "origination_fico": "700"}},
                {"json": {**self.payload, "origination_fico": None}},
                {"json": {**self.payload, "origination_fico": float("nan")}},
                {"json": {**self.payload, "origination_fico": float("inf")}},
                {"json": {**self.payload, "origination_fico": int("9" * 400)}},
                {"json": {**self.payload, "origination_fico": -1}},
                {"json": {**self.payload, "origination_fico": 1001}},
            ]
            for request in invalid_payloads:
                with self.subTest(request=request):
                    response = client.post("/predict", **request)
                    self.assertEqual(400, response.status_code)
                    self.assertEqual({"error": "invalid request"}, response.get_json())

    def test_predict_rejects_duplicate_json_keys_before_flask_collapses_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            response = self._client(Path(directory)).post(
                "/predict",
                data=(
                    '{"origination_fico":700,"origination_fico":700,"original_dti":30,'
                    '"original_cltv":80,"original_interest_rate":4.5,"number_of_borrowers":2}'
                ),
                content_type="application/json",
            )

        self.assertEqual(400, response.status_code)
        self.assertEqual({"error": "invalid request"}, response.get_json())

    def test_predict_rejects_bodies_over_4096_bytes_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            oversized = b" " * 4097 + json.dumps(self.payload).encode()
            response = self._client(Path(directory)).post(
                "/predict",
                data=oversized,
                content_type="application/json",
            )

        self.assertEqual(400, response.status_code)
        self.assertEqual({"error": "invalid request"}, response.get_json())

    def test_predict_rejects_deep_json_under_the_limit_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nested = "[" * 1500 + "0" + "]" * 1500
            self.assertLess(len(nested), 4096)
            with patch("src.api.json.loads", side_effect=RecursionError("nested JSON")):
                response = self._client(Path(directory)).post(
                    "/predict",
                    data=nested,
                    content_type="application/json",
                )

        self.assertEqual(400, response.status_code)
        self.assertEqual({"error": "invalid request"}, response.get_json())

    def test_app_has_only_health_and_predict_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = self._client(Path(directory))

        self.assertEqual({"/health", "/predict"}, {rule.rule for rule in client.application.url_map.iter_rules()})

    def test_app_rejects_a_defective_bundle_before_serving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._bundle_path(root)
            bundle = joblib.load(path)
            bundle["validation_threshold"] = float("nan")
            joblib.dump(bundle, path)
            from src.api import create_app

            with self.assertRaisesRegex(ValueError, "validation_threshold"):
                create_app(path)

    def test_predict_treats_nonfinite_model_output_as_an_internal_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._bundle_path(Path(directory), model=ProbabilityModel(float("nan")))
            from src.api import create_app

            response = create_app(path).test_client().post("/predict", json=self.payload)

        self.assertEqual(500, response.status_code)
        self.assertEqual({"error": "internal server error"}, response.get_json())

    def test_predict_treats_model_exceptions_as_an_internal_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._bundle_path(Path(directory), model=ProbabilityModel(RuntimeError("model failure")))
            from src.api import create_app

            response = create_app(path).test_client().post("/predict", json=self.payload)

        self.assertEqual(500, response.status_code)
        self.assertEqual({"error": "internal server error"}, response.get_json())

    def test_predict_marks_score_at_the_threshold_as_elevated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._bundle_path(Path(directory), model=ProbabilityModel(0.5), threshold=0.5)
            from src.api import create_app

            response = create_app(path).test_client().post("/predict", json=self.payload)

        self.assertEqual(200, response.status_code)
        self.assertEqual("elevated", response.get_json()["risk_level"])


class ProbabilityModel:
    def __init__(self, result: float | Exception) -> None:
        self.result = result

    def predict_proba(self, _: pd.DataFrame) -> np.ndarray:
        if isinstance(self.result, Exception):
            raise self.result
        return np.array([[1 - self.result, self.result]])


if __name__ == "__main__":
    unittest.main()
