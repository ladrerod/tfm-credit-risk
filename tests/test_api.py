from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.pd_model import INTEGER_FEATURES, PRODUCT_FEATURES, fit_model
from src.product import (
    BUNDLE_VERSION,
    EVENT_DEFINITION,
    EXPECTED_DATA_SHA256,
    FIT_LABEL_CUTOFF,
    MODEL_VERSION,
    REFERENCE_BAND_YEARS,
    REFERENCE_DEVELOPMENT_YEARS,
)

PAYLOAD = {
    "origination_fico": 700,
    "original_dti": 30,
    "original_cltv": 80,
    "original_interest_rate": 4.5,
    "number_of_borrowers": 2,
}


class ApiTests(unittest.TestCase):
    def _bundle(
        self, *, model: object | None = None, cutoffs: list[float] | None = None
    ) -> dict[str, object]:
        features = pd.DataFrame(
            [
                {**PAYLOAD, "origination_fico": fico, "original_dti": dti}
                for fico, dti in ((620, 20), (680, 28), (720, 32), (760, 38))
            ]
        )
        fitted = fit_model(features.assign(default_24m=[0, 0, 1, 1]))
        return {
            "bundle_version": BUNDLE_VERSION,
            "model_version": MODEL_VERSION,
            "model": fitted if model is None else model,
            "family": "xgboost",
            "features": list(PRODUCT_FEATURES),
            "input_schema": {
                name: {
                    "type": "integer" if name in INTEGER_FEATURES else "number",
                    "minimum": 0.0,
                    "maximum": 1000.0,
                }
                for name in PRODUCT_FEATURES
            },
            "target": "default_24m",
            "horizon_months": 24,
            "event_definition": EVENT_DEFINITION,
            "periods": {
                "development_years": list(REFERENCE_DEVELOPMENT_YEARS),
                "band_reference_years": list(REFERENCE_BAND_YEARS),
                "fit_label_cutoff": FIT_LABEL_CUTOFF.isoformat(),
            },
            "risk_band_cutoffs": [0.5, 0.9] if cutoffs is None else cutoffs,
            "data_sha256": EXPECTED_DATA_SHA256,
        }

    def _bundle_path(self, directory: Path) -> Path:
        path = directory / "pd24-model.joblib"
        joblib.dump(self._bundle(), path)
        return path

    def _client(self):
        from src.api import create_app

        with patch("src.api.load_bundle", return_value=self._bundle()):
            return create_app("synthetic-bundle").test_client()

    def _client_with_model(self, result: float | Exception):
        from src.api import create_app

        model = ProbabilityModel(result)
        with patch("src.api.load_bundle", return_value=self._bundle(model=model)):
            client = create_app("synthetic-bundle").test_client()
        return client, model

    def test_health_uses_only_current_bundle_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            from src.api import create_app

            response = (
                create_app(self._bundle_path(Path(directory)))
                .test_client()
                .get("/health")
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {"status": "ok", "model_version": MODEL_VERSION, "horizon_months": 24},
            response.get_json(),
        )

    def test_predict_returns_the_exact_public_score_contract(self) -> None:
        client, model = self._client_with_model(0.5)

        response = client.post("/predict", json=PAYLOAD)

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            {
                "risk_score": 0.5,
                "risk_band": "medium",
                "model_version": MODEL_VERSION,
                "horizon_months": 24,
                "warning": "Academic risk estimate; not a credit decision.",
            },
            response.get_json(),
        )
        self.assertEqual(1, model.calls)
        self.assertEqual(list(PRODUCT_FEATURES), list(model.features.columns))

    def test_predict_assigns_low_medium_and_high_at_p50_and_p90_edges(self) -> None:
        for score, expected in (
            (0.499, "low"),
            (0.5, "medium"),
            (0.899, "medium"),
            (0.9, "high"),
            (0.901, "high"),
        ):
            with self.subTest(score=score):
                client, _ = self._client_with_model(score)
                self.assertEqual(
                    expected,
                    client.post("/predict", json=PAYLOAD).get_json()["risk_band"],
                )

    def test_predict_rejects_invalid_client_payloads_without_echoing_them(self) -> None:
        client = self._client()
        duplicate = json.dumps(PAYLOAD)[:-1] + ',"origination_fico":701}'
        invalid_payloads = [
            {"data": json.dumps(PAYLOAD), "content_type": "text/plain"},
            {"data": "{", "content_type": "application/json"},
            {"data": duplicate, "content_type": "application/json"},
            {"data": " " * 5000, "content_type": "application/json"},
            {"json": []},
            {"json": [{}]},
            {"json": {"origination_fico": 700}},
            {"json": {**PAYLOAD, "unexpected": 1}},
            {"json": {**PAYLOAD, "origination_fico": True}},
            {"json": {**PAYLOAD, "origination_fico": "700"}},
            {"json": {**PAYLOAD, "origination_fico": 700.5}},
            {"json": {**PAYLOAD, "origination_fico": None}},
            {"json": {**PAYLOAD, "origination_fico": float("nan")}},
            {"json": {**PAYLOAD, "origination_fico": float("inf")}},
            {"json": {**PAYLOAD, "origination_fico": int("9" * 400)}},
            {"json": {**PAYLOAD, "origination_fico": -1}},
            {"json": {**PAYLOAD, "origination_fico": 1001}},
        ]
        for item in invalid_payloads:
            with self.subTest(item=item):
                response = client.post("/predict", **item)
                self.assertEqual(400, response.status_code)
                self.assertEqual({"error": "invalid request"}, response.get_json())

    def test_predict_hides_model_failures_and_invalid_probabilities(self) -> None:
        for result in (RuntimeError("model failure"), float("nan"), 1.1):
            with self.subTest(result=result):
                client, _ = self._client_with_model(result)
                response = client.post("/predict", json=PAYLOAD)
                self.assertEqual(500, response.status_code)
                self.assertEqual(
                    {"error": "internal server error"}, response.get_json()
                )

    def test_app_has_only_health_and_predict_routes(self) -> None:
        client = self._client()

        self.assertEqual(
            {"/health", "/predict"},
            {rule.rule for rule in client.application.url_map.iter_rules()},
        )


class ProbabilityModel:
    def __init__(self, result: float | Exception) -> None:
        self.result = result
        self.calls = 0
        self.features = pd.DataFrame()

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        self.features = features.copy()
        if isinstance(self.result, Exception):
            raise self.result
        return np.array([[1 - self.result, self.result]])


if __name__ == "__main__":
    unittest.main()
