from __future__ import annotations

import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.pd_model import PRODUCT_FEATURES


class DemoTests(unittest.TestCase):
    def _demo_functions(self):
        try:
            from scripts.serve_demo import create_demo, predict_loan
        except ModuleNotFoundError as error:
            self.fail(f"the Gradio demo is unavailable: {error}")
        return create_demo, predict_loan

    @staticmethod
    def _bundle(model: object) -> dict[str, object]:
        return {
            "model": model,
            "features": list(PRODUCT_FEATURES),
            "input_schema": {
                "origination_fico": {"type": "number", "minimum": 430.0, "maximum": 844.0},
                "original_dti": {"type": "number", "minimum": 1.0, "maximum": 64.0},
                "original_cltv": {"type": "number", "minimum": 2.0, "maximum": 200.0},
                "original_interest_rate": {"type": "number", "minimum": 2.25, "maximum": 8.7},
                "number_of_borrowers": {"type": "number", "minimum": 1.0, "maximum": 2.0},
            },
            "risk_band_cutoffs": [0.5, 0.9],
        }

    def test_predict_loan_returns_pd_band_and_warning(self) -> None:
        _, predict_loan = self._demo_functions()
        model = ProbabilityModel(0.5)

        result = predict_loan(self._bundle(model), 700, 30, 80, 4.5, 2)

        self.assertEqual((50.0, "Media", "Estimación académica; no es una decisión de crédito."), result)
        self.assertEqual(list(PRODUCT_FEATURES), list(model.features.columns))

    def test_predict_loan_rejects_invalid_values_without_scoring(self) -> None:
        _, predict_loan = self._demo_functions()
        for values in (
            (None, 30, 80, 4.5, 2),
            (float("nan"), 30, 80, 4.5, 2),
            (900, 30, 80, 4.5, 2),
        ):
            with self.subTest(values=values):
                model = ProbabilityModel(0.5)
                probability, band, message = predict_loan(self._bundle(model), *values)
                self.assertIsNone(probability)
                self.assertEqual("", band)
                self.assertIn("Entrada inválida", message)
                self.assertEqual(0, model.calls)

    def test_predict_loan_hides_model_failures(self) -> None:
        _, predict_loan = self._demo_functions()

        result = predict_loan(self._bundle(ProbabilityModel(RuntimeError("private detail"))), 700, 30, 80, 4.5, 2)

        self.assertEqual((None, "", "No se pudo calcular la predicción."), result)

    def test_create_demo_has_five_inputs_and_three_outputs(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (DeprecationWarning, PendingDeprecationWarning))
            create_demo, _ = self._demo_functions()
            try:
                with patch("scripts.serve_demo.load_bundle", return_value=self._bundle(ProbabilityModel(0.5))):
                    demo = create_demo("synthetic-bundle")
            except ModuleNotFoundError as error:
                self.fail(f"the Gradio dependency is unavailable: {error}")

        self.assertEqual(5, len(demo.input_components))
        self.assertEqual(3, len(demo.output_components))


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
