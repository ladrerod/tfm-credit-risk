from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src import pd_model
from src.pd_model import PRODUCT_FEATURES, fit_model


class PDModelTests(unittest.TestCase):
    @staticmethod
    def _sample(seed: int, rows: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame(
            {
                "origination_fico": rng.integers(620, 800, rows),
                "original_dti": rng.integers(10, 45, rows),
                "original_cltv": rng.integers(50, 110, rows),
                "original_interest_rate": rng.uniform(2, 8, rows),
                "number_of_borrowers": rng.integers(1, 3, rows),
            }
        )
        frame["default_24m"] = np.tile([0, 1], rows // 2 + 1)[:rows]
        frame["not_a_product_input"] = rng.normal(size=rows)
        return frame

    def test_fit_model_uses_only_the_frozen_features(self) -> None:
        development = self._sample(1, 80)

        fitted = fit_model(development)
        probability = fitted.predict_proba(development[list(PRODUCT_FEATURES)])[:, 1]

        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())
        classifier = fitted.named_steps["classifier"]
        self.assertEqual(classifier.n_estimators, 100)
        self.assertEqual(classifier.max_depth, 2)
        self.assertEqual(classifier.learning_rate, 0.1)
        self.assertFalse(hasattr(pd_model, "train_and_select"))
        with self.assertRaisesRegex(ValueError, "binary outcomes"):
            fit_model(development.assign(default_24m=development["default_24m"] * 2))


if __name__ == "__main__":
    unittest.main()
