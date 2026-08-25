from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import src.pd_model as pd_model
from src.pd_model import PRODUCT_FEATURES, fit_calibrated_model


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

    def test_fit_calibrated_model_uses_only_the_frozen_five_features(self) -> None:
        development = self._sample(1, 80)
        calibration = self._sample(2, 40)

        fitted = fit_calibrated_model(development, calibration)
        probability = fitted.predict_proba(calibration[list(PRODUCT_FEATURES)])[:, 1]

        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())
        self.assertFalse(hasattr(pd_model, "train_and_select"))


if __name__ == "__main__":
    unittest.main()
