from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PRODUCT_FEATURES = (
    "origination_fico",
    "original_dti",
    "original_cltv",
    "original_interest_rate",
    "number_of_borrowers",
)


class SigmoidCalibratedModel:
    def __init__(self, estimator: Pipeline, calibrator: LogisticRegression):
        self.estimator = estimator
        self.calibrator = calibrator
        self.classes_ = np.asarray([0, 1])

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability, 1e-8, 1 - 1e-8)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raw = self.estimator.predict_proba(features)[:, 1]
        positive = self.calibrator.predict_proba(self._logit(raw))[:, 1]
        return np.column_stack((1 - positive, positive))


def _logistic(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=seed, solver="lbfgs")),
        ]
    )


def fit_calibrated_model(
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    seed: int = 20260819,
) -> SigmoidCalibratedModel:
    required = set(PRODUCT_FEATURES).union({"default_24m"})
    for name, frame in (("development", development), ("calibration", calibration)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
        if frame["default_24m"].nunique() != 2:
            raise ValueError(f"{name} must contain both outcomes")
    estimator = _logistic(seed)
    estimator.fit(development[list(PRODUCT_FEATURES)], development["default_24m"])
    raw = estimator.predict_proba(calibration[list(PRODUCT_FEATURES)])[:, 1]
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=seed).fit(
        SigmoidCalibratedModel._logit(raw), calibration["default_24m"]
    )
    return SigmoidCalibratedModel(estimator, calibrator)
