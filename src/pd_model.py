from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .metrics import calibration_table, classification_metrics


LEAKAGE_PATTERN = re.compile(r"(?i)(default|delinquen|foreclos|loss|disposition|zero_balance|current_status|target)")


@dataclass(frozen=True)
class PDConfig:
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    seed: int
    target: str = "default_24m"

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features

    def validate(self) -> None:
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("features must be non-empty and unique")
        blocked = [name for name in self.features if LEAKAGE_PATTERN.search(name)]
        if blocked:
            raise ValueError(f"outcome leakage features are prohibited: {blocked}")


class SigmoidCalibratedModel(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator: object, calibrator: LogisticRegression):
        self.estimator = estimator
        self.calibrator = calibrator
        self.classes_ = np.asarray([0, 1])

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability, 1e-8, 1 - 1e-8)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def predict_proba(self, features: object) -> np.ndarray:
        raw = self.estimator.predict_proba(features)[:, 1]
        positive = self.calibrator.predict_proba(self._logit(raw))[:, 1]
        return np.column_stack((1 - positive, positive))

    def fit(self, features: object, target: object) -> "SigmoidCalibratedModel":
        raise RuntimeError("the calibrated model is immutable after temporal fitting")


def _logistic(config: PDConfig) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(config.numeric_features),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(config.categorical_features),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=500, random_state=config.seed, solver="lbfgs"
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def _hist_gradient_boosting(config: PDConfig) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                list(config.numeric_features),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                                dtype=np.float64,
                            ),
                        ),
                    ]
                ),
                list(config.categorical_features),
            ),
        ],
        sparse_threshold=0,
    )
    classifier = HistGradientBoostingClassifier(
        class_weight="balanced",
        l2_regularization=1.0,
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        random_state=config.seed,
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def _calibrate(model: object, features: pd.DataFrame, target: pd.Series) -> SigmoidCalibratedModel:
    raw = model.predict_proba(features)[:, 1]
    calibrator = LogisticRegression(C=1e6, solver="lbfgs").fit(
        SigmoidCalibratedModel._logit(raw), target
    )
    return SigmoidCalibratedModel(model, calibrator)


def _threshold(target: pd.Series, probability: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    false_positive, true_positive, values = roc_curve(target, probability)
    finite = np.isfinite(values)
    return float(values[finite][np.argmax((true_positive - false_positive)[finite])])


def train_and_select(
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    validation: pd.DataFrame,
    config: PDConfig,
) -> dict[str, object]:
    config.validate()
    required = set(config.features).union({config.target})
    for name, frame in (("development", development), ("calibration", calibration), ("validation", validation)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
        if frame[config.target].nunique() < 2:
            raise ValueError(f"{name} must contain both outcomes")
    candidates: dict[str, object] = {}
    metrics: dict[str, dict[str, object]] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name, model in (("logistic", _logistic(config)), ("hist_gradient_boosting", _hist_gradient_boosting(config))):
        model.fit(development[list(config.features)], development[config.target])
        calibrated = _calibrate(model, calibration[list(config.features)], calibration[config.target])
        probability = calibrated.predict_proba(validation[list(config.features)])[:, 1]
        threshold = _threshold(validation[config.target], probability)
        candidates[name] = calibrated
        probabilities[name] = probability
        metrics[name] = classification_metrics(validation[config.target], probability, threshold)
    selected_name = min(
        metrics,
        key=lambda name: (
            metrics[name]["brier"],
            metrics[name]["log_loss"],
            -(metrics[name]["roc_auc"] or 0.0),
        ),
    )
    selected_probability = probabilities[selected_name]
    return {
        "selected_name": selected_name,
        "selected_model": candidates[selected_name],
        "threshold": metrics[selected_name]["threshold"],
        "metrics": metrics,
        "calibration_table": calibration_table(validation[config.target], selected_probability),
    }
