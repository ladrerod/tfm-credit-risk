from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def regression_metrics(observed: object, predicted: object) -> dict[str, float | int]:
    actual = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or estimate.ndim != 1 or len(actual) != len(estimate) or len(actual) == 0:
        raise ValueError("regression vectors must be aligned and non-empty")
    if not np.isfinite(actual).all() or not np.isfinite(estimate).all():
        raise ValueError("regression vectors must be finite")
    error = estimate - actual
    denominator = float(np.abs(actual).sum())
    return {
        "n": int(len(actual)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(np.abs(error).sum() / denominator) if denominator else 0.0,
        "portfolio_relative_error": (
            float(abs(error.sum()) / denominator) if denominator else 0.0
        ),
        "observed_total": float(actual.sum()),
        "predicted_total": float(estimate.sum()),
        "total_error": float(error.sum()),
    }


class ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def fit(self, features: pd.DataFrame, target: object) -> "ConstantRegressor":
        if len(features) == 0 or len(features) != len(np.asarray(target)):
            raise ValueError("constant model inputs are invalid")
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.value)


class SegmentMeanRegressor:
    def fit(self, features: pd.DataFrame, target: object) -> "SegmentMeanRegressor":
        values = np.asarray(target, dtype=float)
        bands = pd.cut(features["original_cltv"], bins=[-np.inf, 80.0, np.inf])
        self.global_mean_ = float(values.mean())
        self.band_means_ = pd.DataFrame({"band": bands, "target": values}).groupby(
            "band", observed=True
        )["target"].mean().to_dict()
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        bands = pd.cut(features["original_cltv"], bins=[-np.inf, 80.0, np.inf])
        return np.asarray([self.band_means_.get(value, self.global_mean_) for value in bands])


def _linear_transform(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def _tree_transform(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                                dtype=np.float64,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )


def build_huber_regressor(*, numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        [("features", _linear_transform(numeric, categorical)), ("model", HuberRegressor(max_iter=1000))]
    )


def build_hgb_regressor(*, numeric: list[str], categorical: list[str], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("features", _tree_transform(numeric, categorical)),
            (
                "model",
                HistGradientBoostingRegressor(
                    l2_regularization=1.0,
                    learning_rate=0.05,
                    max_iter=150,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    random_state=seed,
                ),
            ),
        ]
    )


class HurdleLGD:
    def __init__(self, *, numeric: list[str], categorical: list[str], seed: int):
        self.numeric = numeric
        self.categorical = categorical
        self.seed = seed

    def fit(self, features: pd.DataFrame, target: object) -> "HurdleLGD":
        values = np.asarray(target, dtype=float)
        positive = values > 0
        if not positive.any() or positive.all():
            raise ValueError("hurdle LGD requires zero and positive observations")
        self.occurrence_ = Pipeline(
            [
                ("features", _linear_transform(self.numeric, self.categorical)),
                ("model", LogisticRegression(max_iter=500, random_state=self.seed)),
            ]
        ).fit(features, positive.astype(int))
        self.severity_ = build_hgb_regressor(
            numeric=self.numeric, categorical=self.categorical, seed=self.seed
        ).fit(features.loc[positive], values[positive])
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        occurrence = self.occurrence_.predict_proba(features)[:, 1]
        severity = np.clip(self.severity_.predict(features), 0.0, 2.0)
        return np.clip(occurrence * severity, 0.0, 2.0)


def train_loss_models(
    ead_development: pd.DataFrame,
    ead_validation: pd.DataFrame,
    *,
    lgd_development: pd.DataFrame | None = None,
    lgd_validation: pd.DataFrame | None = None,
    numeric: list[str],
    categorical: list[str],
    seed: int,
) -> dict[str, object]:
    features = numeric + categorical
    lgd_development = ead_development if lgd_development is None else lgd_development
    lgd_validation = ead_validation if lgd_validation is None else lgd_validation
    populations = (
        ("EAD development", ead_development, "ead_ratio"),
        ("EAD validation", ead_validation, "ead_ratio"),
        ("LGD development", lgd_development, "lgd"),
        ("LGD validation", lgd_validation, "lgd"),
    )
    for name, frame, target in populations:
        missing = sorted(set(features).union({target}).difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
    ead_candidates = {
        "constant_1": ConstantRegressor(1.0),
        "hist_gradient_boosting": build_hgb_regressor(numeric=numeric, categorical=categorical, seed=seed),
    }
    lgd_candidates = {
        "segment_mean": SegmentMeanRegressor(),
        "direct_huber": build_huber_regressor(numeric=numeric, categorical=categorical),
    }
    lgd_values = np.asarray(lgd_development["lgd"], dtype=float)
    if (lgd_values == 0).any() and (lgd_values > 0).any():
        lgd_candidates["hurdle"] = HurdleLGD(numeric=numeric, categorical=categorical, seed=seed)

    def evaluate(
        candidates: dict[str, object],
        development: pd.DataFrame,
        validation: pd.DataFrame,
        target: str,
        upper: float,
    ) -> dict[str, object]:
        metrics = {}
        for name, model in candidates.items():
            model.fit(development[features], development[target])
            prediction = np.clip(model.predict(validation[features]), 0.0, upper)
            metrics[name] = regression_metrics(validation[target], prediction)
        selected = min(
            metrics,
            key=lambda name: (
                metrics[name]["portfolio_relative_error"],
                metrics[name]["mae"],
            ),
        )
        return {
            "selected_name": selected,
            "selected_model": candidates[selected],
            "metrics": metrics,
        }

    ead = evaluate(ead_candidates, ead_development, ead_validation, "ead_ratio", 1.5)
    lgd = evaluate(lgd_candidates, lgd_development, lgd_validation, "lgd", 2.0)
    decision_grade = bool(
        ead["metrics"][ead["selected_name"]]["portfolio_relative_error"] <= 0.15
        and lgd["metrics"][lgd["selected_name"]]["portfolio_relative_error"] <= 0.50
    )
    return {"ead": ead, "lgd": lgd, "decision_grade": decision_grade}
