from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 1) - 1)))

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


CANONICAL_STATES = ("current", "30", "60", "90_plus", "default", "prepay")
MINIMUM_CALIBRATION_CLASS_ROWS = 30
_CURE_STATES = {"30", "60", "90_plus"}
_DEFAULT_NUMERIC_FEATURES = (
    "months_since_first_payment",
    "monthly_ead_ratio",
    "current_interest_rate",
    "original_cltv",
    "original_dti",
    "origination_fico",
)
_DEFAULT_CATEGORICAL_FEATURES = ("current_state", "occupancy_status", "property_state")


@dataclass(frozen=True)
class MultiStateConfig:
    numeric_features: tuple[str, ...] = _DEFAULT_NUMERIC_FEATURES
    categorical_features: tuple[str, ...] = _DEFAULT_CATEGORICAL_FEATURES
    seed: int = 42
    target: str = "next_state"
    current_state: str = "current_state"
    loan_key: str = "loan_key"

    def __post_init__(self) -> None:
        features = self.features
        if not features or len(set(features)) != len(features):
            raise ValueError("features must be non-empty and unique")
        if self.target in features or self.loan_key in features:
            raise ValueError("target and loan_key cannot be model features")

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features


def cumulative_incidence(default_hazard: np.ndarray, prepay_hazard: np.ndarray) -> dict[str, np.ndarray]:
    """Return discrete competing-risk incidence for each supported horizon."""
    default_hazard = np.asarray(default_hazard, dtype=float)
    prepay_hazard = np.asarray(prepay_hazard, dtype=float)
    if default_hazard.ndim != 1 or prepay_hazard.ndim != 1 or default_hazard.shape != prepay_hazard.shape:
        raise ValueError("default and prepay hazards must be one-dimensional arrays of equal length")
    if (
        not np.isfinite(default_hazard).all()
        or not np.isfinite(prepay_hazard).all()
        or (default_hazard < 0).any()
        or (prepay_hazard < 0).any()
        or (default_hazard + prepay_hazard > 1).any()
    ):
        raise ValueError("competing hazards must be finite, non-negative, and sum to at most one")
    survival = np.empty(len(default_hazard), dtype=float)
    default = np.empty(len(default_hazard), dtype=float)
    prepay = np.empty(len(default_hazard), dtype=float)
    remaining, default_total, prepay_total = 1.0, 0.0, 0.0
    for index, (default_probability, prepay_probability) in enumerate(zip(default_hazard, prepay_hazard, strict=True)):
        default_total += remaining * default_probability
        prepay_total += remaining * prepay_probability
        remaining *= 1.0 - default_probability - prepay_probability
        survival[index] = remaining
        default[index] = default_total
        prepay[index] = prepay_total
    return {"survival": survival, "default": default, "prepay": prepay}


def _canonical_probability(probability: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    labels = tuple(str(value) for value in classes)
    if probability.ndim != 2 or probability.shape[1] != len(labels):
        raise ValueError("probability columns must match classes")
    if len(set(labels)) != len(labels) or set(labels) != set(CANONICAL_STATES):
        raise ValueError("classes must contain each canonical state exactly once")
    aligned = probability[:, [labels.index(state) for state in CANONICAL_STATES]]
    if not np.isfinite(aligned).all() or (aligned < 0).any() or not np.allclose(aligned.sum(axis=1), 1.0):
        raise ValueError("probabilities must be finite, non-negative, and sum to one")
    return aligned


def _one_vs_rest(target: pd.Series, probability: np.ndarray, state: str) -> dict[str, float | int | None]:
    observed = target.eq(state).to_numpy(dtype=int)
    metrics: dict[str, float | int | None] = {
        "observed": int(observed.sum()),
        "expected": float(probability.sum()),
        "prevalence": float(observed.mean()),
        "brier": float(brier_score_loss(observed, probability)),
        "log_loss": float(log_loss(observed, probability, labels=[0, 1])),
        "roc_auc": None,
        "pr_auc": None,
    }
    if observed.min() != observed.max():
        metrics["roc_auc"] = float(roc_auc_score(observed, probability))
        metrics["pr_auc"] = float(average_precision_score(observed, probability))
    return metrics


def _incidence_by_horizon(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, object]:
    period_column = "months_since_first_payment" if "months_since_first_payment" in frame else None
    default_index = CANONICAL_STATES.index("default")
    prepay_index = CANONICAL_STATES.index("prepay")
    if period_column is None:
        return {"available": False, "reason": "stable monthly horizon is unavailable"}
    period = pd.to_numeric(frame[period_column], errors="coerce")
    valid = period.notna() & period.ge(0) & period.eq(np.floor(period))
    if not valid.any():
        return {"available": False, "reason": "stable monthly horizon is unavailable"}
    grouped = pd.DataFrame(
        {
            "period": period.loc[valid].astype(int).to_numpy(),
            "default": probability[valid.to_numpy(), default_index],
            "prepay": probability[valid.to_numpy(), prepay_index],
        }
    ).groupby("period", sort=True)[["default", "prepay"]].mean()
    periods = grouped.index.to_numpy(dtype=int)
    expected = np.arange(periods[-1] + 1)
    if not np.array_equal(periods, expected):
        return {"available": False, "reason": "monthly horizons are not contiguous from first payment"}
    horizons = periods + 1
    default_hazard = grouped["default"].to_numpy()
    prepay_hazard = grouped["prepay"].to_numpy()
    incidence = cumulative_incidence(default_hazard, prepay_hazard)
    return {
        "available": True,
        "method": "model-expected competing risks by observed monthly horizon",
        "default_definition": "terminal next_state default only; excludes 90_plus",
        "horizons": horizons.tolist(),
        **{name: values.tolist() for name, values in incidence.items()},
    }


def multistate_metrics(frame: pd.DataFrame, probability: np.ndarray, classes: Sequence[str]) -> dict[str, object]:
    """Score predictions without returning loan-level data."""
    required = {"next_state", "current_state"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    target = frame["next_state"].astype(str)
    invalid = sorted(set(target).difference(CANONICAL_STATES))
    if invalid:
        raise ValueError(f"invalid next states: {invalid}")
    invalid_current = sorted(set(frame["current_state"].astype(str)).difference(CANONICAL_STATES))
    if invalid_current:
        raise ValueError(f"invalid current states: {invalid_current}")
    probability = _canonical_probability(probability, classes)
    if len(frame) != len(probability):
        raise ValueError("frame and probability must have the same number of rows")
    observed = {state: int(target.eq(state).sum()) for state in CANONICAL_STATES}
    expected = {state: float(probability[:, index].sum()) for index, state in enumerate(CANONICAL_STATES)}
    one_hot = np.equal(target.to_numpy()[:, None], np.asarray(CANONICAL_STATES)).astype(float)
    cure_eligible = frame["current_state"].astype(str).isin(_CURE_STATES).to_numpy()
    cure_observed = target.eq("current").to_numpy()[cure_eligible]
    cure_probability = probability[cure_eligible, CANONICAL_STATES.index("current")]
    cures: dict[str, float | int | None] = {
        "eligible": int(cure_eligible.sum()),
        "observed": int(cure_observed.sum()),
        "expected": float(cure_probability.sum()),
        "observed_rate": float(cure_observed.mean()) if len(cure_observed) else None,
        "expected_rate": float(cure_probability.mean()) if len(cure_probability) else None,
        "brier": float(brier_score_loss(cure_observed, cure_probability)) if len(cure_observed) else None,
    }
    ordered_states = sorted(CANONICAL_STATES)
    ordered_probability = probability[:, [CANONICAL_STATES.index(state) for state in ordered_states]]
    return {
        "multiclass_log_loss": float(log_loss(target, ordered_probability, labels=ordered_states)),
        "multiclass_brier": float(np.mean(np.sum((one_hot - probability) ** 2, axis=1))),
        "observed_state_counts": observed,
        "expected_state_counts": expected,
        "one_vs_rest": {
            "default": _one_vs_rest(target, probability[:, CANONICAL_STATES.index("default")], "default"),
            "prepay": _one_vs_rest(target, probability[:, CANONICAL_STATES.index("prepay")], "prepay"),
        },
        "cures": cures,
        "cumulative_incidence": _incidence_by_horizon(frame, probability),
    }


def _validate_cohort(
    frame: pd.DataFrame, config: MultiStateConfig, name: str, *, require_all_states: bool
) -> None:
    required = set(config.features).union({config.target, config.current_state, config.loan_key})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} is empty")
    target = frame[config.target].astype(str)
    invalid = sorted(set(target).difference(CANONICAL_STATES))
    if invalid:
        raise ValueError(f"{name} has invalid next states: {invalid}")
    if require_all_states:
        missing_states = sorted(set(CANONICAL_STATES).difference(target))
        if missing_states:
            raise ValueError(f"{name} is missing canonical next states: {missing_states}")
    current = frame[config.current_state].astype(str)
    invalid_current = sorted(set(current).difference(CANONICAL_STATES))
    if invalid_current:
        raise ValueError(f"{name} has invalid current states: {invalid_current}")
    if frame[config.loan_key].isna().any():
        raise ValueError(f"{name} has missing loan_key values")


def _assert_non_overlapping(cohorts: dict[str, pd.DataFrame], loan_key: str) -> None:
    names = list(cohorts)
    for index, left_name in enumerate(names):
        left = set(cohorts[left_name][loan_key])
        for right_name in names[index + 1 :]:
            if left.intersection(cohorts[right_name][loan_key]):
                raise ValueError(f"cohort populations overlap: {left_name} and {right_name}")


def _logistic_model(config: MultiStateConfig) -> Pipeline:
    transform = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(config.numeric_features),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(config.categorical_features),
            ),
        ]
    )
    return Pipeline(
        [("features", transform), ("model", LogisticRegression(max_iter=1_000, random_state=config.seed))]
    )


def _hgb_model(config: MultiStateConfig) -> Pipeline:
    categorical_start = len(config.numeric_features)
    transform = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), list(config.numeric_features)),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
                    ]
                ),
                list(config.categorical_features),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", transform),
            (
                "model",
                HistGradientBoostingClassifier(
                    categorical_features=list(range(categorical_start, categorical_start + len(config.categorical_features))),
                    early_stopping=False,
                    random_state=config.seed,
                ),
            ),
        ]
    )


def _fit_calibrated(model: Pipeline, development: pd.DataFrame, calibration: pd.DataFrame, config: MultiStateConfig) -> CalibratedClassifierCV:
    features = list(config.features)
    fitted = model.fit(development[features], development[config.target])
    return CalibratedClassifierCV(FrozenEstimator(fitted), method="sigmoid").fit(
        calibration[features], calibration[config.target]
    )


def train_and_compare_multistate(
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    validation: pd.DataFrame,
    config: MultiStateConfig,
) -> dict[str, object]:
    """Fit the fixed logistic champion and an HGB challenger on separate temporal cohorts."""
    cohorts = {"development": development, "calibration": calibration, "validation": validation}
    for name, frame in cohorts.items():
        _validate_cohort(frame, config, name, require_all_states=name != "validation")
    _assert_non_overlapping(cohorts, config.loan_key)
    candidates = {
        "multinomial_logistic": _fit_calibrated(_logistic_model(config), development, calibration, config),
        "hist_gradient_boosting": _fit_calibrated(_hgb_model(config), development, calibration, config),
    }
    metrics = {
        name: multistate_metrics(
            validation.rename(columns={config.target: "next_state", config.current_state: "current_state"}),
            model.predict_proba(validation[list(config.features)]),
            model.classes_,
        )
        for name, model in candidates.items()
    }
    class_counts = {
        name: {
            state: int(frame[config.target].astype(str).eq(state).sum())
            for state in CANONICAL_STATES
        }
        for name, frame in cohorts.items()
    }
    calibration_adequacy = {
        "minimum_rows_per_class": MINIMUM_CALIBRATION_CLASS_ROWS,
        "development": min(class_counts["development"].values()) >= MINIMUM_CALIBRATION_CLASS_ROWS,
        "calibration": min(class_counts["calibration"].values()) >= MINIMUM_CALIBRATION_CLASS_ROWS,
    }
    calibration_adequacy["suitable"] = bool(
        calibration_adequacy["development"] and calibration_adequacy["calibration"]
    )
    return {
        "champion_name": "multinomial_logistic",
        "validation_metrics": metrics,
        "class_counts": class_counts,
        "calibration_adequacy": calibration_adequacy,
        "models": candidates,
    }
