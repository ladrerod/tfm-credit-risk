from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .data_access import load_compact
from .pd_model import (
    INTEGER_FEATURES,
    NUMERIC_FEATURES,
    PRODUCT_FEATURES,
    fit_model,
)

BUNDLE_VERSION = 7
MODEL_VERSION = "pd24-xgboost-5v-7"
EXPECTED_DATA_SHA256 = (
    "316dc7b4c16878d9ca62694626263e3a9276512809000d003f456fffe0740589"
)
TARGET = "default_24m"
HORIZON_MONTHS = 24
EVENT_DEFINITION = "first 90+ delinquency or credit zero-balance code 02, 03, 09 or 15 within 24 months"
REFERENCE_DEVELOPMENT_YEARS = (2013, 2014, 2015, 2016, 2017)
REFERENCE_BAND_YEARS = (2018, 2019)
FIT_LABEL_CUTOFF = pd.Timestamp("2023-01-01")
REQUIRED_BUNDLE_KEYS = {
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
}


def _input_schema(development: pd.DataFrame) -> dict[str, dict[str, object]]:
    schema: dict[str, dict[str, object]] = {}
    for feature in PRODUCT_FEATURES:
        values = development[feature].dropna()
        if values.empty:
            raise ValueError(f"{feature} has no development values")
        schema[feature] = {
            "type": "integer" if feature in INTEGER_FEATURES else "number",
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return schema


def _fit_frame(path: str | Path, chunksize: int) -> tuple[pd.DataFrame, str]:
    frame, data_sha256 = load_compact(
        path,
        EXPECTED_DATA_SHA256,
        years=REFERENCE_DEVELOPMENT_YEARS + REFERENCE_BAND_YEARS,
        chunksize=chunksize,
    )
    # Keep the same reconciled population used by the 5/16/18 comparison.
    frame = frame.loc[
        frame["original_upb"].notna()
        & frame["pd_label_available_date"].lt(FIT_LABEL_CUTOFF)
    ].copy()
    if frame.empty:
        raise ValueError("prepared file contains no rows before the fit cutoff")
    return frame, data_sha256


def train_product(path: str | Path, *, chunksize: int = 100_000) -> dict[str, object]:
    frame, data_sha256 = _fit_frame(path, chunksize)
    development = frame.loc[frame["cohort_year"].isin(REFERENCE_DEVELOPMENT_YEARS)]
    band_reference = frame.loc[frame["cohort_year"].isin(REFERENCE_BAND_YEARS)]
    model = fit_model(development)
    reference_probability = model.predict_proba(band_reference[list(PRODUCT_FEATURES)])[
        :, 1
    ]
    if (
        not np.isfinite(reference_probability).all()
        or ((reference_probability < 0) | (reference_probability > 1)).any()
    ):
        raise RuntimeError("model returned invalid reference probabilities")
    cutoffs = np.quantile(reference_probability, [0.5, 0.9])
    bundle: dict[str, object] = {
        "bundle_version": BUNDLE_VERSION,
        "model_version": MODEL_VERSION,
        "model": model,
        "family": "xgboost",
        "features": list(PRODUCT_FEATURES),
        "input_schema": _input_schema(development),
        "target": TARGET,
        "horizon_months": HORIZON_MONTHS,
        "event_definition": EVENT_DEFINITION,
        "periods": {
            "development_years": list(REFERENCE_DEVELOPMENT_YEARS),
            "band_reference_years": list(REFERENCE_BAND_YEARS),
            "fit_label_cutoff": FIT_LABEL_CUTOFF.isoformat(),
        },
        "risk_band_cutoffs": [float(cutoffs[0]), float(cutoffs[1])],
        "data_sha256": data_sha256,
    }
    return bundle


def save_bundle(bundle: dict[str, object], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    try:
        joblib.dump(bundle, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_bundle(path: str | Path) -> dict[str, object]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise TypeError("model bundle must be a dictionary")
    if set(bundle) != REQUIRED_BUNDLE_KEYS:
        raise ValueError("model bundle is missing keys or contains unsupported content")
    if (
        type(bundle["bundle_version"]) is not int
        or bundle["bundle_version"] != BUNDLE_VERSION
    ):
        raise ValueError("unsupported model bundle version")
    if bundle["model_version"] != MODEL_VERSION or bundle["family"] != "xgboost":
        raise ValueError("model bundle model identity is incompatible")
    model = bundle["model"]
    if (
        type(model) is not Pipeline
        or set(model.named_steps) != {"preprocessor", "classifier"}
        or type(model.named_steps["preprocessor"]) is not ColumnTransformer
        or type(model.named_steps["classifier"]) is not XGBClassifier
        or list(model.feature_names_in_) != list(PRODUCT_FEATURES)
    ):
        raise ValueError("model bundle model is incompatible")
    preprocessor = model.named_steps["preprocessor"]
    if len(preprocessor.transformers) != 1:
        raise ValueError("model bundle preprocessor is incompatible")
    numeric_name, numeric, numeric_columns = preprocessor.transformers[0]
    if (
        numeric_name != "numeric"
        or type(numeric) is not SimpleImputer
        or numeric.strategy != "median"
        or list(numeric_columns) != list(NUMERIC_FEATURES)
        or preprocessor.remainder != "drop"
        or preprocessor.sparse_threshold != 0.0
    ):
        raise ValueError("model bundle preprocessor is incompatible")
    classifier = model.named_steps["classifier"]
    if (
        classifier.n_estimators != 100
        or classifier.max_depth != 2
        or classifier.learning_rate != 0.1
        or classifier.n_jobs != 1
        or classifier.random_state != 20260831
    ):
        raise ValueError("model bundle model parameters are incompatible")
    if bundle["features"] != list(PRODUCT_FEATURES):
        raise ValueError("model bundle feature order does not match the product")
    schema = bundle["input_schema"]
    if not isinstance(schema, Mapping) or set(schema) != set(PRODUCT_FEATURES):
        raise ValueError("model bundle input_schema is incompatible")
    for feature in PRODUCT_FEATURES:
        limits = schema[feature]
        if not isinstance(limits, Mapping):
            raise TypeError("model bundle input_schema is incompatible")
        if (
            set(limits) != {"type", "minimum", "maximum"}
            or limits["type"]
            != ("integer" if feature in INTEGER_FEATURES else "number")
            or type(limits["minimum"]) not in (int, float)
            or type(limits["maximum"]) not in (int, float)
            or not math.isfinite(limits["minimum"])
            or not math.isfinite(limits["maximum"])
            or limits["minimum"] > limits["maximum"]
        ):
            raise ValueError("model bundle input_schema is incompatible")
    if (
        bundle["target"] != TARGET
        or bundle["horizon_months"] != HORIZON_MONTHS
        or bundle["event_definition"] != EVENT_DEFINITION
    ):
        raise ValueError("model bundle definition is incompatible")
    if bundle["periods"] != {
        "development_years": list(REFERENCE_DEVELOPMENT_YEARS),
        "band_reference_years": list(REFERENCE_BAND_YEARS),
        "fit_label_cutoff": FIT_LABEL_CUTOFF.isoformat(),
    }:
        raise ValueError("model bundle periods are incompatible")
    cutoffs = bundle["risk_band_cutoffs"]
    if not (
        isinstance(cutoffs, list)
        and len(cutoffs) == 2
        and all(
            type(value) in (int, float) and math.isfinite(value) for value in cutoffs
        )
        and 0 <= cutoffs[0] < cutoffs[1] <= 1
    ):
        raise ValueError("model bundle risk_band_cutoffs are incompatible")
    if bundle["data_sha256"] != EXPECTED_DATA_SHA256:
        raise ValueError("model bundle data_sha256 is incompatible")
    return bundle


def score_product(
    bundle: Mapping[str, object], payload: Mapping[str, object]
) -> tuple[float, str]:
    features = bundle["features"]
    if (
        not isinstance(features, list)
        or not isinstance(payload, Mapping)
        or set(payload) != set(features)
    ):
        raise ValueError("invalid input")
    schema = bundle["input_schema"]
    for name in features:
        value = payload[name]
        limits = schema[name]
        if type(value) not in (int, float) or (
            type(value) is float and not math.isfinite(value)
        ):
            raise ValueError("invalid input")
        if name in INTEGER_FEATURES and type(value) is float and not value.is_integer():
            raise ValueError("invalid input")
        if value < limits["minimum"] or value > limits["maximum"]:
            raise ValueError("invalid input")
    score = float(
        bundle["model"].predict_proba(pd.DataFrame([payload], columns=features))[0, 1]
    )
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise RuntimeError("model returned an invalid probability")
    p50, p90 = bundle["risk_band_cutoffs"]
    return score, "low" if score < p50 else "medium" if score < p90 else "high"
