from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data_access import load_compact
from .integrity import file_sha256
from .metrics import (
    binned_distribution,
    calibration_table,
    classification_metrics,
    population_stability_index,
    quantile_breaks,
    risk_band_cutoffs,
    risk_bands,
)
from .pd_model import PRODUCT_FEATURES, fit_calibrated_model


BUNDLE_VERSION = 2
MODEL_VERSION = "five-variable-pd-2"
EXPECTED_DATA_SHA256 = "33f69db7662dee9265bc28cb6d01b4099d6727641c1d6e4a579930df5149fb91"
TARGET = "default_24m"
HORIZON_MONTHS = 24
EVENT_DEFINITION = "first 90+ delinquency or credit zero-balance code 02, 03, 09 or 15 within 24 months"
REFERENCE_DEVELOPMENT_YEARS = (2010, 2011, 2012, 2013, 2014, 2015)
REFERENCE_CALIBRATION_YEARS = (2016, 2017)
FIT_LABEL_CUTOFF = pd.Timestamp("2020-03-01")
PRE_HOLDOUT_YEARS = (2018, 2019, 2020, 2021, 2022)
HOLDOUT_YEAR = 2023
_IMPLEMENTATION_FILES = (
    "src/api.py",
    "src/data_access.py",
    "src/integrity.py",
    "src/metrics.py",
    "src/pd_model.py",
    "src/product.py",
    "scripts/train_model.py",
    "scripts/serve_model.py",
)
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
    "calibration_metrics",
    "risk_band_cutoffs",
    "feature_bins",
    "feature_distributions",
    "score_bins",
    "score_distribution",
    "data_sha256",
    "implementation_sha256",
}
_CALIBRATION_METRIC_KEYS = {
    "n", "events", "prevalence", "roc_auc", "pr_auc", "ks", "brier", "log_loss", "calibration_intercept", "calibration_slope"
}


def _implementation_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in _IMPLEMENTATION_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update((root / relative).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    return digest.hexdigest()


def _require_implementation(expected: str) -> None:
    if _implementation_sha256() != expected:
        raise RuntimeError("product implementation changed during this process")


def _input_schema(development: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    schema: dict[str, dict[str, float | str]] = {}
    for feature in PRODUCT_FEATURES:
        values = development[feature].dropna()
        if values.empty:
            raise ValueError(f"{feature} has no development values")
        schema[feature] = {"type": "number", "minimum": float(values.min()), "maximum": float(values.max())}
    return schema


def _references(development: pd.DataFrame, calibration_probability: np.ndarray) -> dict[str, object]:
    feature_bins = {feature: quantile_breaks(development[feature], bins=10).tolist() for feature in PRODUCT_FEATURES}
    return {
        "feature_bins": feature_bins,
        "feature_distributions": {
            feature: binned_distribution(development[feature], np.asarray(feature_bins[feature])).tolist()
            for feature in PRODUCT_FEATURES
        },
        "score_bins": quantile_breaks(calibration_probability, bins=10).tolist(),
    }


def _fit_frame(path: str | Path, chunksize: int) -> tuple[pd.DataFrame, str]:
    frame, data_sha256 = load_compact(
        path,
        EXPECTED_DATA_SHA256,
        years=REFERENCE_DEVELOPMENT_YEARS + REFERENCE_CALIBRATION_YEARS,
        chunksize=chunksize,
    )
    frame = frame.loc[frame["pd_label_available_date"] < FIT_LABEL_CUTOFF].copy()
    if frame.empty:
        raise ValueError("prepared file contains no rows before the fit cutoff")
    return frame, data_sha256


def train_product(path: str | Path, *, chunksize: int = 100_000) -> dict[str, object]:
    implementation_sha256 = _implementation_sha256()
    frame, data_sha256 = _fit_frame(path, chunksize)
    development = frame.loc[frame["cohort_year"].isin(REFERENCE_DEVELOPMENT_YEARS)]
    calibration = frame.loc[frame["cohort_year"].isin(REFERENCE_CALIBRATION_YEARS)]
    model = fit_calibrated_model(development, calibration)
    calibration_probability = model.predict_proba(calibration[list(PRODUCT_FEATURES)])[:, 1]
    cutoffs = risk_band_cutoffs(calibration_probability)
    references = _references(development, calibration_probability)
    bundle: dict[str, object] = {
        "bundle_version": BUNDLE_VERSION,
        "model_version": MODEL_VERSION,
        "model": model,
        "family": "logistic",
        "features": list(PRODUCT_FEATURES),
        "input_schema": _input_schema(development),
        "target": TARGET,
        "horizon_months": HORIZON_MONTHS,
        "event_definition": EVENT_DEFINITION,
        "periods": {
            "development_years": list(REFERENCE_DEVELOPMENT_YEARS),
            "calibration_years": list(REFERENCE_CALIBRATION_YEARS),
            "fit_label_cutoff": FIT_LABEL_CUTOFF.isoformat(),
        },
        "calibration_metrics": classification_metrics(calibration[TARGET], calibration_probability),
        "risk_band_cutoffs": [float(cutoffs[0]), float(cutoffs[1])],
        "feature_bins": references["feature_bins"],
        "feature_distributions": references["feature_distributions"],
        "score_bins": references["score_bins"],
        "score_distribution": binned_distribution(calibration_probability, np.asarray(references["score_bins"])).tolist(),
        "data_sha256": data_sha256,
        "implementation_sha256": implementation_sha256,
    }
    _require_implementation(implementation_sha256)
    return bundle


def save_bundle(bundle: dict[str, object], path: str | Path) -> None:
    _require_implementation(str(bundle.get("implementation_sha256", "")))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    try:
        joblib.dump(bundle, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _valid_distribution(value: object, size: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == size
        and all(type(item) in (int, float) and math.isfinite(item) and item >= 0 for item in value)
        and math.isclose(sum(value), 1.0)
    )


def load_bundle(path: str | Path) -> dict[str, object]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("model bundle must be a dictionary")
    if set(bundle) != REQUIRED_BUNDLE_KEYS:
        raise ValueError("model bundle is missing keys or contains unsupported content")
    if type(bundle["bundle_version"]) is not int or bundle["bundle_version"] != BUNDLE_VERSION:
        raise ValueError("unsupported model bundle version")
    if bundle["model_version"] != MODEL_VERSION or bundle["family"] != "logistic":
        raise ValueError("model bundle model identity is incompatible")
    if not callable(getattr(bundle["model"], "predict_proba", None)):
        raise ValueError("model bundle model is incompatible")
    if bundle["features"] != list(PRODUCT_FEATURES):
        raise ValueError("model bundle feature order does not match the product")
    schema = bundle["input_schema"]
    if not isinstance(schema, Mapping) or set(schema) != set(PRODUCT_FEATURES):
        raise ValueError("model bundle input_schema is incompatible")
    for feature in PRODUCT_FEATURES:
        limits = schema[feature]
        if (
            not isinstance(limits, Mapping)
            or set(limits) != {"type", "minimum", "maximum"}
            or limits["type"] != "number"
            or type(limits["minimum"]) not in (int, float)
            or type(limits["maximum"]) not in (int, float)
            or not math.isfinite(limits["minimum"])
            or not math.isfinite(limits["maximum"])
            or limits["minimum"] > limits["maximum"]
        ):
            raise ValueError("model bundle input_schema is incompatible")
    if bundle["target"] != TARGET or bundle["horizon_months"] != HORIZON_MONTHS or bundle["event_definition"] != EVENT_DEFINITION:
        raise ValueError("model bundle definition is incompatible")
    if bundle["periods"] != {
        "development_years": list(REFERENCE_DEVELOPMENT_YEARS),
        "calibration_years": list(REFERENCE_CALIBRATION_YEARS),
        "fit_label_cutoff": FIT_LABEL_CUTOFF.isoformat(),
    }:
        raise ValueError("model bundle periods are incompatible")
    cutoffs = bundle["risk_band_cutoffs"]
    if not (
        isinstance(cutoffs, list)
        and len(cutoffs) == 2
        and all(type(value) in (int, float) and math.isfinite(value) for value in cutoffs)
        and 0 <= cutoffs[0] <= cutoffs[1] <= 1
    ):
        raise ValueError("model bundle risk_band_cutoffs are incompatible")
    bins = bundle["feature_bins"]
    distributions = bundle["feature_distributions"]
    if not isinstance(bins, Mapping) or not isinstance(distributions, Mapping) or set(bins) != set(PRODUCT_FEATURES) or set(distributions) != set(PRODUCT_FEATURES):
        raise ValueError("model bundle feature PSI references are incompatible")
    for feature in PRODUCT_FEATURES:
        values = bins[feature]
        if not isinstance(values, list) or not values or not all(type(value) in (int, float) and math.isfinite(value) for value in values):
            raise ValueError("model bundle feature PSI references are incompatible")
        if len(values) > 1 and any(left >= right for left, right in zip(values, values[1:])):
            raise ValueError("model bundle feature PSI references are incompatible")
        if not _valid_distribution(distributions[feature], max(1, len(values) - 1) + 1):
            raise ValueError("model bundle feature PSI references are incompatible")
    score_bins = bundle["score_bins"]
    if not isinstance(score_bins, list) or not score_bins or not all(type(value) in (int, float) and math.isfinite(value) for value in score_bins):
        raise ValueError("model bundle score PSI references are incompatible")
    if len(score_bins) > 1 and any(left >= right for left, right in zip(score_bins, score_bins[1:])):
        raise ValueError("model bundle score PSI references are incompatible")
    if not _valid_distribution(bundle["score_distribution"], max(1, len(score_bins) - 1) + 1):
        raise ValueError("model bundle score PSI references are incompatible")
    metrics = bundle["calibration_metrics"]
    if (
        not isinstance(metrics, Mapping)
        or set(metrics) != _CALIBRATION_METRIC_KEYS
        or type(metrics["n"]) is not int
        or type(metrics["events"]) is not int
        or metrics["n"] < 1
        or not 0 <= metrics["events"] <= metrics["n"]
        or any(
            value is not None and (type(value) not in (int, float) or not math.isfinite(value))
            for name, value in metrics.items()
            if name not in {"n", "events"}
        )
    ):
        raise ValueError("model bundle calibration_metrics are incompatible")
    for name in ("data_sha256", "implementation_sha256"):
        if not isinstance(bundle[name], str) or re.fullmatch(r"[0-9a-f]{64}", bundle[name]) is None:
            raise ValueError(f"model bundle {name} is not a SHA-256")
    if bundle["data_sha256"] != EXPECTED_DATA_SHA256:
        raise ValueError("model bundle data_sha256 is incompatible")
    if bundle["implementation_sha256"] != _implementation_sha256():
        raise ValueError("model bundle implementation_sha256 is stale")
    return bundle


def _period_result(frame: pd.DataFrame, probability: np.ndarray, bundle: Mapping[str, object], keys: dict[str, int]) -> dict[str, object]:
    cutoffs = tuple(bundle["risk_band_cutoffs"])
    bands = risk_bands(probability, cutoffs)
    feature_psi = {
        feature: population_stability_index(
            bundle["feature_distributions"][feature],
            binned_distribution(frame[feature], np.asarray(bundle["feature_bins"][feature])),
        )
        for feature in PRODUCT_FEATURES
    }
    score_psi = population_stability_index(
        bundle["score_distribution"], binned_distribution(probability, np.asarray(bundle["score_bins"]))
    )
    return {
        **keys,
        "metrics": classification_metrics(frame[TARGET], probability),
        "bands": calibration_table(frame[TARGET], probability, bands),
        "psi": {"features": feature_psi, "score": score_psi},
    }


def evaluate_product(data_path: str | Path, bundle_path: str | Path, years: tuple[int, ...]) -> dict[str, object]:
    if years == (HOLDOUT_YEAR,):
        allowed = True
    else:
        allowed = bool(years) and all(type(year) is int and year in PRE_HOLDOUT_YEARS for year in years)
    if not allowed:
        raise ValueError("years must be 2018--2022 or exactly (2023,)")
    bundle = load_bundle(bundle_path)
    frame, observed = load_compact(data_path, EXPECTED_DATA_SHA256, years=years)
    probability = bundle["model"].predict_proba(frame[list(PRODUCT_FEATURES)])[:, 1]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("model returned invalid probabilities")
    annual = []
    for year in years:
        mask = frame["cohort_year"] == year
        annual.append(_period_result(frame.loc[mask], probability[mask], bundle, {"cohort_year": year}))
    if years == (HOLDOUT_YEAR,):
        quarters = []
        for quarter in range(1, 5):
            mask = frame["cohort_quarter"] == quarter
            if mask.any():
                quarters.append(_period_result(frame.loc[mask], probability[mask], bundle, {"cohort_year": HOLDOUT_YEAR, "cohort_quarter": quarter}))
        return {"years": [HOLDOUT_YEAR], "data_sha256": observed, "annual": annual[0], "quarters": quarters}
    return {
        "years": list(years),
        "data_sha256": observed,
        "annual": annual,
        "pooled": _period_result(frame, probability, bundle, {}),
    }
