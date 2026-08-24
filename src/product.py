from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 1) - 1)))

from .data_access import read_csv_zst
from .integrity import file_sha256
from .pd_model import PDConfig, train_and_select


BUNDLE_VERSION = 1
MODEL_VERSION = "five-variable-pd-1"
AUTHORIZED_DATA_SHA256 = "468a7a4e0b80b3dec1722b27a549f0c3a937ddc1cf2c99aa7f77ce51f4ae0e0e"
PRODUCT_FEATURES = (
    "origination_fico",
    "original_dti",
    "original_cltv",
    "original_interest_rate",
    "number_of_borrowers",
)
TARGET = "default_24m"
DEVELOPMENT_YEARS = (2015, 2016, 2017, 2018)
CALIBRATION_YEAR = 2019
VALIDATION_YEAR = 2020
TRAINING_YEARS = (*DEVELOPMENT_YEARS, CALIBRATION_YEAR, VALIDATION_YEAR)
HORIZON_MONTHS = 24
EVENT_DEFINITION = "first 90+ delinquency or credit zero-balance code 02, 03, 09 or 15 within 24 months"
REQUIRED_BUNDLE_KEYS = {
    "bundle_version",
    "model_version",
    "model",
    "selected_model_name",
    "features",
    "input_schema",
    "validation_threshold",
    "validation_metrics",
    "target",
    "horizon_months",
    "event_definition",
    "development_years",
    "calibration_year",
    "validation_year",
    "test_evaluated",
    "data_source",
    "data_sha256",
    "implementation_sha256",
}


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("pd_model.py"), Path(__file__).with_name("metrics.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _training_frame(path: str | Path, chunksize: int, expected_sha256: str) -> tuple[pd.DataFrame, str]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing prepared Freddie file: {source}")
    data_sha256 = file_sha256(source)
    if data_sha256 != expected_sha256:
        raise ValueError("prepared Freddie file SHA-256 does not match the authorized data")
    frames = []
    for chunk in read_csv_zst(source, chunksize=chunksize):
        if "cohort_year" not in chunk:
            raise ValueError("prepared file is missing columns: ['cohort_year']")
        selected = chunk.loc[chunk["cohort_year"].isin(TRAINING_YEARS)].copy()
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise ValueError("prepared file contains no 2015-2020 rows")
    frame = pd.concat(frames, ignore_index=True)
    missing = sorted(set((*PRODUCT_FEATURES, TARGET)).difference(frame.columns))
    if missing:
        raise ValueError(f"prepared file is missing columns: {missing}")
    frame["original_cltv"] = pd.to_numeric(frame["original_cltv"], errors="coerce").clip(upper=200)
    return frame, data_sha256


def _input_schema(frame: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    schema = {}
    for feature in PRODUCT_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"{feature} has no numeric training values")
        schema[feature] = {
            "type": "number",
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return schema


def train_product(
    path: str | Path,
    *,
    chunksize: int = 10_000,
    seed: int = 20260819,
    expected_sha256: str = AUTHORIZED_DATA_SHA256,
) -> dict[str, Any]:
    frame, data_sha256 = _training_frame(path, chunksize, expected_sha256)
    fitted = train_and_select(
        frame.loc[frame["cohort_year"].isin(DEVELOPMENT_YEARS)],
        frame.loc[frame["cohort_year"] == CALIBRATION_YEAR],
        frame.loc[frame["cohort_year"] == VALIDATION_YEAR],
        PDConfig(PRODUCT_FEATURES, (), seed, target=TARGET),
    )
    return {
        "bundle_version": BUNDLE_VERSION,
        "model_version": MODEL_VERSION,
        "model": fitted["selected_model"],
        "selected_model_name": fitted["selected_name"],
        "features": list(PRODUCT_FEATURES),
        "input_schema": _input_schema(frame),
        "validation_threshold": fitted["threshold"],
        "validation_metrics": fitted["metrics"],
        "target": TARGET,
        "horizon_months": HORIZON_MONTHS,
        "event_definition": EVENT_DEFINITION,
        "development_years": list(DEVELOPMENT_YEARS),
        "calibration_year": CALIBRATION_YEAR,
        "validation_year": VALIDATION_YEAR,
        "test_evaluated": False,
        "data_source": "prepared_freddie_dataset",
        "data_sha256": data_sha256,
        "implementation_sha256": _implementation_sha256(),
    }


def save_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    try:
        joblib.dump(bundle, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("model bundle must be a dictionary")
    missing = sorted(REQUIRED_BUNDLE_KEYS.difference(bundle))
    if missing:
        raise ValueError(f"model bundle is missing keys: {missing}")
    if type(bundle["bundle_version"]) is not int or bundle["bundle_version"] != BUNDLE_VERSION:
        raise ValueError("unsupported model bundle version")
    if bundle["features"] != list(PRODUCT_FEATURES):
        raise ValueError("model bundle feature order does not match the product")
    if bundle["test_evaluated"] is not False:
        raise ValueError("model bundle must have test_evaluated set to false")
    return bundle
