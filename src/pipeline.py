from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .data_access import fetch_dataset, load_manifest, read_csv_zst, verify_file
from .data_quality import summarize_eda, validate_frame
from .expected_loss import summarize_expected_loss
from .governance import audit_numeric_associations, global_importance, representative_sensitivity
from .integrity import file_sha256, write_json_atomic
from .loss_models import regression_metrics, train_loss_models
from .metrics import calibration_table, classification_metrics, cohort_metrics
from .monitoring import build_alert, jensen_shannon, psi
from .pd_model import PDConfig, train_and_select
from .scenarios import MacroShock, Policy, evaluate_scenario


ROOT = Path(__file__).resolve().parents[1]
NUMERIC_FEATURES = (
    "original_interest_rate",
    "original_upb",
    "original_loan_term",
    "original_ltv",
    "original_cltv",
    "number_of_borrowers",
    "original_dti",
    "origination_fico",
    "mortgage_insurance_percentage",
)
CATEGORICAL_FEATURES = (
    "first_time_home_buyer",
    "loan_purpose",
    "property_type",
    "number_of_units",
    "occupancy_status",
    "property_state",
    "amortization_type",
    "mortgage_insurance_type",
    "high_balance_loan",
)
MACRO_FEATURES = ("unemployment_3m", "unemployment_change_12m", "hpi_yoy")


def _config(name: str) -> dict[str, object]:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def _synthetic_data(seed: int) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    rows = []
    unemployment_by_year = {2017: 4.4, 2018: 3.9, 2019: 3.7, 2020: 8.1, 2021: 5.4, 2022: 3.7}
    hpi_by_year = {2017: 0.06, 2018: 0.055, 2019: 0.04, 2020: 0.09, 2021: 0.16, 2022: 0.11}
    for year in range(2017, 2023):
        count = 1000
        fico = np.clip(generator.normal(720 - 3 * (year == 2022), 48, count), 500, 850)
        ltv = generator.uniform(40, 100, count)
        cltv = np.clip(ltv + generator.uniform(0, 6, count), 40, 110)
        dti = np.clip(generator.normal(35, 9, count), 5, 65)
        occupancy = generator.choice(["P", "I", "S"], count, p=[0.83, 0.12, 0.05])
        unemployment = unemployment_by_year[year] + generator.normal(0, 0.3, count)
        hpi = hpi_by_year[year] + generator.normal(0, 0.015, count)
        log_odds = (
            -3.4
            - 0.016 * (fico - 700)
            + 0.026 * (cltv - 80)
            + 0.020 * (dti - 35)
            + 0.55 * (occupancy == "I")
            + 0.08 * (unemployment - 4)
        )
        probability = 1 / (1 + np.exp(-log_odds))
        default = generator.binomial(1, probability)
        positive_loss_probability = 1 / (1 + np.exp(-(-0.5 + 0.035 * (cltv - 80))))
        positive_loss = generator.binomial(1, positive_loss_probability)
        lgd = positive_loss * np.clip(
            0.28 + 0.007 * (cltv - 80) - 0.5 * hpi + generator.normal(0, 0.10, count),
            0.03,
            1.5,
        )
        months = generator.integers(1, 13, count)
        for index in range(count):
            origin = pd.Timestamp(year, int(months[index]), 1)
            rows.append(
                {
                    "origination_date": origin,
                    "performance_end_date": origin + pd.DateOffset(months=24),
                    "cohort_year": year,
                    "original_interest_rate": float(generator.normal(3.8 + 0.15 * (year - 2017), 0.45)),
                    "original_upb": float(np.clip(generator.lognormal(12.25, 0.45), 50000, 1500000)),
                    "original_loan_term": int(generator.choice([180, 240, 360], p=[0.12, 0.08, 0.80])),
                    "original_ltv": float(ltv[index]),
                    "original_cltv": float(cltv[index]),
                    "number_of_borrowers": int(generator.choice([1, 2, 3], p=[0.48, 0.50, 0.02])),
                    "original_dti": float(dti[index]),
                    "origination_fico": float(fico[index]),
                    "mortgage_insurance_percentage": float(max(0, (ltv[index] - 80) * 1.2)),
                    "first_time_home_buyer": str(generator.choice(["Y", "N"], p=[0.35, 0.65])),
                    "loan_purpose": str(generator.choice(["P", "R", "C"], p=[0.55, 0.32, 0.13])),
                    "property_type": str(generator.choice(["SF", "CO", "PU"], p=[0.76, 0.15, 0.09])),
                    "number_of_units": str(generator.choice(["1", "2-4"], p=[0.96, 0.04])),
                    "occupancy_status": str(occupancy[index]),
                    "property_state": str(generator.choice(["CA", "TX", "FL", "NY", "IL"])),
                    "amortization_type": "FRM",
                    "mortgage_insurance_type": str(generator.choice(["0", "1"], p=[0.72, 0.28])),
                    "high_balance_loan": str(generator.choice(["Y", "N"], p=[0.08, 0.92])),
                    "unemployment_3m": float(unemployment[index]),
                    "unemployment_change_12m": float(unemployment[index] - unemployment_by_year.get(year - 1, 4.6)),
                    "hpi_yoy": float(hpi[index]),
                    "default_24m": int(default[index]),
                    "ead_ratio": float(np.clip(1 + generator.normal(0, 0.02), 0.85, 1.20)),
                    "lgd": float(lgd[index]),
                }
            )
    return pd.DataFrame(rows)


def _private_data(data_config: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    local_value = os.environ.get(str(data_config["dataset_directory_env"]), "")
    repo_id = os.environ.get(str(data_config["dataset_repository_env"]), "")
    if local_value:
        local = Path(local_value)
        manifest_path = local / str(data_config["manifest"])
        manifest = load_manifest(manifest_path)
        paths = [local / item["name"] for item in manifest["files"]]
        for item, path in zip(manifest["files"], paths):
            verify_file(path, item)
        source = "local_private_dataset"
    else:
        local = ROOT / str(data_config["cache_directory"])
        manifest, paths = fetch_dataset(
            repo_id,
            local,
            revision=str(data_config["dataset_revision"]),
            manifest_name=str(data_config["manifest"]),
        )
        manifest_path = local / str(data_config["manifest"])
        source = "private_dataset"
    frames = []
    for item, path in zip(manifest["files"], paths):
        for chunk in read_csv_zst(
            path,
            columns=item["columns"],
            chunksize=int(data_config["chunk_rows"]),
            dtypes=item.get("dtypes"),
        ):
            if "analysis_sample" not in chunk:
                raise ValueError("private data must declare the deterministic analysis sample")
            eligible = chunk["analysis_sample"].astype(bool)
            if {"population_eligible", "eligible"}.issubset(chunk):
                eligible &= chunk["population_eligible"].astype(bool) & chunk["eligible"].astype(bool)
            selected = chunk.loc[eligible].drop(columns="analysis_sample")
            if len(selected):
                frames.append(selected)
    if not frames:
        raise ValueError("private data contains no analysis sample rows")
    return pd.concat(frames, ignore_index=True), {
        "source": source,
        "repository": repo_id,
        "revision": str(data_config["dataset_revision"]),
        "files": len(paths),
        "manifest_sha256": file_sha256(manifest_path),
        "population_rows": int(manifest.get("population_rows", sum(item["rows"] for item in manifest["files"]))),
        "eligible_rows": int(manifest.get("eligible_rows", 0)),
    }


def _drift(reference: pd.DataFrame, current: pd.DataFrame, features: tuple[str, ...]) -> list[dict[str, object]]:
    rows = []
    for name in features:
        left = pd.to_numeric(reference[name], errors="coerce").dropna().to_numpy()
        right = pd.to_numeric(current[name], errors="coerce").dropna().to_numpy()
        edges = np.unique(np.quantile(left, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        expected, _ = np.histogram(left, bins=edges)
        observed, _ = np.histogram(right, bins=edges)
        rows.append(
            {
                "feature": name,
                "psi": psi(expected, observed),
                "jensen_shannon": jensen_shannon(expected, observed),
            }
        )
    return sorted(rows, key=lambda row: row["psi"], reverse=True)


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    runtime_directories = {".git", ".private", ".venv", "__pycache__", "data", "models", "outputs", "results"}
    for path in sorted(
        path for path in ROOT.rglob("*") if path.is_file() and path.suffix in {".py", ".json", ".lock", ".yml"}
    ):
        if runtime_directories.intersection(path.relative_to(ROOT).parts):
            continue
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _public_loss_result(result: dict[str, object], test: pd.DataFrame, features: list[str]) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    ead_model = result["ead"]["selected_model"]
    lgd_model = result["lgd"]["selected_model"]
    ead_prediction = np.clip(ead_model.predict(test[features]), 0.0, 1.5)
    lgd_prediction = np.clip(lgd_model.predict(test[features]), 0.0, 2.0)
    ead_test_metrics = regression_metrics(test["ead_ratio"], ead_prediction)
    lgd_test_metrics = regression_metrics(test["lgd"], lgd_prediction)
    decision_grade = bool(
        result["decision_grade"]
        and ead_test_metrics["portfolio_relative_error"] <= 0.15
        and lgd_test_metrics["portfolio_relative_error"] <= 0.50
    )
    payload = {
        "decision_grade": decision_grade,
        "ead": {
            "selected_name": result["ead"]["selected_name"],
            "validation_metrics": result["ead"]["metrics"],
            "test_metrics": ead_test_metrics,
        },
        "lgd": {
            "selected_name": result["lgd"]["selected_name"],
            "validation_metrics": result["lgd"]["metrics"],
            "test_metrics": lgd_test_metrics,
        },
    }
    return payload, ead_prediction, lgd_prediction


def run_study(mode: str = "synthetic", *, output_path: str | Path = "outputs/study-results.json") -> dict[str, object]:
    if mode not in {"synthetic", "full"}:
        raise ValueError("mode must be synthetic or full")
    data_config = _config("data.json")
    model_config = _config("model.json")
    scenario_config = _config("scenarios.json")
    monitoring_config = _config("monitoring.json")
    seed = int(model_config["seed"])
    if mode == "synthetic":
        frame = _synthetic_data(seed)
        identity = {"source": "generated_in_memory", "rows": int(len(frame)), "seed": seed}
    else:
        frame, identity = _private_data(data_config)
    frame["origination_date"] = pd.to_datetime(frame["origination_date"])
    if "performance_end_date" in frame:
        frame["performance_end_date"] = pd.to_datetime(frame["performance_end_date"])
    validate_frame(
        frame,
        required=(*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, *MACRO_FEATURES, "default_24m", "ead_ratio", "lgd"),
        ranges={
            "origination_fico": (300, 850),
            "original_ltv": (0, 200),
            "original_cltv": (0, 250),
            "original_dti": (0, 100),
            "ead_ratio": (0, 1.5),
        },
    )
    raw_lgd = pd.to_numeric(frame["lgd"], errors="coerce")
    lgd_tail = {
        "observed_rows": int(raw_lgd.notna().sum()),
        "below_zero": int((raw_lgd < 0).sum()),
        "above_two": int((raw_lgd > 2).sum()),
        "minimum": float(raw_lgd.min()),
        "maximum": float(raw_lgd.max()),
    }
    frame["lgd"] = raw_lgd.clip(0.0, 2.0)
    development = frame.loc[frame["cohort_year"].isin(model_config["development_years"])].copy()
    calibration = frame.loc[frame["cohort_year"] == int(model_config["calibration_year"])].copy()
    validation = frame.loc[frame["cohort_year"] == int(model_config["validation_year"])].copy()
    test = frame.loc[frame["cohort_year"].isin(model_config["test_years"])].copy()
    pd_config = PDConfig(NUMERIC_FEATURES, CATEGORICAL_FEATURES, seed)
    fitted_pd = train_and_select(development, calibration, validation, pd_config)
    pd_model = fitted_pd["selected_model"]
    test_probability = pd_model.predict_proba(test[list(pd_config.features)])[:, 1]
    test_pd_metrics = classification_metrics(test["default_24m"], test_probability, fitted_pd["threshold"])
    macro_pd_config = PDConfig(NUMERIC_FEATURES + MACRO_FEATURES, CATEGORICAL_FEATURES, seed)
    macro_pd = train_and_select(development, calibration, validation, macro_pd_config)
    base_validation = fitted_pd["metrics"][fitted_pd["selected_name"]]
    macro_validation = macro_pd["metrics"][macro_pd["selected_name"]]
    macro_promoted = bool(
        macro_validation["brier"] < base_validation["brier"]
        and macro_validation["log_loss"] < base_validation["log_loss"]
        and (macro_validation["roc_auc"] or 0) >= (base_validation["roc_auc"] or 0)
    )
    loss_eligible = frame["default_24m"].eq(1) & frame["ead_ratio"].notna() & frame["lgd"].notna()
    if "lgd_eligible" in frame:
        loss_eligible &= frame["lgd_eligible"].astype(bool)
    loss_development = frame.loc[
        loss_eligible & frame["cohort_year"].isin([2015, 2016, 2017, 2018, 2019])
    ].copy()
    loss_validation = frame.loc[
        loss_eligible & frame["cohort_year"].eq(int(model_config["validation_year"]))
    ].copy()
    loss_test = frame.loc[
        loss_eligible & frame["cohort_year"].isin(model_config["test_years"])
    ].copy()
    loss_features = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    fitted_loss = train_loss_models(
        loss_development,
        loss_validation,
        numeric=list(NUMERIC_FEATURES),
        categorical=list(CATEGORICAL_FEATURES),
        seed=seed,
    )
    loss_payload, _, _ = _public_loss_result(fitted_loss, loss_test, loss_features)
    ead_all = np.clip(fitted_loss["ead"]["selected_model"].predict(test[loss_features]), 0.0, 1.5)
    lgd_all = np.clip(fitted_loss["lgd"]["selected_model"].predict(test[loss_features]), 0.0, 2.0)
    expected_loss = summarize_expected_loss(
        test,
        probability_default=test_probability,
        ead_ratio=ead_all,
        loss_given_default=lgd_all,
    )
    scenario_frame = pd.DataFrame(
        {
            "pd": test_probability,
            "ead_ratio": ead_all,
            "lgd": lgd_all,
            "original_upb": test["original_upb"].to_numpy(),
            "cltv": test["original_cltv"].to_numpy(),
            "dti": test["original_dti"].to_numpy(),
        }
    )
    scenarios = [
        evaluate_scenario(
            scenario_frame,
            Policy(row["name"], row["max_cltv"], row["max_dti"], row["max_pd"]),
            MacroShock(shock["name"], shock["unemployment_points"], shock["hpi_points"]),
        )
        for row in scenario_config["policies"]
        for shock in scenario_config["macro_scenarios"]
    ]
    drift = _drift(development, test, NUMERIC_FEATURES)
    alerts = [
        build_alert(
            row["feature"],
            row["psi"],
            warning=float(monitoring_config["psi_warning"]),
            critical=float(monitoring_config["psi_critical"]),
        )
        for row in drift
    ]
    importance_frame = validation.sample(n=min(25_000, len(validation)), random_state=seed)
    payload = {
        "version": 1,
        "contains_row_data": False,
        "identity": {
            **identity,
            "implementation_sha256": _implementation_sha256(),
            "runtime": {
                "python": platform.python_version(),
                "numpy": importlib.metadata.version("numpy"),
                "pandas": importlib.metadata.version("pandas"),
                "scikit_learn": importlib.metadata.version("scikit-learn"),
            },
        },
        "methodology": {
            "target_horizon_months": int(model_config["target_horizon_months"]),
            "event_definition": model_config["event_definition"],
            "development_years": sorted(int(value) for value in development["cohort_year"].unique()),
            "calibration_year": int(model_config["calibration_year"]),
            "validation_year": int(model_config["validation_year"]),
            "test_years": model_config["test_years"],
        },
        "data_quality": {
            **summarize_eda(frame, target="default_24m", cohort="cohort_year"),
            "lgd_observed_tail": lgd_tail,
        },
        "pd": {
            "selected_name": fitted_pd["selected_name"],
            "threshold": fitted_pd["threshold"],
            "validation_metrics": fitted_pd["metrics"],
            "test_metrics": test_pd_metrics,
            "test_calibration": calibration_table(test["default_24m"], test_probability),
            "test_cohorts": cohort_metrics(test["cohort_year"], test["default_24m"], test_probability),
            "macro_challenger": {
                "selected_name": macro_pd["selected_name"],
                "validation_metrics": macro_validation,
                "promoted": macro_promoted,
            },
        },
        "loss_components": loss_payload,
        "expected_loss": expected_loss,
        "scenarios": scenarios,
        "governance": {
            "associations": audit_numeric_associations(development, NUMERIC_FEATURES),
            "global_importance": global_importance(
                pd_model,
                importance_frame,
                importance_frame["default_24m"],
                pd_config.features,
                seed=seed,
            ),
            "representative_sensitivity": representative_sensitivity(
                pd_model,
                validation[list(pd_config.features)],
                NUMERIC_FEATURES,
            ),
        },
        "monitoring": {"feature_drift": drift, "alerts": alerts},
        "limitations": [
            "Academic portfolio study; not a regulatory capital, provisioning, pricing or credit-decision system.",
            "Macro scenarios are transparent sensitivities and not causal forecasts.",
            "External validation with authorized bank data remains necessary.",
        ],
    }
    write_json_atomic(output_path, payload)
    return payload
