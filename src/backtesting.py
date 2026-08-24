from __future__ import annotations

import numpy as np
import pandas as pd

from .loss_models import regression_metrics, train_ead_models, train_lgd_models
from .metrics import classification_metrics
from .pd_model import PDConfig, train_and_select


TEMPORAL_COLUMNS = {
    "source_cutoff_date",
    "pd_label_available_date",
    "ead_label_available_date",
    "lgd_label_available_date",
}


def _validate_fold(fold: dict[str, object]) -> tuple[list[int], int, int, int, pd.Timestamp]:
    required = {
        "name",
        "as_of_date",
        "development_years",
        "calibration_year",
        "validation_year",
        "test_year",
    }
    missing = sorted(required.difference(fold))
    if missing:
        raise ValueError(f"backtesting fold is missing fields: {missing}")
    development = [int(year) for year in fold["development_years"]]
    calibration = int(fold["calibration_year"])
    validation = int(fold["validation_year"])
    test = int(fold["test_year"])
    if (
        not development
        or len(development) != len(set(development))
        or sorted(development) != development
        or len(set(development + [calibration, validation, test])) != len(development) + 3
        or max(development) >= calibration
        or calibration >= validation
        or validation >= test
    ):
        raise ValueError("development, calibration and validation years must be disjoint and before test")
    as_of = pd.to_datetime(fold["as_of_date"], errors="coerce")
    if pd.isna(as_of):
        raise ValueError("backtesting as_of_date is invalid")
    if as_of > pd.Timestamp(test, 1, 1):
        raise ValueError("backtesting as_of_date cannot follow test start")
    return development, calibration, validation, test, as_of


def _available(
    frame: pd.DataFrame,
    years: list[int],
    date_column: str,
    boundary: pd.Timestamp,
) -> pd.DataFrame:
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    return frame.loc[
        frame["cohort_year"].isin(years) & dates.notna() & dates.le(boundary)
    ].copy()


def _mature_test(frame: pd.DataFrame, year: int, date_column: str) -> pd.DataFrame:
    available = pd.to_datetime(frame[date_column], errors="coerce")
    cutoff = pd.to_datetime(frame["source_cutoff_date"], errors="coerce")
    return frame.loc[
        frame["cohort_year"].eq(year)
        & available.notna()
        & cutoff.notna()
        & available.le(cutoff)
    ].copy()


def _pd_component(
    frame: pd.DataFrame,
    *,
    development_years: list[int],
    calibration_year: int,
    validation_year: int,
    test_year: int,
    as_of: pd.Timestamp,
    config: PDConfig,
) -> tuple[dict[str, object], dict[str, object] | None, pd.DataFrame, np.ndarray | None]:
    cohorts = {
        "development": _available(
            frame, development_years, "pd_label_available_date", as_of
        ),
        "calibration": _available(
            frame, [calibration_year], "pd_label_available_date", as_of
        ),
        "validation": _available(
            frame, [validation_year], "pd_label_available_date", as_of
        ),
        "test": _mature_test(frame, test_year, "pd_label_available_date"),
    }
    summary: dict[str, object] = {
        f"{role}_{measure}": int(
            len(cohort) if measure == "rows" else cohort[config.target].sum()
        )
        for role, cohort in cohorts.items()
        for measure in ("rows", "events")
    }
    invalid = [
        role for role, cohort in cohorts.items() if cohort.empty or cohort[config.target].nunique() < 2
    ]
    if invalid:
        return (
            {
                "status": "insufficient_evidence",
                **summary,
                "reason": f"PD roles must contain both outcomes: {', '.join(invalid)}",
            },
            None,
            cohorts["test"],
            None,
        )
    fitted = train_and_select(
        cohorts["development"], cohorts["calibration"], cohorts["validation"], config
    )
    probabilities = fitted["selected_model"].predict_proba(
        cohorts["test"][list(config.features)]
    )[:, 1]
    return (
        {
            "status": "evaluated",
            **summary,
            "selected_name": fitted["selected_name"],
            "threshold": fitted["threshold"],
            "validation_metrics": fitted["metrics"],
            "test_metrics": classification_metrics(
                cohorts["test"][config.target], probabilities, fitted["threshold"]
            ),
        },
        fitted,
        cohorts["test"],
        probabilities,
    )


def _loss_component(
    frame: pd.DataFrame,
    *,
    target: str,
    date_column: str,
    development_years: list[int],
    calibration_year: int,
    validation_year: int,
    test_year: int,
    as_of: pd.Timestamp,
    minimum_rows: int,
    numeric: list[str],
    categorical: list[str],
    seed: int,
) -> tuple[dict[str, object], dict[str, object] | None, pd.DataFrame]:
    eligible = frame["default_24m"].eq(1) & pd.to_numeric(frame[target], errors="coerce").notna()
    if target == "lgd" and "lgd_eligible" in frame:
        eligible &= frame["lgd_eligible"].fillna(False).astype(bool)
    population = frame.loc[eligible].copy()
    cohorts = {
        "development": _available(
            population, development_years + [calibration_year], date_column, as_of
        ),
        "validation": _available(population, [validation_year], date_column, as_of),
        "test": _mature_test(population, test_year, date_column),
    }
    summary = {f"{role}_rows": int(len(cohort)) for role, cohort in cohorts.items()}
    thin = [role for role, cohort in cohorts.items() if len(cohort) < minimum_rows]
    if thin:
        return (
            {
                "status": "insufficient_evidence",
                **summary,
                "minimum_rows": int(minimum_rows),
                "reason": f"{target.upper()} roles below minimum: {', '.join(thin)}",
            },
            None,
            cohorts["test"],
        )
    trainer = train_ead_models if target == "ead_ratio" else train_lgd_models
    fitted = trainer(
        cohorts["development"],
        cohorts["validation"],
        numeric=numeric,
        categorical=categorical,
        seed=seed,
    )
    prediction = np.clip(
        fitted["selected_model"].predict(cohorts["test"][numeric + categorical]),
        0.0,
        1.5 if target == "ead_ratio" else 2.0,
    )
    return (
        {
            "status": "evaluated",
            **summary,
            "minimum_rows": int(minimum_rows),
            "selected_name": fitted["selected_name"],
            "validation_metrics": fitted["metrics"],
            "test_metrics": regression_metrics(cohorts["test"][target], prediction),
        },
        fitted,
        cohorts["test"],
    )


def _expected_loss(
    pd_result: dict[str, object],
    pd_fitted: dict[str, object] | None,
    pd_test: pd.DataFrame,
    pd_probability: np.ndarray | None,
    ead_result: dict[str, object],
    ead_fitted: dict[str, object] | None,
    lgd_result: dict[str, object],
    lgd_fitted: dict[str, object] | None,
    features: list[str],
) -> dict[str, object]:
    defaults = pd_test.loc[pd_test["default_24m"].eq(1)]
    required = {
        "original_upb",
        "ead_ratio",
        "lgd",
        "ead_label_available_date",
        "lgd_label_available_date",
        "source_cutoff_date",
    }
    complete = defaults.iloc[0:0]
    if required.issubset(defaults):
        cutoff = pd.to_datetime(defaults["source_cutoff_date"], errors="coerce")
        realized_upb = pd.to_numeric(defaults["original_upb"], errors="coerce")
        realized_ead = pd.to_numeric(defaults["ead_ratio"], errors="coerce")
        realized_lgd = pd.to_numeric(defaults["lgd"], errors="coerce")
        eligible = (
            defaults["lgd_eligible"].fillna(False).astype(bool)
            if "lgd_eligible" in defaults
            else pd.Series(True, index=defaults.index)
        )
        complete = defaults.loc[
            eligible
            & np.isfinite(realized_upb)
            & realized_upb.gt(0)
            & np.isfinite(realized_ead)
            & realized_ead.ge(0)
            & np.isfinite(realized_lgd)
            & realized_lgd.ge(0)
            & pd.to_datetime(defaults["ead_label_available_date"], errors="coerce").le(cutoff)
            & pd.to_datetime(defaults["lgd_label_available_date"], errors="coerce").le(cutoff)
        ]
    coverage = float(len(complete) / len(defaults)) if len(defaults) else 0.0
    availability = {
        "test_defaults": int(len(defaults)),
        "mature_realized_losses": int(len(complete)),
        "realized_loss_coverage": coverage,
    }
    if (
        pd_result["status"] != "evaluated"
        or ead_result["status"] != "evaluated"
        or lgd_result["status"] != "evaluated"
        or pd_fitted is None
        or ead_fitted is None
        or lgd_fitted is None
        or pd_probability is None
    ):
        return {
            "status": "unavailable",
            **availability,
            "reason": "PD, EAD and LGD must all be evaluable",
        }
    if not len(defaults) or len(complete) != len(defaults) or "original_upb" not in pd_test:
        return {
            "status": "unavailable",
            **availability,
            "reason": "realized losses require complete mature, eligible and economically valid defaults",
        }
    portfolio_upb = pd.to_numeric(pd_test["original_upb"], errors="coerce").to_numpy()
    if not np.isfinite(portfolio_upb).all() or (portfolio_upb < 0).any():
        return {
            "status": "unavailable",
            **availability,
            "reason": "portfolio original_upb must be numeric, finite and non-negative",
        }
    ead = np.clip(ead_fitted["selected_model"].predict(pd_test[features]), 0.0, 1.5)
    lgd = np.clip(lgd_fitted["selected_model"].predict(pd_test[features]), 0.0, 2.0)
    if not np.isfinite(ead).all() or not np.isfinite(lgd).all():
        return {
            "status": "unavailable",
            **availability,
            "reason": "predicted EAD and LGD must be finite",
        }
    predicted = float((pd_probability * portfolio_upb * ead * lgd).sum())
    realized = float((complete["original_upb"] * complete["ead_ratio"] * complete["lgd"]).sum())
    if not np.isfinite([predicted, realized]).all():
        return {
            "status": "unavailable",
            **availability,
            "reason": "expected and realized loss totals must be finite",
        }
    return {
        "status": "evaluated",
        **availability,
        "predicted_total": predicted,
        "realized_total": realized,
        "total_error": predicted - realized,
        "portfolio_relative_error": abs(predicted - realized) / realized if realized else None,
    }


def run_walk_forward(
    frame: pd.DataFrame,
    folds: list[dict[str, object]],
    pd_config: PDConfig,
    *,
    numeric: list[str],
    categorical: list[str],
    seed: int,
    minimum_ead_rows: int,
    minimum_lgd_rows: int,
) -> dict[str, object]:
    missing_temporal = sorted(TEMPORAL_COLUMNS.difference(frame.columns))
    if missing_temporal:
        raise ValueError(f"backtesting requires temporal columns: {missing_temporal}")
    required = set(pd_config.features).union(numeric, categorical).union(
        {"cohort_year", pd_config.target, "ead_ratio", "lgd"}
    )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"backtesting frame is missing columns: {missing}")
    if not folds or minimum_ead_rows < 1 or minimum_lgd_rows < 1:
        raise ValueError("backtesting folds and minimum rows must be positive")

    validated = [_validate_fold(fold) for fold in folds]
    names = [str(fold["name"]) for fold in folds]
    test_years = [roles[3] for roles in validated]
    if len(names) != len(set(names)) or test_years != sorted(set(test_years)):
        raise ValueError("backtesting fold names and test years must be unique and ordered")
    for previous, current in zip(validated, validated[1:]):
        previous_development, *_, previous_as_of = previous
        current_development, *_, current_as_of = current
        if current_as_of <= previous_as_of:
            raise ValueError("backtesting as_of_date must be strictly increasing across folds")
        if (
            len(current_development) <= len(previous_development)
            or current_development[: len(previous_development)] != previous_development
        ):
            raise ValueError(
                "backtesting development_years must contain the previous window as a strict prefix"
            )

    results = []
    for fold, roles in zip(folds, validated, strict=True):
        development, calibration, validation, test, as_of = roles
        pd_result, pd_fitted, pd_test, probability = _pd_component(
            frame,
            development_years=development,
            calibration_year=calibration,
            validation_year=validation,
            test_year=test,
            as_of=as_of,
            config=pd_config,
        )
        ead_result, ead_fitted, _ = _loss_component(
            frame,
            target="ead_ratio",
            date_column="ead_label_available_date",
            development_years=development,
            calibration_year=calibration,
            validation_year=validation,
            test_year=test,
            as_of=as_of,
            minimum_rows=minimum_ead_rows,
            numeric=numeric,
            categorical=categorical,
            seed=seed,
        )
        lgd_result, lgd_fitted, _ = _loss_component(
            frame,
            target="lgd",
            date_column="lgd_label_available_date",
            development_years=development,
            calibration_year=calibration,
            validation_year=validation,
            test_year=test,
            as_of=as_of,
            minimum_rows=minimum_lgd_rows,
            numeric=numeric,
            categorical=categorical,
            seed=seed,
        )
        results.append(
            {
                "name": str(fold["name"]),
                "as_of_date": as_of.date().isoformat(),
                "development_years": development,
                "calibration_year": calibration,
                "validation_year": validation,
                "test_year": test,
                "pd": pd_result,
                "ead": ead_result,
                "lgd": lgd_result,
                "expected_loss": _expected_loss(
                    pd_result,
                    pd_fitted,
                    pd_test,
                    probability,
                    ead_result,
                    ead_fitted,
                    lgd_result,
                    lgd_fitted,
                    numeric + categorical,
                ),
            }
        )
    return {
        "available": True,
        "contains_row_data": False,
        "method": "expanding-window walk-forward with as-of label maturity",
        "folds": results,
    }
