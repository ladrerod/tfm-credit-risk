from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .expected_loss import compose_expected_loss


@dataclass(frozen=True)
class Policy:
    name: str
    max_cltv: float
    max_dti: float
    max_pd: float


@dataclass(frozen=True)
class MacroShock:
    name: str
    unemployment_points: float
    hpi_points: float


def stress_probability(
    probability: pd.Series,
    shock: MacroShock,
    *,
    unemployment_log_odds: float = 0.20,
    hpi_log_odds: float = -0.02,
) -> pd.Series:
    values = probability.astype(float).clip(1e-9, 1 - 1e-9)
    log_odds = np.log(values / (1 - values))
    delta = unemployment_log_odds * shock.unemployment_points + hpi_log_odds * shock.hpi_points
    return pd.Series(1 / (1 + np.exp(-(log_odds + delta))), index=probability.index)


def evaluate_scenario(
    frame: pd.DataFrame,
    policy: Policy,
    shock: MacroShock,
) -> dict[str, object]:
    required = {"pd", "ead_ratio", "lgd", "original_upb", "cltv", "dti"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"scenario input is missing columns: {missing}")
    probability = stress_probability(frame["pd"], shock)
    retained = (
        (frame["cltv"] <= policy.max_cltv)
        & (frame["dti"] <= policy.max_dti)
        & (probability <= policy.max_pd)
    )
    exposure = frame["original_upb"].to_numpy(dtype=float) * frame["ead_ratio"].to_numpy(dtype=float)
    expected_loss = compose_expected_loss(
        probability.to_numpy(), exposure, frame["lgd"].to_numpy(dtype=float)
    )
    indices = retained.to_numpy()
    retained_exposure = float(exposure[indices].sum())
    retained_loss = float(expected_loss[indices].sum())
    return {
        "policy": policy.name,
        "macro_scenario": shock.name,
        "n": int(len(frame)),
        "retained": int(retained.sum()),
        "retention_rate": float(retained.mean()),
        "retained_exposure": retained_exposure,
        "mean_retained_pd": float(probability[retained].mean()) if retained.any() else None,
        "mean_retained_ead_ratio": float(frame.loc[retained, "ead_ratio"].mean()) if retained.any() else None,
        "mean_retained_lgd": float(frame.loc[retained, "lgd"].mean()) if retained.any() else None,
        "retained_expected_loss": retained_loss,
        "retained_expected_loss_rate": retained_loss / retained_exposure if retained_exposure else None,
        "is_forecast": False,
        "contains_row_data": False,
        "interpretation": "Illustrative portfolio-retention sensitivity; not market underwriting or a causal forecast.",
    }
