from __future__ import annotations

import numpy as np
import pandas as pd


def compose_expected_loss(
    probability_default: object,
    exposure_at_default: object,
    loss_given_default: object,
) -> np.ndarray:
    probability = np.asarray(probability_default, dtype=float)
    exposure = np.asarray(exposure_at_default, dtype=float)
    severity = np.asarray(loss_given_default, dtype=float)
    if probability.ndim != 1 or exposure.ndim != 1 or severity.ndim != 1:
        raise ValueError("expected-loss components must be vectors")
    if not (len(probability) == len(exposure) == len(severity)) or len(probability) == 0:
        raise ValueError("expected-loss components must align")
    if not np.isfinite(probability).all() or not np.isfinite(exposure).all() or not np.isfinite(severity).all():
        raise ValueError("expected-loss components must be finite")
    if ((probability < 0) | (probability > 1)).any() or (exposure < 0).any() or (severity < 0).any():
        raise ValueError("expected-loss components are outside economic bounds")
    return probability * exposure * severity


def summarize_expected_loss(
    frame: pd.DataFrame,
    *,
    probability_default: object,
    ead_ratio: object,
    loss_given_default: object,
) -> dict[str, object]:
    if not {"original_upb", "cohort_year"}.issubset(frame):
        raise ValueError("expected-loss summary requires balance and cohort")
    probability = np.asarray(probability_default, dtype=float)
    ratio = np.asarray(ead_ratio, dtype=float)
    severity = np.asarray(loss_given_default, dtype=float)
    upb = frame["original_upb"].to_numpy(dtype=float)
    if not len(frame) == len(probability) == len(ratio) == len(severity):
        raise ValueError("expected-loss inputs must align")
    exposure = upb * ratio
    losses = compose_expected_loss(probability, exposure, severity)

    def aggregate(indices: np.ndarray) -> dict[str, object]:
        total_exposure = float(exposure[indices].sum())
        total_loss = float(losses[indices].sum())
        return {
            "n": int(len(indices)),
            "original_upb": float(upb[indices].sum()),
            "exposure_at_default": total_exposure,
            "mean_pd": float(probability[indices].mean()),
            "mean_ead_ratio": float(ratio[indices].mean()),
            "mean_lgd": float(severity[indices].mean()),
            "total_expected_loss": total_loss,
            "expected_loss_rate": float(total_loss / total_exposure) if total_exposure else None,
        }

    result = aggregate(np.arange(len(frame)))
    result["cohorts"] = [
        {"cohort_year": int(year), **aggregate(np.asarray(indices, dtype=int))}
        for year, indices in frame.groupby("cohort_year", sort=True).indices.items()
    ]
    result["contains_row_data"] = False
    return result
