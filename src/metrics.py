from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def _probability(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) == 0 or not np.isfinite(result).all():
        raise ValueError("probability must be a finite non-empty vector")
    if ((result < 0) | (result > 1)).any():
        raise ValueError("probability must remain inside [0, 1]")
    return result


def classification_metrics(
    target: Iterable[int], probability: Iterable[float], threshold: float
) -> dict[str, object]:
    y = np.asarray(target, dtype=int)
    p = _probability(probability)
    if len(y) != len(p) or not 0 < threshold < 1:
        raise ValueError("classification inputs are invalid")
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    if both:
        false_positive, true_positive, _ = roc_curve(y, p)
        auc = float(roc_auc_score(y, p))
        pr_auc = float(average_precision_score(y, p))
        ks = float(np.max(true_positive - false_positive))
        clipped = np.clip(p, 1e-8, 1 - 1e-8)
        calibrator = LogisticRegression(C=1e6, solver="lbfgs").fit(
            np.log(clipped / (1 - clipped)).reshape(-1, 1), y
        )
        calibration_intercept = float(calibrator.intercept_[0])
        calibration_slope = float(calibrator.coef_[0, 0])
    else:
        auc = pr_auc = ks = calibration_intercept = calibration_slope = None
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "gini": None if auc is None else float(2 * auc - 1),
        "ks": ks,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "threshold": float(threshold),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def calibration_table(
    target: Iterable[int], probability: Iterable[float], bins: int = 10
) -> list[dict[str, object]]:
    y = np.asarray(target, dtype=int)
    p = _probability(probability)
    if len(y) != len(p) or bins < 1:
        raise ValueError("calibration inputs are invalid")
    ordered = np.argsort(p, kind="stable")
    return [
        {
            "bin": number,
            "count": int(len(indices)),
            "events": int(y[indices].sum()),
            "mean_probability": float(p[indices].mean()),
            "event_rate": float(y[indices].mean()),
        }
        for number, indices in enumerate(np.array_split(ordered, min(bins, len(y))), start=1)
    ]


def cohort_metrics(
    cohort: Iterable[int], target: Iterable[int], probability: Iterable[float]
) -> list[dict[str, object]]:
    years = np.asarray(cohort)
    y = np.asarray(target)
    p = _probability(probability)
    if not len(years) == len(y) == len(p):
        raise ValueError("cohort inputs must align")
    return [
        {"cohort_year": int(year), **classification_metrics(y[years == year], p[years == year], 0.5)}
        for year in sorted(set(years))
    ]
