from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve


def _probability(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=float)
    if result.ndim != 1 or len(result) == 0 or not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("probability must be a finite non-empty vector inside [0, 1]")
    return result


def classification_metrics(target: Iterable[int], probability: Iterable[float]) -> dict[str, int | float | None]:
    raw_target = np.asarray(list(target), dtype=float)
    probability_array = _probability(probability)
    if (
        raw_target.ndim != 1
        or len(raw_target) != len(probability_array)
        or not np.isfinite(raw_target).all()
        or not np.isin(raw_target, [0, 1]).all()
    ):
        raise ValueError("target must be a binary vector aligned with probability")
    target_array = raw_target.astype(int)
    if len(np.unique(target_array)) == 2:
        false_positive, true_positive, _ = roc_curve(target_array, probability_array)
        clipped = np.clip(probability_array, 1e-8, 1 - 1e-8)
        calibrator = LogisticRegression(C=1e6, solver="lbfgs").fit(
            np.log(clipped / (1 - clipped)).reshape(-1, 1), target_array
        )
        roc_auc = float(roc_auc_score(target_array, probability_array))
        pr_auc = float(average_precision_score(target_array, probability_array))
        ks = float(np.max(true_positive - false_positive))
        calibration_intercept = float(calibrator.intercept_[0])
        calibration_slope = float(calibrator.coef_[0, 0])
    else:
        roc_auc = pr_auc = ks = calibration_intercept = calibration_slope = None
    return {
        "n": int(len(target_array)),
        "events": int(target_array.sum()),
        "prevalence": float(target_array.mean()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ks": ks,
        "brier": float(brier_score_loss(target_array, probability_array)),
        "log_loss": float(log_loss(target_array, probability_array, labels=[0, 1])),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }


def calibration_table(
    target: Iterable[int], probability: Iterable[float], bands: Iterable[str]
) -> list[dict[str, object]]:
    target_array = np.asarray(list(target), dtype=float)
    probability_array = _probability(probability)
    band_array = np.asarray(list(bands), dtype=object)
    if (
        target_array.ndim != 1
        or len(target_array) != len(probability_array)
        or len(target_array) != len(band_array)
        or not np.isfinite(target_array).all()
        or not np.isin(target_array, [0, 1]).all()
        or not all(isinstance(band, str) for band in band_array)
    ):
        raise ValueError("calibration inputs are invalid")
    return [
        {
            "band": band,
            "count": int(mask.sum()),
            "events": int(target_array[mask].sum()),
            "mean_probability": float(probability_array[mask].mean()),
            "event_rate": float(target_array[mask].mean()),
        }
        for band in dict.fromkeys(band_array)
        if (mask := band_array == band).any()
    ]


def risk_band_cutoffs(calibration_probability: Iterable[float]) -> tuple[float, float]:
    probability_array = _probability(calibration_probability)
    return tuple(float(value) for value in np.quantile(probability_array, [0.5, 0.9]))


def risk_bands(probability: Iterable[float], cutoffs: tuple[float, float]) -> np.ndarray:
    probability_array = _probability(probability)
    if len(cutoffs) != 2 or not all(np.isfinite(cutoffs)) or not 0 <= cutoffs[0] <= cutoffs[1] <= 1:
        raise ValueError("risk band cutoffs are invalid")
    return np.where(
        probability_array < cutoffs[0], "low", np.where(probability_array < cutoffs[1], "medium", "high")
    )


def quantile_breaks(reference: Iterable[float], bins: int = 10) -> np.ndarray:
    values = np.asarray(list(reference), dtype=float)
    if values.ndim != 1 or not isinstance(bins, int) or bins < 1 or np.isinf(values).any():
        raise ValueError("reference values or bins are invalid")
    finite = values[~np.isnan(values)]
    if not len(finite):
        raise ValueError("reference must contain a non-missing value")
    return np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)))


def binned_distribution(values: Iterable[float], breaks: np.ndarray) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    boundaries = np.asarray(breaks, dtype=float)
    if (
        array.ndim != 1
        or not len(array)
        or boundaries.ndim != 1
        or not len(boundaries)
        or not np.isfinite(boundaries).all()
        or (np.diff(boundaries) <= 0).any()
    ):
        raise ValueError("values or breaks are invalid")
    if np.isinf(array).any():
        raise ValueError("values must not be infinite")
    missing = np.isnan(array)
    counts = np.bincount(
        np.digitize(array[~missing], boundaries[1:-1], right=True), minlength=max(1, len(boundaries) - 1)
    )
    return np.append(counts, missing.sum()).astype(float) / len(array)


def population_stability_index(reference_distribution: Iterable[float], current_distribution: Iterable[float]) -> float:
    reference = np.asarray(list(reference_distribution), dtype=float)
    current = np.asarray(list(current_distribution), dtype=float)
    if (
        reference.ndim != 1
        or not len(reference)
        or reference.shape != current.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(current).all()
        or (reference < 0).any()
        or (current < 0).any()
        or not np.isclose(reference.sum(), 1)
        or not np.isclose(current.sum(), 1)
    ):
        raise ValueError("PSI distributions are invalid")
    return float(np.sum((reference - current) * np.log(np.maximum(reference, 1e-6) / np.maximum(current, 1e-6))))


def summarize_year_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows or any(type(row.get("cohort_year")) is not int for row in rows):
        raise ValueError("year metric rows require integer cohort_year")
    summary: dict[str, object] = {}
    metrics = set().union(*(row.keys() for row in rows)) - {"cohort_year", "n", "events", "prevalence"}
    for metric in metrics:
        values = [(int(row["cohort_year"]), float(row[metric])) for row in rows if row.get(metric) is not None]
        if len(values) < 2 or not np.isfinite([value for _, value in values]).all():
            summary[metric] = {"macro_mean": None, "median": None, "iqr": None, "worst_year": None}
            continue
        years, scores = zip(*values, strict=True)
        worst = int(np.argmax(scores) if metric in {"brier", "log_loss"} else np.argmin(scores))
        summary[metric] = {
            "macro_mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "iqr": float(np.diff(np.quantile(scores, [0.25, 0.75]))[0]),
            "worst_year": years[worst],
        }
    return summary


def material_winner(
    challenger: list[dict[str, float]], baseline: list[dict[str, float]], *, require_calibration: bool = False
) -> bool:
    paired = []
    for new, old in zip(challenger, baseline, strict=True):
        auc_delta = new["roc_auc"] - old["roc_auc"]
        brier_delta = (new["brier"] - old["brier"]) / old["brier"]
        logloss_delta = (new["log_loss"] - old["log_loss"]) / old["log_loss"]
        win = auc_delta >= 0.01 and (brier_delta <= -0.05 or logloss_delta <= -0.05)
        loss = auc_delta <= -0.01 or brier_delta >= 0.05 or logloss_delta >= 0.05
        paired.append((win, loss))
    if require_calibration and sum(
        -0.25 <= row["calibration_intercept"] <= 0.25 and 0.8 <= row["calibration_slope"] <= 1.2
        for row in challenger
    ) < 2:
        return False
    return sum(win for win, _ in paired) >= 2 and not any(loss for _, loss in paired)
