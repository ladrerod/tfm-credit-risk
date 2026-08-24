from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


def validate_frame(
    frame: pd.DataFrame,
    *,
    required: Sequence[str],
    ranges: Mapping[str, tuple[float, float]],
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    for key in ("record_key", "loan_key"):
        if key in frame and frame[key].duplicated().any():
            raise ValueError(f"duplicate {key} values")
    for name, (lower, upper) in ranges.items():
        values = pd.to_numeric(frame[name], errors="coerce").dropna()
        if not values.between(lower, upper, inclusive="both").all():
            raise ValueError(f"{name} contains values outside [{lower}, {upper}]")


def summarize_eda(frame: pd.DataFrame, *, target: str, cohort: str) -> dict[str, object]:
    events = pd.to_numeric(frame[target], errors="coerce")
    numeric = frame.select_dtypes(include="number")
    correlations = numeric.corr(method="spearman")
    pairs = []
    for left_index, left in enumerate(correlations.columns):
        for right in correlations.columns[left_index + 1 :]:
            value = correlations.loc[left, right]
            if pd.notna(value):
                pairs.append({"left": left, "right": right, "spearman": float(value)})
    cohorts = []
    for value, part in frame.groupby(cohort, dropna=False, sort=True):
        cohort_events = pd.to_numeric(part[target], errors="coerce")
        cohorts.append(
            {
                "cohort": int(value) if pd.notna(value) else None,
                "rows": int(len(part)),
                "events": int(cohort_events.fillna(0).sum()),
                "event_rate": float(cohort_events.mean()),
            }
        )
    return {
        "rows": int(len(frame)),
        "events": int(events.fillna(0).sum()),
        "event_rate": float(events.mean()),
        "missingness": {name: float(value) for name, value in frame.isna().mean().items()},
        "cohorts": cohorts,
        "correlations": pairs,
        "contains_row_data": False,
    }
