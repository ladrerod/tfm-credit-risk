from __future__ import annotations

from itertools import combinations
from math import isfinite
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def audit_numeric_associations(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    strong_threshold: float = 0.8,
) -> dict[str, object]:
    if not features or not 0 < strong_threshold <= 1:
        raise ValueError("features and threshold are invalid")
    numeric = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    pairs = [
        {
            "feature_a": left,
            "feature_b": right,
            "pearson": float(pearson.loc[left, right]),
            "spearman": float(spearman.loc[left, right]),
            "pairwise_rows": int(numeric[[left, right]].dropna().shape[0]),
        }
        for left, right in combinations(features, 2)
    ]
    complete = numeric.fillna(numeric.median())
    standard = (complete - complete.mean()) / complete.std(ddof=0).replace(0, 1)
    condition = float(np.linalg.cond(standard.to_numpy()))
    return {
        "rows": int(len(frame)),
        "features": list(features),
        "coverage": {name: float(numeric[name].notna().mean()) for name in features},
        "pairs": pairs,
        "strong_threshold": strong_threshold,
        "strong_pairs": [
            row for row in pairs if max(abs(row["pearson"]), abs(row["spearman"])) >= strong_threshold
        ],
        "condition_number": condition if isfinite(condition) else None,
        "contains_row_data": False,
    }


def global_importance(
    model: object,
    frame: pd.DataFrame,
    target: pd.Series,
    features: Sequence[str],
    *,
    seed: int,
) -> list[dict[str, object]]:
    result = permutation_importance(
        model,
        frame[list(features)],
        target,
        scoring="neg_brier_score",
        n_repeats=3,
        random_state=seed,
    )
    rows = [
        {
            "feature": name,
            "importance": float(result.importances_mean[index]),
            "standard_deviation": float(result.importances_std[index]),
        }
        for index, name in enumerate(features)
    ]
    return sorted(rows, key=lambda row: abs(row["importance"]), reverse=True)


def representative_sensitivity(
    model: object,
    frame: pd.DataFrame,
    numeric_features: Sequence[str],
) -> list[dict[str, object]]:
    profile = {
        name: frame[name].median() if pd.api.types.is_numeric_dtype(frame[name]) else frame[name].mode().iloc[0]
        for name in frame.columns
    }
    rows = []
    for name in numeric_features:
        low, high = frame[name].quantile([0.25, 0.75])
        left = pd.DataFrame([{**profile, name: low}])
        right = pd.DataFrame([{**profile, name: high}])
        rows.append(
            {
                "feature": name,
                "p25": float(low),
                "p75": float(high),
                "probability_change": float(model.predict_proba(right)[:, 1][0] - model.predict_proba(left)[:, 1][0]),
            }
        )
    return sorted(rows, key=lambda row: abs(row["probability_change"]), reverse=True)
