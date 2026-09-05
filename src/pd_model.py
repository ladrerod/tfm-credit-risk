from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PRODUCT_FEATURES = (
    "origination_fico",
    "original_dti",
    "original_cltv",
    "original_interest_rate",
    "number_of_borrowers",
)
NUMERIC_FEATURES = PRODUCT_FEATURES
INTEGER_FEATURES = {
    "origination_fico",
    "number_of_borrowers",
}


def build_model(seed: int = 20260831) -> Pipeline:
    preprocessor = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), list(NUMERIC_FEATURES))],
        sparse_threshold=0.0,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=2,
                    learning_rate=0.1,
                    n_jobs=1,
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_model(development: pd.DataFrame, *, seed: int = 20260831) -> Pipeline:
    required = set(PRODUCT_FEATURES).union({"default_24m"})
    missing = sorted(required.difference(development.columns))
    if missing:
        raise ValueError(f"development is missing columns: {missing}")
    if set(development["default_24m"].unique()) != {0, 1}:
        raise ValueError("development must contain binary outcomes 0 and 1")
    model = build_model(seed)
    model.fit(
        development[list(PRODUCT_FEATURES)],
        development["default_24m"],
    )
    return model
