from __future__ import annotations

from pathlib import Path

import math

import pandas as pd
from flask import Flask, jsonify, request

from .product import load_bundle


def create_app(bundle_path: str | Path) -> Flask:
    bundle = load_bundle(bundle_path)
    app = Flask(__name__)
    features = bundle["features"]
    schema = bundle["input_schema"]

    def invalid_request():
        return jsonify(error="invalid request"), 400

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            model_version=bundle["model_version"],
            model_name=bundle["selected_model_name"],
            target=bundle["target"],
            horizon_months=bundle["horizon_months"],
            source=bundle["data_source"],
        )

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != set(features):
            return invalid_request()
        for name in features:
            value = payload[name]
            if type(value) not in (int, float) or not math.isfinite(value):
                return invalid_request()
            limits = schema[name]
            if value < limits["minimum"] or value > limits["maximum"]:
                return invalid_request()
        score = float(bundle["model"].predict_proba(pd.DataFrame([payload], columns=features))[0, 1])
        return jsonify(
            risk_score=score,
            risk_level="elevated" if score >= bundle["validation_threshold"] else "standard",
            model_version=bundle["model_version"],
            horizon_months=bundle["horizon_months"],
            warning="Experimental academic risk score; not a credit decision.",
        )

    return app
