from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

from .product import load_bundle


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def create_app(bundle_path: str | Path) -> Flask:
    bundle = load_bundle(bundle_path)
    app = Flask(__name__)
    features = bundle["features"]
    schema = bundle["input_schema"]

    def invalid_request():
        return jsonify(error="invalid request"), 400

    def internal_error():
        return jsonify(error="internal server error"), 500

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
        if not request.is_json:
            return invalid_request()
        try:
            payload = json.loads(request.get_data(cache=True), object_pairs_hook=_unique_json_object)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return invalid_request()
        if not isinstance(payload, dict) or set(payload) != set(features):
            return invalid_request()
        for name in features:
            value = payload[name]
            if type(value) is float and not math.isfinite(value):
                return invalid_request()
            if type(value) not in (int, float):
                return invalid_request()
            limits = schema[name]
            if value < limits["minimum"] or value > limits["maximum"]:
                return invalid_request()
        try:
            score = float(bundle["model"].predict_proba(pd.DataFrame([payload], columns=features))[0, 1])
        except Exception:
            return internal_error()
        if not math.isfinite(score):
            return internal_error()
        return jsonify(
            risk_score=score,
            risk_level="elevated" if score >= bundle["validation_threshold"] else "standard",
            model_version=bundle["model_version"],
            horizon_months=bundle["horizon_months"],
            warning="Experimental academic risk score; not a credit decision.",
        )

    return app
