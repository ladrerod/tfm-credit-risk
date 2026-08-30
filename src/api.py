from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from .product import load_bundle, score_product


PUBLIC_MODEL_VERSION = "pd24-v1"
ACADEMIC_WARNING = "Academic risk estimate; not a credit decision."


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def create_app(bundle_path: str | Path) -> Flask:
    bundle = load_bundle(bundle_path)
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 4096

    def invalid_request():
        return jsonify(error="invalid request"), 400

    def internal_error():
        return jsonify(error="internal server error"), 500

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            model_version=PUBLIC_MODEL_VERSION,
            horizon_months=bundle["horizon_months"],
        )

    @app.post("/predict")
    def predict():
        if request.mimetype != "application/json":
            return invalid_request()
        try:
            payload = json.loads(request.get_data(cache=False), object_pairs_hook=_unique_json_object)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, RecursionError, RequestEntityTooLarge):
            return invalid_request()
        try:
            score, band = score_product(bundle, payload)
        except ValueError:
            return invalid_request()
        except Exception:
            return internal_error()
        return jsonify(
            risk_score=score,
            risk_band=band,
            model_version=PUBLIC_MODEL_VERSION,
            horizon_months=bundle["horizon_months"],
            warning=ACADEMIC_WARNING,
        )

    return app
