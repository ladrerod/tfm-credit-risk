from __future__ import annotations

import argparse
from collections.abc import Mapping

from src.product import load_bundle, score_product

ACADEMIC_WARNING = "Estimación académica; no es una decisión de crédito."
INVALID_INPUT = "Entrada inválida: revise los valores y sus rangos."
INTERNAL_ERROR = "No se pudo calcular la predicción."
BAND_LABELS = {"low": "Baja", "medium": "Media", "high": "Alta"}
LABELS = {
    "origination_fico": "FICO de originación",
    "original_dti": "DTI original (%)",
    "original_cltv": "CLTV original (%)",
    "original_interest_rate": "Tipo de interés original (%)",
    "number_of_borrowers": "Número de prestatarios",
}
DEFAULTS = {
    "origination_fico": 700,
    "original_dti": 30,
    "original_cltv": 80,
    "original_interest_rate": 4.5,
    "number_of_borrowers": 2,
}


def predict_loan(
    bundle: Mapping[str, object],
    *values: object,
) -> tuple[float | None, str, str]:
    features = bundle["features"]
    if not isinstance(features, list) or len(values) != len(features):
        return None, "", INVALID_INPUT
    payload = dict(zip(features, values))
    try:
        score, band = score_product(bundle, payload)
    except ValueError:
        return None, "", INVALID_INPUT
    except Exception:  # noqa: BLE001 - the local UI must not expose model internals
        return None, "", INTERNAL_ERROR
    return score, BAND_LABELS[band], ACADEMIC_WARNING


def create_demo(bundle_path: str):
    import gradio as gr

    bundle = load_bundle(bundle_path)
    schema = bundle["input_schema"]

    def component(feature: str):
        limits = schema[feature]
        return gr.Number(
            label=LABELS[feature],
            value=DEFAULTS[feature],
            minimum=limits["minimum"],
            maximum=limits["maximum"],
            precision=0 if limits["type"] == "integer" else None,
        )

    return gr.Interface(
        fn=lambda *values: predict_loan(bundle, *values),
        inputs=[component(feature) for feature in bundle["features"]],
        outputs=[
            gr.Number(label="Score de riesgo a 24 meses (0-1)", precision=6),
            gr.Textbox(label="Banda de riesgo"),
            gr.Textbox(label="Alcance"),
        ],
        title="PD24 - Riesgo de impago a 24 meses",
        description="Introduzca las variables de originación para obtener una estimación académica.",
        allow_flagging="never",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the local Gradio demo for the PD24 model."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    create_demo(args.model).launch(
        server_name="127.0.0.1", server_port=args.port, share=False
    )


if __name__ == "__main__":
    main()
