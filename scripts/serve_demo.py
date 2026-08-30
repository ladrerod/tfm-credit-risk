from __future__ import annotations

import argparse
from collections.abc import Mapping

from src.product import load_bundle, score_product


ACADEMIC_WARNING = "Estimación académica; no es una decisión de crédito."
INVALID_INPUT = "Entrada inválida: revise los cinco valores y sus rangos."
INTERNAL_ERROR = "No se pudo calcular la predicción."
BAND_LABELS = {"low": "Baja", "medium": "Media", "high": "Alta"}


def predict_loan(
    bundle: Mapping[str, object],
    origination_fico: object,
    original_dti: object,
    original_cltv: object,
    original_interest_rate: object,
    number_of_borrowers: object,
) -> tuple[float | None, str, str]:
    payload = {
        "origination_fico": origination_fico,
        "original_dti": original_dti,
        "original_cltv": original_cltv,
        "original_interest_rate": original_interest_rate,
        "number_of_borrowers": number_of_borrowers,
    }
    try:
        score, band = score_product(bundle, payload)
    except ValueError:
        return None, "", INVALID_INPUT
    except Exception:
        return None, "", INTERNAL_ERROR
    return score * 100, BAND_LABELS[band], ACADEMIC_WARNING


def create_demo(bundle_path: str):
    import gradio as gr

    bundle = load_bundle(bundle_path)
    schema = bundle["input_schema"]

    def number(feature: str, label: str, value: float, step: float):
        limits = schema[feature]
        return gr.Number(
            label=label,
            value=value,
            minimum=limits["minimum"],
            maximum=limits["maximum"],
            step=step,
        )

    return gr.Interface(
        fn=lambda fico, dti, cltv, rate, borrowers: predict_loan(
            bundle, fico, dti, cltv, rate, borrowers
        ),
        inputs=[
            number("origination_fico", "FICO de originación", 700, 1),
            number("original_dti", "DTI original (%)", 30, 1),
            number("original_cltv", "CLTV original (%)", 80, 1),
            number("original_interest_rate", "Tipo de interés original (%)", 4.5, 0.01),
            number("number_of_borrowers", "Número de prestatarios", 2, 1),
        ],
        outputs=[
            gr.Number(label="PD a 24 meses (%)", precision=4),
            gr.Textbox(label="Banda de riesgo"),
            gr.Textbox(label="Alcance"),
        ],
        title="PD24 - Riesgo de impago a 24 meses",
        description="Introduzca las cinco variables de originación para obtener una estimación académica.",
        allow_flagging="never",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local Gradio demo for the PD24 model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    create_demo(args.model).launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
