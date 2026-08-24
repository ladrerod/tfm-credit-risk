from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIGURE_NAMES = (
    "architecture_walk_forward.png",
    "event_rate.png",
    "pd_diagnostics.png",
    "loss_components_error.png",
    "expected_loss_cohort.png",
    "drift.png",
)

BLUE = "#1F5A85"
BLUE_LIGHT = "#A9C4D8"
GOLD = "#B8860B"
INK = "#20252B"
MID = "#68727D"
GRID = "#D9E0E6"
PALE = "#F3F5F7"
WHITE = "#FFFFFF"

FEATURE_LABELS = {
    "original_interest_rate": "Tipo de interés",
    "original_upb": "Saldo original",
    "original_cltv": "CLTV original",
    "original_ltv": "LTV original",
    "mortgage_insurance_percentage": "Seguro hipotecario",
    "original_loan_term": "Plazo original",
    "original_dti": "DTI original",
    "origination_fico": "FICO de originación",
}


class EvidenceError(ValueError):
    pass


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"evidencia agregada ausente o inválida: {path}")
    return value


def _get(study: dict[str, Any], path: str) -> Any:
    value: Any = study
    for part in path.split("."):
        value = _mapping(value, path).get(part)
        if value is None:
            raise EvidenceError(f"evidencia agregada ausente: {path}")
    return value


def _rows(
    study: dict[str, Any], path: str, fields: tuple[str, ...], *, minimum: int = 1
) -> list[dict[str, Any]]:
    value = _get(study, path)
    if not isinstance(value, list) or len(value) < minimum:
        raise EvidenceError(f"evidencia agregada insuficiente: {path} requiere {minimum} filas")
    rows = []
    for index, row in enumerate(value):
        item = _mapping(row, f"{path}[{index}]")
        missing = [field for field in fields if item.get(field) is None]
        if missing:
            raise EvidenceError(
                f"evidencia agregada ausente: {path}[{index}] -> {', '.join(missing)}"
            )
        rows.append(item)
    return rows


def _number(value: object, path: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"evidencia numérica inválida: {path}") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise EvidenceError(f"evidencia numérica inválida: {path}")
    return result


def _validate(study: dict[str, Any], model_config: dict[str, Any]) -> None:
    identity = _mapping(_get(study, "identity"), "identity")
    source = identity.get("source")
    if source == "generated_in_memory":
        raise EvidenceError("generated_in_memory no es evidencia Freddie y no puede generar el TFM")
    if not source:
        raise EvidenceError("evidencia agregada ausente: identity.source")
    _number(identity.get("rows"), "identity.rows", minimum=1)
    _mapping(_get(study, "backtesting"), "backtesting")

    collections = (
        (
            "data_quality.cohorts",
            ("cohort", "rows", "events", "event_rate"),
            ("cohort", "rows", "events", "event_rate"),
            2,
        ),
        (
            "pd.test_cohorts",
            ("cohort_year", "n", "events", "roc_auc"),
            ("cohort_year", "n", "events", "roc_auc"),
            2,
        ),
        (
            "pd.test_calibration",
            ("bin", "count", "events", "mean_probability", "event_rate"),
            ("bin", "count", "events", "mean_probability", "event_rate"),
            2,
        ),
        (
            "expected_loss.cohorts",
            (
                "cohort_year",
                "n",
                "exposure_at_default",
                "total_expected_loss",
                "expected_loss_rate",
            ),
            (
                "cohort_year",
                "n",
                "exposure_at_default",
                "total_expected_loss",
                "expected_loss_rate",
            ),
            2,
        ),
        (
            "monitoring.feature_drift",
            ("feature", "psi", "jensen_shannon"),
            ("psi", "jensen_shannon"),
            1,
        ),
    )
    for path, fields, numeric_fields, minimum in collections:
        for index, row in enumerate(_rows(study, path, fields, minimum=minimum)):
            for field in numeric_fields:
                _number(row[field], f"{path}[{index}].{field}", minimum=0)

    test = _mapping(_get(study, "pd.test_metrics"), "pd.test_metrics")
    _number(test.get("n"), "pd.test_metrics.n", minimum=1)
    _number(test.get("events"), "pd.test_metrics.events", minimum=0)
    for component in ("ead", "lgd"):
        section = _mapping(
            _get(study, f"loss_components.{component}"), f"loss_components.{component}"
        )
        if not section.get("selected_name"):
            raise EvidenceError(f"evidencia agregada ausente: loss_components.{component}.selected_name")
        metrics = _mapping(
            section.get("test_metrics"), f"loss_components.{component}.test_metrics"
        )
        for field in ("n", "portfolio_relative_error", "wape"):
            _number(metrics.get(field), f"loss_components.{component}.test_metrics.{field}", minimum=0)

    backtesting = _mapping(model_config.get("backtesting"), "configs.model.backtesting")
    folds = backtesting.get("folds")
    if not isinstance(folds, list) or not folds:
        raise EvidenceError("evidencia agregada insuficiente: configs.model.backtesting.folds")
    required_fold = {
        "name",
        "as_of_date",
        "development_years",
        "calibration_year",
        "validation_year",
        "test_year",
    }
    for index, fold in enumerate(folds):
        if not isinstance(fold, dict) or not required_fold.issubset(fold):
            raise EvidenceError(f"configuración walk-forward incompleta: fold {index}")


def _format_int(value: object) -> str:
    return f"{int(value):,}".replace(",", ".")


def _format_percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%".replace(".", ",")


def _configure_matplotlib() -> tuple[Any, Any, Any, Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )
    return plt, FancyBboxPatch, Patch, Rectangle, FuncFormatter


def _save(plt: Any, figure: Any, path: Path, title: str, subtitle: str) -> None:
    figure.suptitle(title, x=0.055, y=0.98, ha="left", fontsize=15, fontweight="bold", color=INK)
    figure.text(0.055, 0.91, subtitle, ha="left", va="top", fontsize=9, color=MID)
    figure.tight_layout(rect=(0.025, 0.025, 0.985, 0.86))
    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "Matplotlib 3.10.5"},
    )
    plt.close(figure)


def _architecture(
    study: dict[str, Any], model_config: dict[str, Any], output: Path, plotting: tuple[Any, ...]
) -> None:
    plt, FancyBboxPatch, Patch, Rectangle, _ = plotting
    figure, (flow, matrix) = plt.subplots(
        2, 1, figsize=(9, 5.1), gridspec_kw={"height_ratios": (0.8, 1.7)}
    )
    flow.set_axis_off()
    nodes = (
        ("ZIP Freddie\nautorizados", BLUE),
        ("Compacto\nprivado", BLUE_LIGHT),
        ("PD · EAD · LGD", GOLD),
        ("Resultados\nagregados", BLUE_LIGHT),
        ("LaTeX · PDF", BLUE),
    )
    for index, (label, color) in enumerate(nodes):
        x = 0.01 + index * 0.2
        flow.add_patch(
            FancyBboxPatch(
                (x, 0.28),
                0.16,
                0.46,
                boxstyle="round,pad=0.012,rounding_size=0.015",
                facecolor=color,
                edgecolor=INK,
                linewidth=0.9,
            )
        )
        flow.text(
            x + 0.08,
            0.51,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color=WHITE if color in {BLUE, GOLD} else INK,
            fontweight="bold",
        )
        if index < len(nodes) - 1:
            flow.annotate(
                "",
                xy=(x + 0.195, 0.51),
                xytext=(x + 0.17, 0.51),
                arrowprops={"arrowstyle": "->", "color": MID, "lw": 1.2},
            )
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)

    folds = model_config["backtesting"]["folds"]
    years = sorted(
        {
            int(year)
            for fold in folds
            for year in (
                *fold["development_years"],
                fold["calibration_year"],
                fold["validation_year"],
                fold["test_year"],
            )
        }
    )
    styles = {
        "D": (BLUE, ""),
        "C": (GOLD, "//"),
        "V": (BLUE_LIGHT, ".."),
        "T": (PALE, "xx"),
    }
    for row_index, fold in enumerate(folds):
        roles = {int(year): "D" for year in fold["development_years"]}
        roles[int(fold["calibration_year"])] = "C"
        roles[int(fold["validation_year"])] = "V"
        roles[int(fold["test_year"])] = "T"
        for year, role in roles.items():
            color, hatch = styles[role]
            matrix.add_patch(
                Rectangle(
                    (year - 0.45, row_index - 0.36),
                    0.9,
                    0.72,
                    facecolor=color,
                    edgecolor=INK,
                    hatch=hatch,
                    linewidth=0.8,
                )
            )
            matrix.text(
                year,
                row_index,
                role,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=WHITE if role in {"D", "C"} else INK,
            )
    matrix.set_xlim(min(years) - 0.6, max(years) + 0.6)
    matrix.set_ylim(len(folds) - 0.5, -0.5)
    matrix.set_xticks(years)
    matrix.set_yticks(range(len(folds)), [str(fold["name"]) for fold in folds])
    matrix.set_xlabel("Año de cohorte")
    matrix.grid(False)
    matrix.legend(
        handles=[
            Patch(facecolor=color, edgecolor=INK, hatch=hatch, label=label)
            for label, (color, hatch) in zip(
                ("Desarrollo", "Calibración", "Validación", "Prueba"), styles.values(), strict=True
            )
        ],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
    )
    status = _mapping(study["backtesting"], "backtesting")
    freddie = "evaluado" if status.get("available") else "pendiente de regeneración temporal"
    _save(
        plt,
        figure,
        output / "architecture_walk_forward.png",
        "Arquitectura y backtesting walk-forward",
        f"Flujo sin filas públicas · Motor verificado con sintético · Freddie {freddie}",
    )


def _event_rate(study: dict[str, Any], output: Path, plotting: tuple[Any, ...]) -> None:
    plt, *_, FuncFormatter = plotting
    rows = sorted(study["data_quality"]["cohorts"], key=lambda row: int(row["cohort"]))
    years = [int(row["cohort"]) for row in rows]
    rates = [float(row["event_rate"]) for row in rows]
    figure, axis = plt.subplots(figsize=(9, 4.2))
    bars = axis.bar(years, rates, color=BLUE, edgecolor=INK, linewidth=0.6, width=0.72)
    axis.set_ylabel("Tasa de evento")
    axis.set_xlabel("Cohorte de originación")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_percent(value, 1)))
    axis.set_ylim(0, max(rates) * 1.34)
    axis.set_xticks(years)
    for bar, row in zip(bars, rows, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(rates) * 0.035,
            f"{_format_percent(float(row['event_rate']))}\n{_format_int(row['events'])}/{_format_int(row['rows'])}",
            ha="center",
            va="bottom",
            fontsize=7.7,
            color=INK,
        )
    events = sum(int(row["events"]) for row in rows)
    _save(
        plt,
        figure,
        output / "event_rate.png",
        "Evento de crédito a 24 meses por cohorte",
        f"Freddie Mac · n={_format_int(study['identity']['rows'])} elegibles · {_format_int(events)} eventos",
    )


def _pd_diagnostics(study: dict[str, Any], output: Path, plotting: tuple[Any, ...]) -> None:
    plt, *_, FuncFormatter = plotting
    cohorts = sorted(study["pd"]["test_cohorts"], key=lambda row: int(row["cohort_year"]))
    calibration = sorted(study["pd"]["test_calibration"], key=lambda row: int(row["bin"]))
    figure, (auc_axis, calibration_axis) = plt.subplots(
        1, 2, figsize=(9, 4.5), gridspec_kw={"width_ratios": (0.85, 1.55)}
    )

    years = [int(row["cohort_year"]) for row in cohorts]
    auc = [float(row["roc_auc"]) for row in cohorts]
    auc_axis.axhline(0.5, color=GOLD, linestyle="--", linewidth=1.2, label="Sin discriminación")
    auc_axis.scatter(years, auc, color=BLUE, edgecolor=INK, linewidth=0.6, s=70, zorder=3)
    auc_axis.set_ylim(0.5, max(0.85, max(auc) + 0.04))
    auc_axis.set_xticks(
        years,
        [
            f"{row['cohort_year']}\n{_format_int(row['events'])}/{_format_int(row['n'])} eventos"
            for row in cohorts
        ],
    )
    auc_axis.set_ylabel("AUC ROC")
    auc_axis.set_title("Discriminación por cohorte", loc="left", fontsize=10.5, fontweight="bold")
    for year, value in zip(years, auc, strict=True):
        auc_axis.text(year, value + 0.012, f"{value:.3f}".replace(".", ","), ha="center")
    auc_axis.legend(loc="lower left", frameon=False, fontsize=8)

    probabilities = [float(row["mean_probability"]) for row in calibration]
    observed = [float(row["event_rate"]) for row in calibration]
    limit = max(probabilities + observed) * 1.1
    calibration_axis.plot(
        [0, limit], [0, limit], color=GOLD, linestyle="--", linewidth=1.5, label="Calibración perfecta"
    )
    calibration_axis.plot(
        probabilities,
        observed,
        color=BLUE,
        marker="o",
        markerfacecolor=WHITE,
        markeredgewidth=1.4,
        linewidth=1.6,
        label="Modelo",
    )
    calibration_axis.set_xlim(0, limit)
    calibration_axis.set_ylim(0, limit)
    calibration_axis.set_xlabel("PD media")
    calibration_axis.set_ylabel("Evento observado")
    calibration_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_percent(value, 1)))
    calibration_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_percent(value, 1)))
    calibration_axis.set_title("Calibración por deciles", loc="left", fontsize=10.5, fontweight="bold")
    calibration_axis.legend(loc="upper left", frameon=False, fontsize=8)

    metrics = study["pd"]["test_metrics"]
    _save(
        plt,
        figure,
        output / "pd_diagnostics.png",
        "Diagnóstico fuera de tiempo de PD",
        f"Holdout 2021–2022 · n={_format_int(metrics['n'])} · {_format_int(metrics['events'])} eventos",
    )


def _loss_error(study: dict[str, Any], output: Path, plotting: tuple[Any, ...]) -> None:
    plt, *_, FuncFormatter = plotting
    components = [study["loss_components"][name] for name in ("ead", "lgd")]
    aggregate = [float(row["test_metrics"]["portfolio_relative_error"]) for row in components]
    wape = [float(row["test_metrics"]["wape"]) for row in components]
    labels = [
        f"{name.upper()}\nn={_format_int(row['test_metrics']['n'])}"
        for name, row in zip(("ead", "lgd"), components, strict=True)
    ]
    positions = (0, 1)
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.6, 4.2))
    left = axis.bar(
        [value - width / 2 for value in positions],
        aggregate,
        width,
        color=BLUE,
        edgecolor=INK,
        linewidth=0.7,
        label="Error agregado",
    )
    right = axis.bar(
        [value + width / 2 for value in positions],
        wape,
        width,
        color=GOLD,
        edgecolor=INK,
        linewidth=0.7,
        hatch="//",
        label="WAPE",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Error respecto al total observado")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_percent(value, 0)))
    axis.set_ylim(0, max(aggregate + wape) * 1.24)
    axis.legend(loc="upper left", frameon=False, ncol=2)
    for bars, values in ((left, aggregate), (right, wape)):
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(aggregate + wape) * 0.025,
                _format_percent(value, 1),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    model_labels = {"hist_gradient_boosting": "HGB", "hurdle": "hurdle"}
    selected = " · ".join(
        f"{name.upper()}: {model_labels.get(row['selected_name'], row['selected_name'])}"
        for name, row in zip(("ead", "lgd"), components, strict=True)
    )
    _save(
        plt,
        figure,
        output / "loss_components_error.png",
        "Errores fuera de tiempo de EAD y LGD",
        f"Modelos seleccionados · {selected} · El error agregado permite compensación de signos",
    )


def _expected_loss(study: dict[str, Any], output: Path, plotting: tuple[Any, ...]) -> None:
    plt, *_, FuncFormatter = plotting
    rows = sorted(study["expected_loss"]["cohorts"], key=lambda row: int(row["cohort_year"]))
    years = [int(row["cohort_year"]) for row in rows]
    losses = [float(row["total_expected_loss"]) / 1_000_000 for row in rows]
    rates = [float(row["expected_loss_rate"]) for row in rows]
    figure, (amount_axis, rate_axis) = plt.subplots(1, 2, figsize=(9, 4.25))
    bars = amount_axis.bar(years, losses, color=BLUE, edgecolor=INK, linewidth=0.7, width=0.62)
    amount_axis.set_title("Importe", loc="left", fontsize=10.5, fontweight="bold")
    amount_axis.set_ylabel("Millones de USD")
    amount_axis.set_xticks(years)
    amount_axis.set_ylim(0, max(losses) * 1.22)
    for bar, value in zip(bars, losses, strict=True):
        amount_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(losses) * 0.025,
            f"{value:.2f}".replace(".", ","),
            ha="center",
            va="bottom",
        )

    rate_axis.vlines(years, 0, rates, color=GOLD, linewidth=2)
    rate_axis.scatter(years, rates, s=65, color=GOLD, edgecolor=INK, linewidth=0.8, zorder=3)
    rate_axis.set_title("Tasa sobre EAD", loc="left", fontsize=10.5, fontweight="bold")
    rate_axis.set_ylabel("EL / EAD")
    rate_axis.set_xticks(years)
    rate_axis.set_ylim(0, max(rates) * 1.28)
    rate_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_percent(value, 2)))
    for year, value in zip(years, rates, strict=True):
        rate_axis.text(year, value + max(rates) * 0.04, _format_percent(value, 3), ha="center")
    total_n = sum(int(row["n"]) for row in rows)
    total_ead = sum(float(row["exposure_at_default"]) for row in rows) / 1_000_000
    _save(
        plt,
        figure,
        output / "expected_loss_cohort.png",
        "Pérdida esperada modelizada por cohorte",
        f"Holdout Freddie · n={_format_int(total_n)} · EAD={_format_int(round(total_ead))} MUSD · No es provisión ni capital",
    )


def _drift(
    study: dict[str, Any], model_config: dict[str, Any], monitoring_config: dict[str, Any], output: Path,
    plotting: tuple[Any, ...],
) -> None:
    plt, *_ = plotting
    rows = sorted(study["monitoring"]["feature_drift"], key=lambda row: float(row["psi"]))
    labels = [FEATURE_LABELS.get(str(row["feature"]), str(row["feature"]).replace("_", " ")) for row in rows]
    values = [float(row["psi"]) for row in rows]
    warning = float(monitoring_config["psi_warning"])
    critical = float(monitoring_config["psi_critical"])
    colors = [GOLD if value >= critical else BLUE for value in values]
    figure, axis = plt.subplots(figsize=(9, 4.6))
    bars = axis.barh(labels, values, color=colors, edgecolor=INK, linewidth=0.65)
    for bar, value in zip(bars, values, strict=True):
        if value >= critical:
            bar.set_hatch("//")
        axis.text(
            value + max(values) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}".replace(".", ","),
            va="center",
            ha="left",
            fontsize=8.5,
        )
    axis.axvline(warning, color=MID, linestyle="--", linewidth=1.2, label="Advertencia 0,10")
    axis.axvline(critical, color=GOLD, linestyle="-.", linewidth=1.2, label="Crítico 0,25")
    axis.set_xlim(0, max(values) * 1.12)
    axis.set_xlabel("Population Stability Index (PSI)")
    axis.legend(loc="lower right", frameon=False, ncol=2, fontsize=8)
    development = model_config["development_years"]
    test = model_config["test_years"]
    test_n = study["pd"]["test_metrics"]["n"]
    _save(
        plt,
        figure,
        output / "drift.png",
        "Deriva de variables entre desarrollo y prueba",
        f"Desarrollo {min(development)}–{max(development)} vs prueba {min(test)}–{max(test)} · n prueba={_format_int(test_n)}",
    )


def generate_figures(study: dict[str, Any], output: Path) -> None:
    model_config = json.loads((ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
    monitoring_config = json.loads((ROOT / "configs" / "monitoring.json").read_text(encoding="utf-8"))
    _validate(study, model_config)
    plotting = _configure_matplotlib()
    output.mkdir(parents=True, exist_ok=True)
    _architecture(study, model_config, output, plotting)
    _event_rate(study, output, plotting)
    _pd_diagnostics(study, output, plotting)
    _loss_error(study, output, plotting)
    _expected_loss(study, output, plotting)
    _drift(study, model_config, monitoring_config, output, plotting)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera las figuras estáticas del TFM.")
    parser.add_argument("--input", required=True, type=Path, help="JSON agregado Freddie explícito.")
    parser.add_argument("--output-dir", type=Path, default=Path("tfm/figures"))
    args = parser.parse_args()

    try:
        study = json.loads(args.input.read_text(encoding="utf-8"))
        generate_figures(_mapping(study, "root"), args.output_dir)
    except (OSError, json.JSONDecodeError, EvidenceError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
