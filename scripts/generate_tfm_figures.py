from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIGURE_NAMES = (
    "architecture_walk_forward.png",
    "correlation_heatmap.png",
    "event_rate.png",
    "interpretability.png",
    "pd_diagnostics.png",
    "loss_components_error.png",
    "expected_loss_cohort.png",
    "drift.png",
    "scenario_tradeoff.png",
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
    "amortization_type": "Amortización",
    "first_time_home_buyer": "Primer comprador",
    "high_balance_loan": "Préstamo de saldo alto",
    "loan_purpose": "Finalidad",
    "original_interest_rate": "Tipo de interés",
    "original_upb": "Saldo original",
    "original_cltv": "CLTV original",
    "original_ltv": "LTV original",
    "mortgage_insurance_percentage": "Seguro hipotecario",
    "mortgage_insurance_type": "Tipo de seguro",
    "number_of_borrowers": "Número de prestatarios",
    "number_of_units": "Número de unidades",
    "occupancy_status": "Ocupación",
    "original_loan_term": "Plazo original",
    "original_dti": "DTI original",
    "origination_fico": "FICO de originación",
    "property_state": "Estado de la propiedad",
    "property_type": "Tipo de propiedad",
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


def _number(
    value: object,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"evidencia numérica inválida: {path}") from exc
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
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

    calibration = _rows(
        study,
        "pd.test_calibration",
        ("bin", "count", "events", "mean_probability", "event_rate"),
        minimum=2,
    )
    for index, row in enumerate(calibration):
        count = _number(row["count"], f"pd.test_calibration[{index}].count", minimum=1)
        _number(
            row["events"],
            f"pd.test_calibration[{index}].events",
            minimum=0,
            maximum=count,
        )
        for field in ("mean_probability", "event_rate"):
            _number(
                row[field],
                f"pd.test_calibration[{index}].{field}",
                minimum=0,
                maximum=1,
            )

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

    scenario_fields = (
        "policy",
        "macro_scenario",
        "n",
        "retained",
        "retention_rate",
        "retained_exposure",
        "retained_expected_loss",
        "retained_expected_loss_rate",
    )
    scenario_rows = _rows(study, "scenarios", scenario_fields, minimum=2)
    for index, row in enumerate(scenario_rows):
        for field in scenario_fields[2:]:
            _number(row[field], f"scenarios[{index}].{field}", minimum=0)
    expected_scenarios = {
        (policy, scenario)
        for policy in ("base", "conservative")
        for scenario in ("observed", "moderate_stress", "severe_stress")
    }
    scenario_keys = {
        (str(row["policy"]), str(row["macro_scenario"])) for row in scenario_rows
    }
    if scenario_keys != expected_scenarios or len(scenario_rows) != len(expected_scenarios):
        raise EvidenceError("evidencia agregada incompleta: scenarios")

    importance = _rows(
        study,
        "governance.global_importance",
        ("feature", "importance", "standard_deviation"),
    )
    for index, row in enumerate(importance):
        _number(row["importance"], f"governance.global_importance[{index}].importance")
        _number(
            row["standard_deviation"],
            f"governance.global_importance[{index}].standard_deviation",
            minimum=0,
        )

    sensitivity = _rows(
        study,
        "governance.representative_sensitivity",
        ("feature", "p25", "p75", "probability_change"),
    )
    for index, row in enumerate(sensitivity):
        _number(row["p25"], f"governance.representative_sensitivity[{index}].p25")
        _number(row["p75"], f"governance.representative_sensitivity[{index}].p75")
        _number(
            row["probability_change"],
            f"governance.representative_sensitivity[{index}].probability_change",
            minimum=-1,
            maximum=1,
        )

    associations = _mapping(_get(study, "governance.associations"), "governance.associations")
    features = associations.get("features")
    if (
        not isinstance(features, list)
        or len(features) < 2
        or any(not isinstance(feature, str) or not feature for feature in features)
        or len(features) != len(set(features))
    ):
        raise EvidenceError("evidencia agregada inválida: governance.associations.features")
    _number(associations.get("rows"), "governance.associations.rows", minimum=1)
    pairs = _rows(
        study,
        "governance.associations.pairs",
        ("feature_a", "feature_b", "pairwise_rows", "spearman"),
        minimum=len(features) * (len(features) - 1) // 2,
    )
    expected_pairs = {
        frozenset((feature_a, feature_b))
        for index, feature_a in enumerate(features)
        for feature_b in features[index + 1 :]
    }
    seen_pairs: set[frozenset[str]] = set()
    for index, row in enumerate(pairs):
        pair = frozenset((row["feature_a"], row["feature_b"]))
        if pair not in expected_pairs or pair in seen_pairs:
            raise EvidenceError(f"evidencia agregada inválida: governance.associations.pairs[{index}]")
        seen_pairs.add(pair)
        _number(
            row["pairwise_rows"],
            f"governance.associations.pairs[{index}].pairwise_rows",
            minimum=1,
        )
        _number(
            row["spearman"],
            f"governance.associations.pairs[{index}].spearman",
            minimum=-1,
            maximum=1,
        )
    if seen_pairs != expected_pairs:
        raise EvidenceError("evidencia agregada incompleta: governance.associations.pairs")

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
            "font.size": 10.5,
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


def _save(
    plt: Any,
    figure: Any,
    path: Path,
    title: str,
    subtitle: str,
    *,
    tight_layout: bool = True,
) -> None:
    figure.suptitle(title, x=0.055, y=0.98, ha="left", fontsize=15, fontweight="bold", color=INK)
    figure.text(0.055, 0.91, subtitle, ha="left", va="top", fontsize=9, color=MID)
    if tight_layout:
        figure.tight_layout(rect=(0.025, 0.025, 0.985, 0.86))
    figure.savefig(
        path,
        dpi=240,
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
    event_counts = [int(row["events"]) for row in rows]
    rates = [float(row["event_rate"]) for row in rows]
    figure, (count_axis, rate_axis) = plt.subplots(
        2,
        1,
        figsize=(9, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": (0.9, 1.25), "hspace": 0.12},
    )
    bars = count_axis.bar(
        years, event_counts, color=BLUE_LIGHT, edgecolor=INK, linewidth=0.6, width=0.68
    )
    count_axis.set_ylabel("Eventos")
    count_axis.set_ylim(0, max(event_counts) * 1.24)
    count_axis.set_title("Volumen de eventos", loc="left", fontsize=10.5, fontweight="bold")
    for bar, value in zip(bars, event_counts, strict=True):
        count_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(event_counts) * 0.025,
            _format_int(value),
            ha="center",
            va="bottom",
            fontsize=8.3,
            color=INK,
        )

    rate_axis.plot(
        years,
        rates,
        color=BLUE,
        marker="o",
        markerfacecolor=WHITE,
        markeredgewidth=1.4,
        linewidth=1.8,
    )
    total_rows = sum(int(row["rows"]) for row in rows)
    events = sum(event_counts)
    total_rate = events / total_rows
    rate_axis.axhline(
        total_rate,
        color=GOLD,
        linestyle="--",
        linewidth=1.4,
        label=f"Total: {_format_percent(total_rate)}",
    )
    rate_axis.set_ylabel("Tasa de evento")
    rate_axis.set_xlabel("Cohorte de originación")
    rate_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_percent(value, 1)))
    rate_axis.set_ylim(0, max(rates) * 1.22)
    rate_axis.set_xticks(years)
    rate_axis.set_title("Frecuencia a 24 meses", loc="left", fontsize=10.5, fontweight="bold")
    rate_axis.legend(loc="upper left", frameon=False, fontsize=8.5)
    for year, value in zip(years, rates, strict=True):
        rate_axis.text(
            year,
            value + max(rates) * 0.035,
            _format_percent(value),
            ha="center",
            va="bottom",
            fontsize=8.3,
        )
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.76, hspace=0.16)
    _save(
        plt,
        figure,
        output / "event_rate.png",
        "Evento de crédito a 24 meses por cohorte",
        f"Freddie Mac · n={_format_int(total_rows)} elegibles · {_format_int(events)} eventos",
        tight_layout=False,
    )


def _pd_diagnostics(study: dict[str, Any], output: Path, plotting: tuple[Any, ...]) -> None:
    plt, *_, FuncFormatter = plotting
    cohorts = sorted(study["pd"]["test_cohorts"], key=lambda row: int(row["cohort_year"]))
    calibration = sorted(study["pd"]["test_calibration"], key=lambda row: int(row["bin"]))
    probabilities = [float(row["mean_probability"]) for row in calibration]
    observed = [float(row["event_rate"]) for row in calibration]
    counts = [int(row["count"]) for row in calibration]
    events = [int(row["events"]) for row in calibration]
    bins = [int(row["bin"]) for row in calibration]
    figure, (calibration_axis, residual_axis) = plt.subplots(
        1, 2, figsize=(9, 4.7), gridspec_kw={"width_ratios": (1.15, 1)}
    )

    z = 1.959964
    lower: list[float] = []
    upper: list[float] = []
    for successes, count in zip(events, counts, strict=True):
        proportion = successes / count
        denominator = 1 + z * z / count
        centre = (proportion + z * z / (2 * count)) / denominator
        half_width = (
            z
            * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count))
            / denominator
        )
        lower.append(max(0, centre - half_width))
        upper.append(min(1, centre + half_width))

    limit = max(probabilities + upper) * 1.1
    calibration_axis.plot(
        [0, limit], [0, limit], color=GOLD, linestyle="--", linewidth=1.5, label="Calibración perfecta"
    )
    calibration_axis.errorbar(
        probabilities,
        observed,
        yerr=(
            [value - bound for value, bound in zip(observed, lower, strict=True)],
            [bound - value for value, bound in zip(observed, upper, strict=True)],
        ),
        color=BLUE,
        marker="o",
        markerfacecolor=WHITE,
        markeredgewidth=1.4,
        linewidth=1.6,
        capsize=2.5,
        label="Modelo",
    )
    total_count = sum(counts)
    aggregate_predicted = sum(
        probability * count for probability, count in zip(probabilities, counts, strict=True)
    ) / total_count
    aggregate_observed = sum(events) / total_count
    calibration_axis.scatter(
        aggregate_predicted,
        aggregate_observed,
        marker="D",
        s=62,
        color=GOLD,
        edgecolor=INK,
        linewidth=0.7,
        zorder=4,
        label="Agregado",
    )
    ratio = aggregate_predicted / aggregate_observed if aggregate_observed else math.inf
    calibration_axis.annotate(
        f"Agregado: {_format_percent(aggregate_predicted)} / {_format_percent(aggregate_observed)}\n"
        f"predicho/observado = {ratio:.2f}x".replace(".", ","),
        xy=(aggregate_predicted, aggregate_observed),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=8.2,
        color=INK,
    )
    calibration_axis.set_xlim(0, limit)
    calibration_axis.set_ylim(0, limit)
    calibration_axis.set_xlabel("PD media")
    calibration_axis.set_ylabel("Evento observado")
    calibration_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_percent(value, 1)))
    calibration_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_percent(value, 1)))
    calibration_axis.set_title("Nivel predicho frente a observado", loc="left", fontsize=10.5, fontweight="bold")
    calibration_axis.legend(loc="upper left", frameon=False, fontsize=8)

    residuals = [actual - predicted for actual, predicted in zip(observed, probabilities, strict=True)]
    bars = residual_axis.bar(
        bins,
        residuals,
        color=[GOLD if value < 0 else BLUE for value in residuals],
        edgecolor=INK,
        linewidth=0.55,
        width=0.72,
    )
    residual_axis.axhline(0, color=INK, linewidth=1)
    residual_axis.set_xticks(bins)
    residual_axis.set_xlabel("Decil de PD")
    residual_axis.set_ylabel("Observado - predicho")
    residual_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_percent(value, 1)))
    residual_axis.set_title("Error de calibración", loc="left", fontsize=10.5, fontweight="bold")
    residual_low = min(0.0, *residuals)
    residual_high = max(0.0, *residuals)
    residual_span = max(residual_high - residual_low, 0.001)
    residual_axis.set_ylim(
        residual_low - residual_span * 0.10,
        residual_high + residual_span * 0.10,
    )
    for bar, value in zip(bars, residuals, strict=True):
        residual_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value - residual_span * 0.025 if value < 0 else value + residual_span * 0.025,
            f"{value * 100:+.1f}".replace(".", ","),
            ha="center",
            va="top" if value < 0 else "bottom",
            fontsize=7.4,
        )

    metrics = study["pd"]["test_metrics"]
    cohort_auc = " · ".join(
        f"AUC {row['cohort_year']}={float(row['roc_auc']):.3f}".replace(".", ",")
        for row in cohorts
    )
    _save(
        plt,
        figure,
        output / "pd_diagnostics.png",
        "Diagnóstico fuera de tiempo de PD",
        f"Holdout 2021-2022 · n={_format_int(metrics['n'])} · {_format_int(metrics['events'])} eventos · {cohort_auc}",
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
    warning = float(monitoring_config["psi_warning"])
    critical = float(monitoring_config["psi_critical"])
    dominant = rows[-1]
    remaining = rows[:-1]
    figure, (dominant_axis, detail_axis) = plt.subplots(
        1, 2, figsize=(9, 5.25), gridspec_kw={"width_ratios": (0.82, 1.55)}
    )

    dominant_value = float(dominant["psi"])
    dominant_label = FEATURE_LABELS.get(
        str(dominant["feature"]), str(dominant["feature"]).replace("_", " ")
    )
    dominant_bar = dominant_axis.barh(
        [dominant_label], [dominant_value], color=GOLD, edgecolor=INK, linewidth=0.7, hatch="//"
    )[0]
    dominant_axis.axvline(warning, color=MID, linestyle="--", linewidth=1.1)
    dominant_axis.axvline(critical, color=GOLD, linestyle="-.", linewidth=1.1)
    dominant_axis.set_xlim(0, dominant_value * 1.08)
    dominant_axis.set_xlabel("PSI - escala completa")
    dominant_axis.set_title("Cambio dominante", loc="left", fontsize=10.5, fontweight="bold")
    dominant_axis.text(
        dominant_value * 0.97,
        dominant_bar.get_y() + dominant_bar.get_height() / 2,
        f"{dominant_value:.3f}".replace(".", ","),
        ha="right",
        va="center",
        color=WHITE,
        fontweight="bold",
    )

    labels = [
        FEATURE_LABELS.get(str(row["feature"]), str(row["feature"]).replace("_", " "))
        for row in remaining
    ]
    values = [float(row["psi"]) for row in remaining]
    colors = [GOLD if value >= critical else BLUE for value in values]
    bars = detail_axis.barh(labels, values, color=colors, edgecolor=INK, linewidth=0.65)
    detail_limit = max(critical * 1.1, max(values, default=critical) * 1.18)
    for bar, value in zip(bars, values, strict=True):
        if value >= critical:
            bar.set_hatch("//")
        detail_axis.text(
            value + detail_limit * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}".replace(".", ","),
            va="center",
            ha="left",
            fontsize=8.5,
        )
    detail_axis.axvline(warning, color=MID, linestyle="--", linewidth=1.2, label="Advertencia 0,10")
    detail_axis.axvline(critical, color=GOLD, linestyle="-.", linewidth=1.2, label="Crítico 0,25")
    detail_axis.set_xlim(0, detail_limit)
    detail_axis.set_xlabel("PSI - detalle hasta el umbral crítico")
    detail_axis.set_title("Resto de variables", loc="left", fontsize=10.5, fontweight="bold")
    detail_axis.legend(loc="lower right", frameon=False, fontsize=8)
    development = model_config["development_years"]
    test = model_config["test_years"]
    test_n = study["pd"]["test_metrics"]["n"]
    _save(
        plt,
        figure,
        output / "drift.png",
        "Deriva de variables entre desarrollo y prueba",
        f"Desarrollo {min(development)}-{max(development)} vs prueba {min(test)}-{max(test)} · "
        f"n prueba={_format_int(test_n)} · Escalas diferenciadas para conservar la lectura",
    )


def _scenario_tradeoff(study: dict[str, Any], output: Path, plotting: tuple[Any, ...]) -> None:
    plt, *_, FuncFormatter = plotting
    rows = study["scenarios"]
    scenario_order = {"observed": 0, "moderate_stress": 1, "severe_stress": 2}
    scenario_labels = {
        "observed": "Observado",
        "moderate_stress": "Moderado",
        "severe_stress": "Severo",
    }
    policy_styles = {
        "base": (BLUE, "o", "-", "Base"),
        "conservative": (GOLD, "s", "--", "Conservadora"),
    }
    figure, (tradeoff_axis, amount_axis) = plt.subplots(
        1, 2, figsize=(9, 4.8), gridspec_kw={"width_ratios": (1.35, 1)}
    )
    all_exposures = [float(row["retained_exposure"]) / 1_000_000 for row in rows]
    all_rates = [float(row["retained_expected_loss_rate"]) for row in rows]

    for policy, (color, marker, linestyle, label) in policy_styles.items():
        policy_rows = sorted(
            (row for row in rows if row["policy"] == policy),
            key=lambda row: scenario_order[str(row["macro_scenario"])],
        )
        exposures = [float(row["retained_exposure"]) / 1_000_000 for row in policy_rows]
        rates = [float(row["retained_expected_loss_rate"]) for row in policy_rows]
        tradeoff_axis.plot(
            exposures,
            rates,
            color=color,
            marker=marker,
            markerfacecolor=WHITE if policy == "conservative" else color,
            markeredgecolor=INK,
            markeredgewidth=0.8,
            linestyle=linestyle,
            linewidth=1.7,
            markersize=7,
            label=label,
        )
        for row, x_value, y_value in zip(policy_rows, exposures, rates, strict=True):
            tradeoff_axis.annotate(
                f"{scenario_labels[str(row['macro_scenario'])]}\n"
                f"{_format_percent(float(row['retention_rate']), 1)} préstamos · "
                f"{_format_percent(y_value, 3)}",
                (x_value, y_value),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=7.8,
                color=INK,
            )
    tradeoff_axis.set_xlabel("EAD retenida (MUSD)")
    tradeoff_axis.set_ylabel("EL / EAD retenida")
    tradeoff_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_int(round(value))))
    tradeoff_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_percent(value, 2)))
    exposure_span = max(all_exposures) - min(all_exposures)
    rate_span = max(all_rates) - min(all_rates)
    tradeoff_axis.set_xlim(
        max(0, min(all_exposures) - exposure_span * 0.10),
        max(all_exposures) + exposure_span * 0.10,
    )
    tradeoff_axis.set_ylim(
        max(0, min(all_rates) - rate_span * 0.12),
        max(all_rates) + rate_span * 0.12,
    )
    tradeoff_axis.set_title(
        "Riesgo relativo y EAD retenida", loc="left", fontsize=10.5, fontweight="bold"
    )
    tradeoff_axis.legend(loc="upper left", frameon=False, fontsize=8.5)

    positions = range(3)
    width = 0.34
    for policy_index, (policy, (color, _, _, label)) in enumerate(policy_styles.items()):
        by_scenario = {
            str(row["macro_scenario"]): row for row in rows if row["policy"] == policy
        }
        losses = [
            float(by_scenario[scenario]["retained_expected_loss"]) / 1_000_000
            for scenario in scenario_order
        ]
        bars = amount_axis.bar(
            [position + (policy_index - 0.5) * width for position in positions],
            losses,
            width,
            color=color if policy == "base" else WHITE,
            edgecolor=INK if policy == "base" else color,
            linewidth=1,
            hatch="//" if policy == "conservative" else None,
            label=label,
        )
        for bar, value in zip(bars, losses, strict=True):
            amount_axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(losses) * 0.025,
                f"{value:.1f}".replace(".", ","),
                ha="center",
                va="bottom",
                fontsize=7.8,
            )
    amount_axis.set_xticks(positions, ["Observado", "Moderado", "Severo"])
    amount_axis.set_ylabel("EL retenida (MUSD)")
    amount_axis.set_ylim(
        0,
        max(float(row["retained_expected_loss"]) for row in rows) / 1_000_000 * 1.18,
    )
    amount_axis.set_title("Pérdida absoluta", loc="left", fontsize=10.5, fontweight="bold")
    amount_axis.legend(loc="upper right", frameon=False, fontsize=8.5)

    n = max(int(row["n"]) for row in rows)
    _save(
        plt,
        figure,
        output / "scenario_tradeoff.png",
        "Intercambio entre riesgo y exposición retenida",
        f"Holdout Freddie · n={_format_int(n)} · Estados discretos de sensibilidad, no trayectoria temporal ni pronóstico",
    )


def _interpretability(study: dict[str, Any], output: Path, plotting: tuple[Any, ...]) -> None:
    plt, *_ = plotting
    importance_rows = sorted(
        study["governance"]["global_importance"],
        key=lambda row: abs(float(row["importance"])),
        reverse=True,
    )[:8]
    importance_rows.sort(key=lambda row: float(row["importance"]))
    sensitivity_rows = [
        row
        for row in study["governance"]["representative_sensitivity"]
        if float(row["p25"]) != float(row["p75"])
    ]
    sensitivity_rows.sort(key=lambda row: float(row["probability_change"]))
    figure, (importance_axis, sensitivity_axis) = plt.subplots(
        1, 2, figsize=(9, 5.8), gridspec_kw={"width_ratios": (1, 1.12)}
    )

    labels = [
        FEATURE_LABELS.get(str(row["feature"]), str(row["feature"]).replace("_", " "))
        for row in importance_rows
    ]
    values = [float(row["importance"]) * 10_000 for row in importance_rows]
    deviations = [float(row["standard_deviation"]) * 10_000 for row in importance_rows]
    positions = list(range(len(importance_rows)))
    importance_axis.hlines(
        positions,
        [value - deviation for value, deviation in zip(values, deviations, strict=True)],
        [value + deviation for value, deviation in zip(values, deviations, strict=True)],
        color=MID,
        linewidth=1.5,
    )
    importance_axis.scatter(
        values,
        positions,
        color=[GOLD if value > 0 else BLUE if value < 0 else MID for value in values],
        edgecolor=INK,
        linewidth=0.65,
        s=48,
        zorder=3,
    )
    importance_axis.axvline(0, color=INK, linewidth=1)
    importance_axis.set_yticks(positions, labels)
    importance_axis.set_xlabel(r"Cambio del Brier al permutar (unidades de $10^{-4}$)")
    importance_axis.set_title("Importancia por permutación", loc="left", fontsize=10.5, fontweight="bold")
    importance_span = max(
        abs(value) + deviation for value, deviation in zip(values, deviations, strict=True)
    )
    for position, value in zip(positions, values, strict=True):
        importance_axis.text(
            value + 0.04 * importance_span,
            position,
            f"{value:.2f}".replace(".", ","),
            ha="left",
            va="center",
            fontsize=8,
        )
    importance_axis.set_xlim(
        min(value - deviation for value, deviation in zip(values, deviations, strict=True))
        - importance_span * 0.15,
        max(value + deviation for value, deviation in zip(values, deviations, strict=True))
        + importance_span * 0.18,
    )

    def transition_label(row: dict[str, Any]) -> str:
        feature = str(row["feature"])
        first = float(row["p25"])
        last = float(row["p75"])
        if feature == "original_upb":
            transition = f"{first / 1000:.0f} a {last / 1000:.0f} kUSD"
        elif feature == "original_interest_rate":
            transition = f"{first:.3f} a {last:.3f}%".replace(".", ",")
        else:
            transition = f"{first:g} a {last:g}".replace(".", ",")
        label = FEATURE_LABELS.get(feature, feature.replace("_", " "))
        return f"{label} ({transition})"

    sensitivity_labels = [transition_label(row) for row in sensitivity_rows]
    changes = [float(row["probability_change"]) * 100 for row in sensitivity_rows]
    sensitivity_positions = list(range(len(sensitivity_rows)))
    bars = sensitivity_axis.barh(
        sensitivity_positions,
        changes,
        color=[BLUE if value < 0 else GOLD for value in changes],
        edgecolor=INK,
        linewidth=0.6,
    )
    sensitivity_axis.axvline(0, color=INK, linewidth=1)
    sensitivity_axis.set_yticks(sensitivity_positions, sensitivity_labels)
    sensitivity_axis.set_xlabel("Cambio de PD de p25 a p75 (p.p.)")
    sensitivity_axis.set_title("Sensibilidad de un perfil", loc="left", fontsize=10.5, fontweight="bold")
    sensitivity_span = max(abs(value) for value in changes)
    sensitivity_axis.set_xlim(min(changes) - sensitivity_span * 0.22, max(changes) + sensitivity_span * 0.22)
    for bar, value in zip(bars, changes, strict=True):
        if value < -0.4:
            label_x = value / 2
            alignment = "center"
            label_color = WHITE
        elif value < 0:
            label_x = sensitivity_span * 0.04
            alignment = "left"
            label_color = INK
        else:
            label_x = value + sensitivity_span * 0.04
            alignment = "left"
            label_color = INK
        sensitivity_axis.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f} pp".replace(".", ","),
            ha=alignment,
            va="center",
            fontsize=8,
            color=label_color,
        )

    _save(
        plt,
        figure,
        output / "interpretability.png",
        "Qué utiliza el modelo y cómo responde",
        "Izquierda: importancia global media +/- 1 DE · Derecha: perturbación local p25-p75 · No son efectos causales",
    )


def _correlation_heatmap(study: dict[str, Any], output: Path, plotting: tuple[Any, ...]) -> None:
    plt, _, _, Rectangle, FuncFormatter = plotting
    from matplotlib.colors import LinearSegmentedColormap

    associations = study["governance"]["associations"]
    features = list(associations["features"])
    lookup = {
        frozenset((row["feature_a"], row["feature_b"])): float(row["spearman"])
        for row in associations["pairs"]
    }
    matrix = [
        [lookup[frozenset((feature_a, feature_b))] if column < row else math.nan for column, feature_b in enumerate(features)]
        for row, feature_a in enumerate(features)
    ]
    labels = [FEATURE_LABELS.get(feature, feature.replace("_", " ")) for feature in features]
    figure, axis = plt.subplots(figsize=(9, 6.2))
    palette = LinearSegmentedColormap.from_list("tfm_diverging", (BLUE, WHITE, GOLD))
    image = axis.imshow(matrix, cmap=palette, vmin=-1, vmax=1)
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            if math.isnan(value):
                continue
            axis.text(
                column,
                row,
                f"{value:.2f}".replace(".", ","),
                ha="center",
                va="center",
                fontsize=8.1,
                color=WHITE if abs(value) >= 0.7 else INK,
                fontweight="bold" if abs(value) >= 0.7 else "normal",
            )
            if abs(value) >= 0.7:
                axis.add_patch(
                    Rectangle(
                        (column - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor=INK,
                        linewidth=1.3,
                    )
                )
    axis.set_xticks(range(len(features)), labels, rotation=38, ha="right", rotation_mode="anchor")
    axis.set_yticks(range(len(features)), labels)
    axis.grid(False)
    axis.set_title("Matriz triangular de Spearman", loc="left", fontsize=10.5, fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.035)
    colorbar.set_label("Correlación de Spearman")
    colorbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.2f}".replace(".", ","))
    )
    figure.subplots_adjust(left=0.29, right=0.86, bottom=0.22, top=0.77)
    _save(
        plt,
        figure,
        output / "correlation_heatmap.png",
        "Dependencia entre predictores de originación",
        f"Muestra de desarrollo · n base={_format_int(associations['rows'])} · Cobertura pareada variable · Asociación, no causalidad",
        tight_layout=False,
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
    _scenario_tradeoff(study, output, plotting)
    _interpretability(study, output, plotting)
    _correlation_heatmap(study, output, plotting)


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
