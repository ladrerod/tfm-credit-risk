from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable


SOURCES = [
    ("Fannie Mae, Single-Family Loan Performance Data", "https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data"),
    ("Fannie Mae, Glossary and File Layout", "https://www.fanniemae.com/media/document/pdf/cas-glossarypdf"),
    ("Fannie Mae, Legal Disclosure", "https://www.fanniemae.com/about-us/legal-disclosure"),
    ("BLS, Local Area Unemployment Statistics", "https://www.bls.gov/lau/home.htm"),
    ("FHFA, House Price Index datasets", "https://www.fhfa.gov/house-price-index?tab=HPI+Datasets"),
    ("Basel Committee, IRB risk components", "https://www.bis.org/basel_framework/chapter/CRE/32.htm"),
    ("Federal Reserve, Revised Guidance on Model Risk Management", "https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm"),
    ("scikit-learn, Logistic Regression", "https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression"),
    ("scikit-learn, Histogram-Based Gradient Boosting", "https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting"),
    ("Hugging Face, Trusted Publishers", "https://huggingface.co/docs/hub/trusted-publishers"),
    ("GitHub, OpenID Connect reference", "https://docs.github.com/en/actions/reference/security/oidc"),
]


def _f(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/d"
    number = float(value)
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:,.1f} M".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{number:.{digits}f}".replace(".", ",")


def _pct(value: object, digits: int = 2) -> str:
    return "n/d" if value is None else f"{100 * float(value):.{digits}f}%".replace(".", ",")


def _bar_chart(title: str, labels: Iterable[object], values: Iterable[float], *, percent: bool = False) -> str:
    names, data = list(labels), [float(value or 0) for value in values]
    width, height, left, top, bottom = 760, 300, 72, 40, 70
    inner_width, inner_height = width - left - 25, height - top - bottom
    maximum = max(data, default=1) or 1
    gap = inner_width / max(len(data), 1)
    bars = []
    for index, (label, value) in enumerate(zip(names, data)):
        bar_height = inner_height * value / maximum
        x = left + index * gap + gap * 0.16
        y = top + inner_height - bar_height
        shown = _pct(value) if percent else _f(value, 2)
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{gap * .68:.1f}" height="{bar_height:.1f}" rx="4" fill="#2673b8"/>'
            f'<text x="{x + gap * .34:.1f}" y="{max(y - 7, 18):.1f}" text-anchor="middle">{html.escape(shown)}</text>'
            f'<text x="{x + gap * .34:.1f}" y="{height - 38}" text-anchor="middle">{html.escape(str(label))}</text>'
        )
    return (
        f'<figure><figcaption>{html.escape(title)}</figcaption><svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        f'<line x1="{left}" y1="{top + inner_height}" x2="{width - 25}" y2="{top + inner_height}" stroke="#6b7785"/>'
        + "".join(bars)
        + "</svg></figure>"
    )


def _line_chart(
    title: str,
    labels: Iterable[object],
    series: list[tuple[str, Iterable[float]]],
    *,
    unit_max: float | None = None,
) -> str:
    names = list(labels)
    prepared = [(name, [float(value or 0) for value in values]) for name, values in series]
    width, height, left, top, bottom = 760, 300, 72, 40, 60
    inner_width, inner_height = width - left - 25, height - top - bottom
    maximum = unit_max or max((max(values, default=0) for _, values in prepared), default=1) or 1
    colors = ("#2673b8", "#d4662f", "#2f8a66")
    paths = []
    for series_index, (name, values) in enumerate(prepared):
        points = []
        for index, value in enumerate(values):
            x = left + (inner_width * index / max(len(names) - 1, 1))
            y = top + inner_height * (1 - value / maximum)
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[series_index % len(colors)]
        paths.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>'
            f'<text x="{left + series_index * 170}" y="20" fill="{color}">{html.escape(name)}</text>'
        )
    ticks = "".join(
        f'<text x="{left + inner_width * index / max(len(names) - 1, 1):.1f}" y="{height - 25}" text-anchor="middle">{html.escape(str(label))}</text>'
        for index, label in enumerate(names)
    )
    return (
        f'<figure><figcaption>{html.escape(title)}</figcaption><svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        f'<line x1="{left}" y1="{top + inner_height}" x2="{width - 25}" y2="{top + inner_height}" stroke="#6b7785"/>'
        + "".join(paths)
        + ticks
        + "</svg></figure>"
    )


def _table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<div class=table-wrap><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _section(number: int, title: str, content: str) -> str:
    return f'<section data-section="{number}"><p class=kicker>Módulo {number}</p><h2>{html.escape(title)}</h2>{content}</section>'


def build_report(artifact_path: str | Path, output_path: str | Path) -> None:
    result = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    if result.get("contains_row_data") is not False:
        raise ValueError("the report accepts aggregate artifacts only")
    quality, pd_result = result["data_quality"], result["pd"]
    loss, expected = result["loss_components"], result["expected_loss"]
    test_pd = pd_result["test_metrics"]
    cohorts = quality["cohorts"]
    test_cohorts = pd_result["test_cohorts"]
    ead_metrics = loss["ead"]["validation_metrics"]
    lgd_metrics = loss["lgd"]["validation_metrics"]
    scenarios = result["scenarios"]
    importance = result["governance"]["global_importance"][:10]
    drift = result["monitoring"]["feature_drift"][:10]
    source_label = "datos privados verificados" if result["identity"]["source"] in {"private_dataset", "local_private_dataset"} else "datos sintéticos de comprobación"

    sections = []
    sections.append(
        _section(
            1,
            "Resumen ejecutivo",
            f"""
            <p>En este trabajo he construido una solución reproducible para estudiar riesgo hipotecario desde la originación hasta la pérdida económica. El núcleo enlaza una PD a 24 meses con EAD y LGD para obtener pérdida esperada, y añade políticas de retención, sensibilidades macroeconómicas, explicabilidad y monitorización. La ejecución mostrada utiliza <strong>{source_label}</strong>.</p>
            <div class=cards><article><span>Préstamos analizados</span><strong>{quality['rows']:,}</strong></article><article><span>Tasa de evento</span><strong>{_pct(quality['event_rate'])}</strong></article><article><span>AUC de prueba</span><strong>{_f(test_pd['roc_auc'])}</strong></article><article><span>Brier de prueba</span><strong>{_f(test_pd['brier'],4)}</strong></article><article><span>Pérdida esperada</span><strong>{_f(expected['total_expected_loss'])}</strong></article></div>
            <p>El resultado principal no es un motor de decisión. Es una demostración analítica de cartera: separa discriminación y calibración, reconoce el cambio temporal y conserva referencias sencillas cuando un método complejo no mejora fuera de muestra. Esta cautela coincide con el enfoque de gobierno basado en finalidad y materialidad de la guía supervisora revisada [7].</p>
            {_bar_chart('Volumen por cohorte', [row['cohort'] for row in cohorts], [row['rows'] for row in cohorts])}
            """,
        )
    )
    sections.append(
        _section(
            2,
            "Contexto, motivación y objetivos",
            f"""
            <p>Trabajo en el sector bancario y planteo el problema como lo haría ante una cartera real: primero debo definir qué riesgo mido, cuándo lo observo y para qué decisión se emplearía. El préstamo hipotecario combina horizonte largo, baja frecuencia de default, recuperaciones tardías y fuerte dependencia del ciclo. Por eso una AUC aislada no basta.</p>
            <p>Mi objetivo técnico es estimar <em>PD</em>, <em>EAD</em> y <em>LGD</em>, reconciliarlas como <code>EL = PD × EAD × LGD</code> y someter el resultado a cortes temporales. Basel define LGD como pérdida en porcentaje de EAD [6], pero este estudio no pretende reproducir un cálculo IRB: su PD usa 24 meses, no el horizonte regulatorio de un año, y carece de validación interna bancaria.</p>
            <ul><li>Medir capacidad predictiva y calibración por cohortes.</li><li>Cuantificar exposición y severidad sin esconder el exceso de ceros.</li><li>Comparar reglas de retención y shocks macro transparentes.</li><li>Dejar trazabilidad de datos, código, supuestos y límites.</li></ul>
            {_bar_chart('Tasa de evento por cohorte', [row['cohort'] for row in cohorts], [row['event_rate'] for row in cohorts], percent=True)}
            """,
        )
    )
    sections.append(
        _section(
            3,
            "Fuentes, derechos de uso y privacidad",
            """
            <p>La fuente de préstamo es Fannie Mae Single-Family Loan Performance Data, creada para facilitar el análisis de comportamiento crediticio de una parte de su libro hipotecario [1]. El diccionario oficial define originación, mora, saldo, foreclosure, disposición y costes [2]. No intento identificar personas ni enlazar registros con fuentes externas.</p>
            <p>Los términos de Fannie Mae restringen la redistribución externa de los datos y permiten resultados académicos no comerciales siempre que no permitan reconstruir registros [3]. Por ello el repositorio de código no contiene datos: el acceso automatizado a una copia privada solo debe activarse cuando la autorización escrita cubra expresamente a los revisores. El HTML contiene exclusivamente agregados.</p>
            <p>Para contexto macro utilizo LAUS de BLS, que publica empleo y desempleo por estado y residencia [4], y el HPI de FHFA, índice público de compraventas repetidas para vivienda unifamiliar [5]. Impongo retardos de publicación y rechazo cualquier observación cuyo <em>as-of date</em> no preceda a la originación.</p>
            <div class=note><strong>Privacidad por diseño.</strong> Los fragmentos privados excluyen identificadores, se verifican con SHA-256 y se leen por bloques. La identidad de datos del resultado es <code>{identity}</code>.</div>
            """.replace("{identity}", html.escape(result["identity"]["implementation_sha256"][:16] + "…")),
        )
    )
    sections.append(
        _section(
            4,
            "Metodología y arquitectura",
            f"""
            <p>He organizado el flujo como una secuencia determinista: validación del manifiesto, lectura comprimida por bloques, controles de calidad, cortes temporales, entrenamiento, calibración, evaluación, pérdida esperada, escenarios y un artefacto agregado común al HTML y al documento académico.</p>
            <div class=flow><span>CSV.ZST privado</span><b>→</b><span>integridad y esquema</span><b>→</b><span>cohortes maduras</span><b>→</b><span>PD / EAD / LGD</span><b>→</b><span>EL y escenarios</span><b>→</b><span>informe</span></div>
            <p>El corte temporal es estricto: desarrollo {result['methodology']['development_years']}, calibración {result['methodology']['calibration_year']}, validación {result['methodology']['validation_year']} y prueba {result['methodology']['test_years']}. El conjunto de prueba no participa en selección ni calibración. Las credenciales de automatización son efímeras mediante OIDC [10][11].</p>
            {_table(['Control', 'Aplicación'], [['Horizonte', '24 meses completos'], ['Fuga', 'solo variables conocidas en originación'], ['Selección', 'validación temporal'], ['Prueba', '2021–2022'], ['Salida', 'agregados sin filas']])}
            """,
        )
    )
    missing = sorted(quality["missingness"].items(), key=lambda item: item[1], reverse=True)[:10]
    sections.append(
        _section(
            5,
            "EDA y preparación",
            f"""
            <p>La preparación convierte el panel mensual en una observación por préstamo, conserva variables de originación y etiqueta el primer evento de 90+ días, foreclosure o equivalente dentro de 24 meses. Exijo madurez completa; de otro modo un préstamo reciente parecería sano por censura.</p>
            <p>La tasa global de evento es {_pct(quality['event_rate'])}. Las correlaciones se interpretan como asociación descriptiva, no como causalidad. También reviso cobertura, rangos y colinealidad antes de ajustar modelos.</p>
            {_bar_chart('Variables con mayor ausencia', [name for name, _ in missing], [value for _, value in missing], percent=True)}
            {_table(['Cohorte', 'Filas', 'Eventos', 'Tasa'], [[row['cohort'], f"{row['rows']:,}", row['events'], _pct(row['event_rate'])] for row in cohorts])}
            """,
        )
    )
    validation_metrics = pd_result["validation_metrics"]
    sections.append(
        _section(
            6,
            "Modelo de probabilidad de incumplimiento",
            f"""
            <p>Comparo regresión logística regularizada [8] con HistGradientBoosting [9]. Calibro ambos sobre 2019 y selecciono sobre 2020 priorizando Brier y log-loss, con AUC como criterio posterior. El modelo elegido es <strong>{html.escape(pd_result['selected_name'])}</strong>.</p>
            <p>En prueba obtengo AUC {_f(test_pd['roc_auc'])}, PR-AUC {_f(test_pd['pr_auc'])}, Brier {_f(test_pd['brier'],4)}, KS {_f(test_pd['ks'])} y prevalencia {_pct(test_pd['prevalence'])}. Presento PR-AUC porque el default es infrecuente y la precisión media responde mejor a ese desequilibrio que la exactitud.</p>
            {_bar_chart('Brier en validación (menor es mejor)', list(validation_metrics), [row['brier'] for row in validation_metrics.values()])}
            {_line_chart('Calibración en prueba', [row['bin'] for row in pd_result['test_calibration']], [('PD media', [row['mean_probability'] for row in pd_result['test_calibration']]), ('Evento observado', [row['event_rate'] for row in pd_result['test_calibration']])], unit_max=max(max(row['mean_probability'], row['event_rate']) for row in pd_result['test_calibration']) * 1.1)}
            {_line_chart('AUC por cohorte de prueba', [row['cohort_year'] for row in test_cohorts], [('AUC', [row['roc_auc'] for row in test_cohorts])], unit_max=1.0)}
            <p>El challenger macro queda <strong>{'promocionado' if pd_result['macro_challenger']['promoted'] else 'no promocionado'}</strong>. Que una variable sea económicamente razonable no basta: debe mejorar fuera de tiempo sin degradar calibración.</p>
            """,
        )
    )
    sections.append(
        _section(
            7,
            "Modelización de EAD, LGD y pérdida esperada",
            f"""
            <p>EAD se expresa como saldo al default sobre saldo original. Comparo la referencia 1,0 con un regresor no lineal y conservo <strong>{html.escape(loss['ead']['selected_name'])}</strong>. Para LGD uso una arquitectura de dos etapas: probabilidad de pérdida positiva y severidad condicionada a pérdida. Esta separación evita que una regresión robusta colapse a cero ante una distribución con mucha masa nula.</p>
            {_bar_chart('EAD: error relativo del total en validación', list(ead_metrics), [row['portfolio_relative_error'] for row in ead_metrics.values()], percent=True)}
            {_bar_chart('LGD: error relativo del total en validación', list(lgd_metrics), [row['portfolio_relative_error'] for row in lgd_metrics.values()], percent=True)}
            <p>El modelo LGD seleccionado es <strong>{html.escape(loss['lgd']['selected_name'])}</strong>. Reporto también WAPE y MAE: recuperar el total de cartera no garantiza precisión préstamo a préstamo. Para el objetivo económico acoto la LGD observada a [0, 2], conservando en calidad el recuento de {quality['lgd_observed_tail']['below_zero']:,} valores bajo cero y {quality['lgd_observed_tail']['above_two']:,} sobre dos. La marca de aptitud para decisión es <strong>{'sí' if loss['decision_grade'] else 'no'}</strong>; incluso cuando los umbrales académicos se superan, sigue faltando validación bancaria.</p>
            {_bar_chart('Pérdida esperada por cohorte', [row['cohort_year'] for row in expected['cohorts']], [row['total_expected_loss'] for row in expected['cohorts']])}
            <p>La exposición total modelizada es {_f(expected['exposure_at_default'])} y la pérdida esperada {_f(expected['total_expected_loss'])}, equivalente a {_pct(expected['expected_loss_rate'])} de EAD. No la interpreto como provisión IFRS 9 ni capital regulatorio.</p>
            """,
        )
    )
    scenario_labels = [f"{row['policy']} / {row['macro_scenario']}" for row in scenarios]
    sections.append(
        _section(
            8,
            "Simulación de políticas de riesgo y escenarios macroeconómicos",
            f"""
            <p>Las reglas actúan sobre préstamos Fannie ya originados. Por tanto hablo de <em>retención</em>, no de aprobación. Cada combinación aplica límites de CLTV, DTI y PD; después desplaza los log-odds con shocks declarados de desempleo y HPI.</p>
            {_bar_chart('Retención por política y escenario', scenario_labels, [row['retention_rate'] for row in scenarios], percent=True)}
            {_bar_chart('Pérdida esperada retenida', scenario_labels, [row['retained_expected_loss'] for row in scenarios])}
            <p>Los shocks son sensibilidades ilustrativas, no previsiones causales. Sirven para comprobar dirección, concentración y materialidad. Una aplicación bancaria exigiría escenarios aprobados, coeficientes validados y reconciliación con presupuesto y capital.</p>
            {_table(['Política / macro', 'Retención', 'Exposición', 'PD media', 'EL'], [[label, _pct(row['retention_rate']), _f(row['retained_exposure']), _pct(row['mean_retained_pd']), _f(row['retained_expected_loss'])] for label, row in zip(scenario_labels, scenarios)])}
            """,
        )
    )
    sections.append(
        _section(
            9,
            "Interpretabilidad, gobierno y productivización",
            f"""
            <p>Uso importancia por permutación sobre validación y sensibilidad de un perfil representativo. La primera mide cuánto empeora Brier al romper una variable; la segunda cambia del percentil 25 al 75 manteniendo el resto en mediana o moda. Ninguna constituye explicación causal.</p>
            {_bar_chart('Importancia global por permutación', [row['feature'] for row in importance], [max(row['importance'], 0) for row in importance])}
            {_bar_chart('PSI de desarrollo a prueba', [row['feature'] for row in drift], [row['psi'] for row in drift]) if drift else '<p>No hay variables suficientes para calcular PSI.</p>'}
            <p>La guía supervisora vigente enfatiza validación, monitorización y controles proporcionales al uso [7]. La solución conserva identidad de datos y código, seed, runtime, métricas por cohorte y alertas. Si el deterioro persiste, la respuesta puede ser ajuste, recalibración o reconstrucción; la complejidad no se presume mejor.</p>
            """,
        )
    )
    sections.append(
        _section(
            10,
            "Resultados, conclusiones y líneas futuras",
            f"""
            <p>Concluyo que la separación PD–EAD–LGD es viable con datos de performance, pero no todos los componentes alcanzan la misma fiabilidad. La PD ofrece discriminación útil con AUC {_f(test_pd['roc_auc'])}; el cambio entre 2021 y 2022 muestra que la estabilidad temporal es un resultado, no un supuesto. La LGD exige atención especial por resolución tardía, ceros y colas.</p>
            <p>Mi decisión metodológica principal es conservar el modelo más sencillo que mantenga calibración y estabilidad. El componente macro queda como challenger o sensibilidad cuando no mejora fuera de tiempo. Las políticas son comparaciones de cartera y no sustituyen underwriting.</p>
            <ol><li>Validar con datos bancarios autorizados.</li><li>Reestimar LGD y EAD con costes, recuperaciones y exposición interna.</li><li>Recalibrar con cohortes recientes cuando maduren sus etiquetas.</li><li>Revisar escenarios macro con supuestos aprobados y fuentes con derechos claros.</li><li>Automatizar monitorización con resultados maduros y revisión humana.</li></ol>
            <div class=note><strong>Conclusión operativa.</strong> El estudio es reproducible y auditable, pero su uso debe permanecer académico hasta superar validación independiente, cobertura de derechos y pruebas con datos internos.</div>
            """,
        )
    )
    bibliography = "".join(
        f'<li id="ref-{index}"><a href="{html.escape(url)}">{html.escape(title)}</a></li>'
        for index, (title, url) in enumerate(SOURCES, start=1)
    )
    sections.append(
        _section(
            11,
            "Bibliografía y anexos",
            f"""
            <h3>Bibliografía</h3><ol class=references>{bibliography}</ol>
            <h3>Anexo A · Definiciones</h3>
            {_table(['Componente', 'Definición del estudio'], [['PD', 'Evento 90+ días, foreclosure o equivalente en 24 meses'], ['EAD', 'Saldo al default / saldo original'], ['LGD', 'Pérdida económica / EAD, acotada entre 0 y 2'], ['EL', 'PD × EAD × LGD']])}
            <h3>Anexo B · Reproducibilidad</h3>
            <p>Versión del artefacto: {result['version']}. Seed: {result['identity'].get('seed', 'n/d')}. Implementación: <code>{html.escape(result['identity']['implementation_sha256'])}</code>. Python {html.escape(result['identity']['runtime']['python'])}; numpy {html.escape(result['identity']['runtime']['numpy'])}; pandas {html.escape(result['identity']['runtime']['pandas'])}; scikit-learn {html.escape(result['identity']['runtime']['scikit_learn'])}.</p>
            <h3>Anexo C · Limitaciones declaradas</h3><ul>{''.join(f'<li>{html.escape(value)}</li>' for value in result['limitations'])}</ul>
            """,
        )
    )

    css = """
    :root{--ink:#172434;--muted:#5d6b7a;--blue:#123f6d;--accent:#2673b8;--paper:#fff;--line:#dce3ea;--soft:#edf4fa}
    *{box-sizing:border-box}body{margin:0;background:#e9eef3;color:var(--ink);font:16px/1.62 Georgia,serif}main,.cover{width:min(1100px,calc(100% - 32px));margin:24px auto;background:var(--paper);box-shadow:0 10px 30px #10203022}.cover{min-height:92vh;padding:12vh 9%;display:flex;flex-direction:column;justify-content:center;border-top:12px solid var(--blue)}.cover h1{font:700 clamp(2.4rem,6vw,5.4rem)/1.02 system-ui;margin:.2em 0}.cover p{font:1.2rem/1.5 system-ui;color:var(--muted);max-width:60ch}.eyebrow,.kicker{font:700 .78rem/1 system-ui;text-transform:uppercase;letter-spacing:.16em;color:var(--accent)}section{padding:72px 8%;min-height:100vh;break-before: page}section+section{border-top:1px solid var(--line)}h2{font:700 clamp(2rem,4vw,3.2rem)/1.08 system-ui;color:var(--blue);margin:.25em 0 .8em}h3{font:700 1.25rem/1.2 system-ui;color:var(--blue);margin-top:2em}p,li{max-width:82ch}code{font:13px/1.4 ui-monospace,monospace;background:var(--soft);padding:.15rem .35rem;border-radius:4px;overflow-wrap:anywhere}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:28px 0}.cards article{border:1px solid var(--line);border-radius:10px;padding:18px;background:#fbfdff}.cards span{display:block;font:12px/1.3 system-ui;color:var(--muted);text-transform:uppercase}.cards strong{display:block;font:700 1.65rem/1.2 system-ui;color:var(--blue);margin-top:8px}.note{border-left:5px solid var(--accent);background:var(--soft);padding:18px 22px;margin:28px 0}.flow{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:28px 0}.flow span{background:var(--soft);border:1px solid #cbdbea;border-radius:999px;padding:9px 14px;font:600 13px/1 system-ui}.flow b{color:var(--accent)}figure{margin:32px 0;border:1px solid var(--line);border-radius:12px;padding:16px;background:#fff}figcaption{font:700 14px/1.3 system-ui;color:var(--blue);margin-bottom:8px}svg{display:block;width:100%;height:auto;font:12px system-ui;overflow:visible}.table-wrap{overflow-x:auto;margin:26px 0}table{width:100%;border-collapse:collapse;font:14px/1.4 system-ui}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left}th{background:var(--soft);color:var(--blue)}a{color:#0b65a3}.cite{font:700 12px system-ui;text-decoration:none}.references li{margin:.45rem 0}.footer{padding:24px 8%;color:var(--muted);font:13px system-ui}
    @media(max-width:700px){main,.cover{width:100%;margin:0;box-shadow:none}.cover,section{padding:42px 22px}.flow b{display:none}table{min-width:620px}}
    @media print{body{background:#fff;font-size:10.3pt}main,.cover{width:auto;margin:0;box-shadow:none}.cover{height:100vh}.cover,section{padding:16mm 14mm;min-height:0}section{break-before: page}figure{break-inside:avoid}a{color:inherit;text-decoration:none}.footer{display:none}@page{size:A4;margin:0}}
    """
    document = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Estudio de riesgo crediticio hipotecario</title><style>{css}</style></head>
<body><header class=cover><p class=eyebrow>Trabajo Fin de Máster · Data Science</p><h1>Riesgo crediticio hipotecario</h1><p>PD, EAD, LGD, pérdida esperada y escenarios con Fannie Mae y contexto macroeconómico oficial.</p><p><strong>Daniel Ladrero Meca</strong><br>Estudio académico reproducible</p></header><main>{''.join(sections)}<footer class=footer>Documento autónomo generado desde agregados verificables.</footer></main></body></html>"""
    for index in range(1, len(SOURCES) + 1):
        document = document.replace(
            f"[{index}]", f'<a class="cite" href="#ref-{index}">[{index}]</a>'
        )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
