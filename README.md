# Credit Risk Audit Model

Solución software reproducible para estudiar riesgo crediticio hipotecario con datos Freddie Mac. Estima probabilidad de incumplimiento (PD), exposición al incumplimiento (EAD), pérdida dado el incumplimiento (LGD) y pérdida esperada; cuando existe el panel mensual, añade transiciones entre estados de mora, curas y prepago. Su capacidad de validación principal es un backtesting *walk-forward* con cortes *as-of* que impiden utilizar etiquetas aún no disponibles.

El resultado analítico es un artefacto agregado. La memoria final se mantiene en [LaTeX](tfm/main.tex); el PDF compilado es un entregable local excluido de GitHub. No es un sistema de concesión, pricing, provisiones o capital regulatorio. IFRS 9 y CRR3 estándar se tratan únicamente como rutas secundarias de preparación, nunca como cálculos implementados para uso.

## Capacidades

- lectura directa del archivo analítico Freddie Mac ya preparado;
- preparación opcional de los ZIP trimestrales en una vista por préstamo y particiones mensuales;
- controles de esquema, integridad, privacidad y censura temporal;
- holdout fuera de tiempo y backtesting walk-forward 2020--2022 con desarrollo expansivo, roles separados y semántica as-of;
- comparación de modelos para PD, EAD y LGD;
- modelo mensual multiestado con regresión logística calibrada y HistGradientBoosting como challenger;
- pérdida esperada, escenarios de cartera y sensibilidad macroeconómica;
- métricas de calibración, discriminación, estabilidad y monitorización;
- memoria TFM reproducible en LaTeX.

## Requisitos

- Python 3.12 o 3.13; la ejecución Freddie está verificada con 3.12.13 y la integración continua pública con 3.13;
- dependencias fijadas en `requirements.lock`;
- `freddie-analysis.csv.zst` ya preparado y autorizado para el modo `full`;
- para el modelo multiestado real, `freddie-monthly/*.csv.zst` o los ZIP trimestrales originales para generarlo.

Creación del entorno en PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock
```

En macOS o Linux, la activación equivalente es:

```bash
source .venv/bin/activate
```

## Ejecución sintética

La comprobación pública no requiere datos privados:

```bash
python -m scripts.run_study --mode synthetic
```

El conjunto y el panel mensual de este modo son artificiales: comprueban contratos, reproducibilidad y funcionamiento del software, pero sus métricas no son evidencia empírica sobre Freddie Mac.

Genera `outputs/study-results.json` con resultados agregados reproducibles.

## Ejecución con datos Freddie Mac

Si ya tienes el archivo compacto, colócalo como `freddie-analysis.csv.zst` en la raíz y ejecuta:

```powershell
python -m scripts.run_study --mode full
```

Esa ruta conserva compatibilidad con el estudio agregado. Un compacto legacy sin `source_cutoff_date`, `pd_label_available_date`, `ead_label_available_date` y `lgd_label_available_date` mantiene el holdout histórico, pero marca el backtesting como no disponible y debe regenerarse. Si no existe `freddie-monthly`, el artefacto indicará además que la capa multiestado no está disponible.

Para reproducir también el histórico mensual, prepara primero los ZIP autorizados:

```powershell
python -m scripts.prepare_freddie --raw-root C:\ruta\freddie --years 2015 2016 2017 2018 2019 2020 2021 2022 --quarters 1 2 3 4 --sample-size 12500
python -m scripts.run_study --mode full
```

La recomendación Q1--Q4 con 12.500 préstamos por trimestre conserva aproximadamente 50.000 préstamos por año y mejora la cobertura estacional respecto a usar solo Q1. Aumentar `--sample-size` aporta más evidencia, especialmente para recuperaciones LGD, a costa de descarga, almacenamiento y tiempo de cálculo.

La preparación genera `freddie-analysis.csv.zst`, particiones `freddie-monthly/YYYYQn.csv.zst` y el linaje `.private/freddie/manifest.json`. El directorio mensual debe estar vacío para impedir mezclas entre ejecuciones. El identificador original no se escribe: ambas vistas se unen mediante una clave sustituta determinista. Estos archivos están excluidos del repositorio y su uso sigue sujeto a los términos de Freddie Mac.

## Producto local de cinco entradas

El producto mínimo entrena un único modelo de PD a 24 meses y acepta exactamente estas cinco entradas numéricas, en este orden: `origination_fico`, `original_dti`, `original_cltv`, `original_interest_rate` y `number_of_borrowers`. El entrenamiento usa desarrollo 2015--2018, calibración 2019 y validación 2020; no evalúa 2021--2022.

En PowerShell, con el compacto autorizado en la raíz, entrena el bundle local así:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_model --data freddie-analysis.csv.zst --output models\pd-model.joblib
```

Inicia el servicio local en otra consola:

```powershell
.\.venv\Scripts\python.exe -m scripts.serve_model --model models\pd-model.joblib --port 5000
```

Y envía una solicitud con las cinco entradas:

```powershell
$body = @{ origination_fico = 700; original_dti = 30; original_cltv = 80; original_interest_rate = 4.5; number_of_borrowers = 2 } | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -ContentType application/json -Body $body
```

La respuesta tiene esta forma:

```json
{
  "risk_score": 0.0237,
  "risk_level": "elevated",
  "model_version": "five-variable-pd-1",
  "horizon_months": 24,
  "warning": "Experimental academic risk score; not a credit decision."
}
```

`risk_score` es un score relativo y experimental para uso académico. No constituye aprobación o denegación, pricing, regulación, provisiones ni una probabilidad contractual. El compacto de Freddie y `models/pd-model.joblib` están ignorados y no se incluyen en el repositorio; su uso sigue sujeto a los términos de Freddie Mac.

`joblib.load` deserializa y puede ejecutar código. Sirve únicamente el bundle local generado por este proyecto; nunca cargues un `.joblib` descargado o de origen no confiable.

## Pruebas

```bash
python -m unittest discover -s tests
```

La integración continua instala las versiones bloqueadas, ejecuta toda la suite y reproduce el estudio sintético.

## Memoria TFM

La fuente publicable está en `tfm/main.tex` y conserva el índice académico de once capítulos. Para compilarla localmente con Tectonic:

```bash
cd tfm
tectonic main.tex
```

El PDF resultante está excluido del repositorio.

## Estructura

| Ruta | Responsabilidad |
| --- | --- |
| `configs/` | Datos, modelos, monitorización y escenarios |
| `scripts/run_study.py` | Ejecución completa y generación del artefacto agregado |
| `scripts/prepare_freddie.py` | Preparación local de ZIP trimestrales |
| `tfm/main.tex` | Memoria final y única fuente del informe |
| `src/data_access.py` | Lectura y escritura segura de CSV.ZST preparados |
| `src/freddie_data.py` | Estados mensuales, curas, EAD y LGD desde Freddie Mac |
| `src/pipeline.py` | Orquestación del estudio |
| `src/monthly_model.py` | Modelo calibrado de transiciones mensuales |
| `src/pd_model.py` | Entrenamiento, calibración y selección de PD |
| `src/loss_models.py` | Entrenamiento y selección de EAD y LGD |
| `tests/` | Pruebas unitarias y contratos de entrega |

## Controles principales

- huella SHA-256 calculada directamente sobre el archivo preparado;
- rechazo de identificadores privados en el archivo analítico;
- clave sustituta sin persistir el identificador Freddie original;
- validación de rangos, claves y componentes económicos;
- separación temporal estricta, ventanas walk-forward y filtros de disponibilidad de PD, EAD y LGD en cada fecha as-of;
- validación mensual actual por vintage de originación; el backtesting por calendario requiere cortar el panel mediante `performance_date`;
- transiciones solo entre meses consecutivos y estados terminales absorbentes;
- variables macro disponibles antes de la fecha de originación;
- resultados agregados sin observaciones individuales;
- identidad de implementación y versiones del entorno en cada ejecución.

## Limitaciones

El estudio es académico. El holdout 2021--2022 y sus cohortes no equivalen a folds repetidos; el motor walk-forward queda probado públicamente con datos sintéticos y requiere regenerar el compacto Freddie para producir evidencia real. Los escenarios son sensibilidades transparentes, no previsiones causales. LGD y pérdida esperada no deben declararse backtesteadas mientras falten recuperaciones maduras y cobertura completa. Cualquier uso bancario requeriría datos internos autorizados, validación independiente, gobierno del modelo y controles adecuados a la finalidad.
