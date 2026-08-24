# Credit Risk Audit Model

Estudio reproducible de riesgo crediticio hipotecario. El proyecto consume datos Freddie Mac ya preparados y estima probabilidad de incumplimiento (PD), exposición al incumplimiento (EAD), pérdida dado el incumplimiento (LGD) y pérdida esperada. Cuando existe el panel mensual, añade transiciones entre estados de mora, curas y prepago.

El resultado analítico es un artefacto agregado. La memoria final se mantiene en [LaTeX](tfm/main.tex); el PDF compilado es un entregable local excluido de GitHub. No es un sistema de concesión, pricing, provisiones o capital regulatorio.

## Capacidades

- lectura directa del archivo analítico Freddie Mac ya preparado;
- preparación opcional de los ZIP trimestrales en una vista por préstamo y particiones mensuales;
- controles de esquema, integridad, privacidad y censura temporal;
- desarrollo, calibración, validación y prueba mediante cohortes separadas;
- comparación de modelos para PD, EAD y LGD;
- modelo mensual multiestado con regresión logística calibrada y HistGradientBoosting como challenger;
- pérdida esperada, escenarios de cartera y sensibilidad macroeconómica;
- métricas de calibración, discriminación, estabilidad y monitorización;
- memoria TFM reproducible en LaTeX.

## Requisitos

- Python 3.13;
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

El panel mensual de este modo contiene una transición artificial por préstamo: comprueba el flujo, pero no representa trayectorias Freddie reales.

Genera `outputs/study-results.json` con resultados agregados reproducibles.

## Ejecución con datos Freddie Mac

Si ya tienes el archivo compacto, colócalo como `freddie-analysis.csv.zst` en la raíz y ejecuta:

```powershell
python -m scripts.run_study --mode full
```

Esa ruta conserva compatibilidad con el estudio agregado; si no existe `freddie-monthly`, el artefacto indicará que la capa multiestado no está disponible.

Para reproducir también el histórico mensual, prepara primero los ZIP autorizados:

```powershell
python -m scripts.prepare_freddie --raw-root C:\ruta\freddie --years 2015 2016 2017 2018 2019 2020 2021 2022 --quarters 1 --sample-size 50000
python -m scripts.run_study --mode full
```

La preparación genera `freddie-analysis.csv.zst`, particiones `freddie-monthly/YYYYQn.csv.zst` y el linaje `.private/freddie/manifest.json`. El directorio mensual debe estar vacío para impedir mezclas entre ejecuciones. El identificador original no se escribe: ambas vistas se unen mediante una clave sustituta determinista. Estos archivos están excluidos del repositorio y su uso sigue sujeto a los términos de Freddie Mac.

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
- separación temporal estricta para la PD agregada y validación por vintage de originación para el modelo mensual;
- transiciones solo entre meses consecutivos y estados terminales absorbentes;
- variables macro disponibles antes de la fecha de originación;
- resultados agregados sin observaciones individuales;
- identidad de implementación y versiones del entorno en cada ejecución.

## Limitaciones

El estudio es académico. Los escenarios son sensibilidades transparentes, no previsiones causales. Cualquier uso bancario requeriría datos internos autorizados, reconciliación contable, validación independiente, gobierno del modelo y controles adecuados a la decisión.
