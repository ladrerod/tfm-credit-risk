# Credit Risk Audit Model

Estudio reproducible de riesgo crediticio hipotecario. El proyecto consume un archivo Freddie Mac ya preparado, incorpora su contexto macroeconómico y estima probabilidad de incumplimiento (PD), exposición al incumplimiento (EAD), pérdida dado el incumplimiento (LGD) y pérdida esperada.

El resultado es un artefacto agregado y un [informe HTML autónomo](results/mortgage-credit-risk-study.html). No es un sistema de concesión, pricing, provisiones o capital regulatorio.

## Capacidades

- lectura directa del archivo analítico Freddie Mac ya preparado;
- controles de esquema, integridad, privacidad y censura temporal;
- desarrollo, calibración, validación y prueba mediante cohortes separadas;
- comparación de modelos para PD, EAD y LGD;
- pérdida esperada, escenarios de cartera y sensibilidad macroeconómica;
- métricas de calibración, discriminación, estabilidad y monitorización;
- informe reproducible sin dependencias web externas.

## Requisitos

- Python 3.13;
- dependencias fijadas en `requirements.lock`;
- `freddie-analysis.csv.zst` ya preparado y autorizado únicamente para el modo `full`.

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

Genera:

- `outputs/study-results.json`: resultados agregados reproducibles;
- `results/mortgage-credit-risk-study.html`: informe autónomo.

## Ejecución con datos Freddie Mac

Coloca el archivo ya tratado que recibirás con el nombre `freddie-analysis.csv.zst` en la raíz del proyecto y ejecuta:

```powershell
python -m scripts.run_study --mode full
```

No se necesitan ZIP, manifiestos, variables de entorno ni pasos de transformación. El archivo preparado está excluido del repositorio y su uso sigue sujeto a los términos de Freddie Mac.

## Pruebas

```bash
python -m unittest discover -s tests
```

La integración continua instala las versiones bloqueadas, ejecuta toda la suite y reproduce el estudio sintético.

## Estructura

| Ruta | Responsabilidad |
| --- | --- |
| `configs/` | Datos, modelos, monitorización y escenarios |
| `scripts/run_study.py` | Ejecución completa y generación del informe |
| `scripts/build_report.py` | Reconstrucción del HTML desde un artefacto JSON |
| `src/data_access.py` | Lectura segura del archivo analítico preparado |
| `src/pipeline.py` | Orquestación del estudio |
| `src/pd_model.py` | Entrenamiento, calibración y selección de PD |
| `src/loss_models.py` | Entrenamiento y selección de EAD y LGD |
| `src/reporting.py` | Informe HTML autónomo |
| `tests/` | Pruebas unitarias y contratos de entrega |

## Controles principales

- huella SHA-256 calculada directamente sobre el archivo preparado;
- rechazo de identificadores privados en el archivo analítico;
- validación de rangos, claves y componentes económicos;
- separación temporal estricta entre desarrollo y prueba;
- variables macro disponibles antes de la fecha de originación;
- resultados agregados sin observaciones individuales;
- identidad de implementación y versiones del entorno en cada ejecución.

## Limitaciones

El estudio es académico. Los escenarios son sensibilidades transparentes, no previsiones causales. Cualquier uso bancario requeriría datos internos autorizados, reconciliación contable, validación independiente, gobierno del modelo y controles adecuados a la decisión.
