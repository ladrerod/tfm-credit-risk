# PD24

PD24 es un producto académico local que estima el riesgo de impago hipotecario a 24 meses. Devuelve un `risk_score` y una banda relativa; no aprueba ni deniega préstamos y no estima EAD, LGD o pérdida esperada.

## Resultado

El máximo numérico de la comparación es XGBoost con 16 variables, pero su mejora no compensa once campos adicionales. El producto final usa XGBoost con cinco variables, ventana móvil de cinco años, 100 árboles, `learning_rate=0.1` y profundidad 2.

| Contrato | AUC media | KS medio | Peor AUC anual |
|---|---:|---:|---:|
| **5 variables: producto** | 0,755128 | 0,388325 | 0,706411 |
| 16 variables: máximo numérico | 0,756902 | 0,396730 | 0,710510 |

La diferencia 16–5 es solo +0,001775 de AUC media y +0,004099 en el peor año. Se conserva cinco por parsimonia y facilidad de uso, dejando claro que no es el máximo literal. Las 18 columnas históricas no forman un contrato independiente: `amortization_type` es constante y `mortgage_insurance_type` se deriva del porcentaje de seguro.

“Mejor” significa mejor entre las alternativas declaradas y sobre los años 2018, 2019, 2021 y 2022. El año 2020 se conserva como estrés retrospectivo y 2023 como evaluación temporal reutilizada, no como holdout virgen. El score sirve para ordenar riesgo, pero no se presenta como una PD contractual calibrada.

## Datos y notebook

El único input es `freddie-pd24-wide.csv.zst`, en la raíz del proyecto e ignorado por Git:

- 999.965 filas y 15.154 eventos en el fichero fuente.
- 996.447 filas y 15.082 eventos en la población común de comparación.
- 3.518 filas se excluyen porque no permiten enlazar de forma unívoca las variables ampliadas; la exclusión es idéntica para todos los contratos.
- SHA-256: `316dc7b4c16878d9ca62694626263e3a9276512809000d003f456fffe0740589`.
- Cobertura utilizada: cohortes 2004–2023, cuatro trimestres por año.

[`notebooks/01_seleccion_modelo.ipynb`](notebooks/01_seleccion_modelo.ipynb) se ejecuta de arriba abajo. Valida el input y el diseño sin fuga, revalida las ventanas de 1, 3 y 5 años y la expansiva, rehace las comparaciones relevantes de variables y familias y genera las figuras. Después reutiliza `src/` para entrenar, guardar y cargar el bundle, puntuar un préstamo, probar la API con una entrada válida y otra inválida, construir la interfaz Gradio y ejecutar `unittest`.

La ruta *forward* completa y la criba histórica de hiperparámetros se conservan como auditoría congelada porque repetir toda esa exploración tarda aproximadamente 90 minutos. El notebook sí reejecuta las decisiones que afectan al producto final.

La ventana de diez años no se prueba porque el corte 2018 necesitaría la cohorte 2003, ausente. Truncarla solo en ese fold rompería la comparabilidad; la ventana expansiva cubre la alternativa de historia larga desde 2004.

## Entradas del producto

`origination_fico`, `original_dti`, `original_cltv`, `original_interest_rate` y `number_of_borrowers`.

Son cinco variables numéricas disponibles al inicio del préstamo. El pipeline imputa la mediana aprendida en desarrollo antes de XGBoost.

## Ejecutar

Entorno verificado: Python 3.12.13.

Para ejecutar el notebook, abra la raíz del proyecto en VS Code o Jupyter, abra `notebooks/01_seleccion_modelo.ipynb`, seleccione como kernel `.venv\Scripts\python.exe` y pulse **Run All**.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m scripts.train_model --data freddie-pd24-wide.csv.zst --output models\pd24-model.joblib
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Interfaz local:

```powershell
.\.venv\Scripts\python.exe -m scripts.serve_demo --model models\pd24-model.joblib --port 7860
```

Abra `http://127.0.0.1:7860`. La interfaz no guarda los préstamos.

API local:

```powershell
.\.venv\Scripts\python.exe -m scripts.serve_model --model models\pd24-model.joblib --port 5000
```

`GET /health` comprueba el servicio y `POST /predict` calcula el score. `GET /` devuelve 404 de forma intencionada porque la API no incluye una página web.

Ejemplo de cuerpo JSON:

```json
{
  "origination_fico": 700,
  "original_dti": 30,
  "original_cltv": 80,
  "original_interest_rate": 4.5,
  "number_of_borrowers": 2
}
```

El bundle `joblib` debe considerarse un fichero local de confianza: no se debe cargar un bundle recibido de terceros.

## Límites

Es una demostración académica local, no un servicio multiusuario ni una herramienta de concesión, pricing, provisiones o capital. Un uso real exigiría una cohorte futura madura y completamente no observada, recalibración, validación externa, control de deriva, análisis de sesgo y gobierno del modelo.
