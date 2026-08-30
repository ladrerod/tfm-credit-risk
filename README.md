# PD24

PD24 es un producto académico local que estima el riesgo de impago hipotecario a 24 meses con cinco variables de originación. Necesita un único input local: `freddie-pd24.csv.zst` en la raíz; el compacto está ignorado por Git.

## Ejecutar

Entorno verificado: Python 3.12.13. Las dependencias exactas están fijadas en `requirements.lock`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m scripts.train_model --data freddie-pd24.csv.zst --output models\pd24-model.joblib
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

La demostración web impartida con Gradio se inicia en local con:

```powershell
.\.venv\Scripts\python.exe -m scripts.serve_demo --model models\pd24-model.joblib --port 7860
```

Abra `http://127.0.0.1:7860` e introduzca FICO, DTI, CLTV, tipo de interés y número de prestatarios. La pantalla devuelve la PD a 24 meses, la banda de riesgo y la advertencia académica. No guarda préstamos ni toma decisiones de crédito.

Las métricas anuales publicadas se recalculan con el mismo compacto y bundle:

```powershell
.\.venv\Scripts\python.exe -c "from pprint import pprint; from src.product import evaluate_product; pprint(evaluate_product('freddie-pd24.csv.zst','models/pd24-model.joblib',(2018,2019,2020,2021,2022)))"
.\.venv\Scripts\python.exe -c "from pprint import pprint; from src.product import evaluate_product; pprint(evaluate_product('freddie-pd24.csv.zst','models/pd24-model.joblib',(2023,)))"
```

La API Flask se puede iniciar por separado con:

```powershell
.\.venv\Scripts\python.exe -m scripts.serve_model --model models\pd24-model.joblib --port 5000
```

Con la API activa, la demostración JSON usa este contrato:

```json
{"number_of_borrowers":2,"original_interest_rate":4.5,"origination_fico":700,"original_dti":30,"original_cltv":80}
```

```powershell
$health = Invoke-RestMethod -Method Get -Uri http://127.0.0.1:5000/health
$body = '{"number_of_borrowers":2,"original_interest_rate":4.5,"origination_fico":700,"original_dti":30,"original_cltv":80}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -ContentType application/json -Body $body
```

```json
{"horizon_months":24,"model_version":"pd24-v1","risk_band":"medium","risk_score":0.005874119344598262,"warning":"Academic risk estimate; not a credit decision."}
```

## Límites

El holdout de 2023 conserva discriminación y orden de bandas, pero falla la guarda anual de intercepto de calibración (-0.6540807066 fuera de [-0.5, 0.5]); no demuestra una probabilidad absoluta estable ni autoriza ajustes posteriores. Es un prototipo académico, no una decisión de crédito. Para pérdida esperada se necesitarían EAD/LGD además de PD; no son parte de PD24.

El producto final y sus métricas 2018-2023 son reproducibles desde el compacto local. La construcción de la etiqueta desde los ficheros mensuales y la experimentación histórica 5 frente a 18/HGB se documentan en la memoria, pero no forman parte del código reducido ejecutable.
