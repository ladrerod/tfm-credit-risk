# PD24

PD24 es un producto académico local que estima el riesgo de impago hipotecario a 24 meses con cinco variables de originación. Necesita un único input local: `freddie-pd24.csv.zst` en la raíz; el compacto está ignorado por Git.

## Ejecutar

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m scripts.train_model --data freddie-pd24.csv.zst --output models\pd24-model.joblib
.\.venv\Scripts\python.exe -m scripts.serve_model --model models\pd24-model.joblib --port 5000
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Con el servicio activo, la demostración real usa este contrato JSON:

```json
{"number_of_borrowers":2,"original_interest_rate":4.5,"origination_fico":700,"original_dti":30,"original_cltv":80}
```

```powershell
$body = '{"number_of_borrowers":2,"original_interest_rate":4.5,"origination_fico":700,"original_dti":30,"original_cltv":80}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -ContentType application/json -Body $body
```

```json
{"horizon_months":24,"model_version":"pd24-v1","risk_band":"medium","risk_score":0.005874119344598262,"warning":"Academic risk estimate; not a credit decision."}
```

## Límites

El holdout de 2023 conserva discriminación y orden de bandas, pero falla la guarda anual de intercepto de calibración (-0.6540807066 fuera de [-0.5, 0.5]); no demuestra una probabilidad absoluta estable ni autoriza ajustes posteriores. Es un prototipo académico, no una decisión de crédito. Para pérdida esperada se necesitarían EAD/LGD además de PD; no son parte de PD24.
