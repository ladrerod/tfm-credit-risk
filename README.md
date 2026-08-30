# PD24

Producto académico local para estimar la probabilidad de impago hipotecario a 24 meses con cinco variables de originación. Requiere el único input local `freddie-pd24.csv.zst` en la raíz del proyecto; el fichero está ignorado por Git.

## Entorno

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
```

## Entrenar y servir

```powershell
.\.venv\Scripts\python.exe -m scripts.train_model --data freddie-pd24.csv.zst --output models\pd24-model.joblib
.\.venv\Scripts\python.exe -m scripts.serve_model --model models\pd24-model.joblib --port 5000
```

En otra consola, comprueba el servicio y envía las cinco entradas requeridas:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
$body = @{ origination_fico = 700; original_dti = 30; original_cltv = 80; original_interest_rate = 4.5; number_of_borrowers = 2 } | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -ContentType application/json -Body $body
```

## Verificación y memoria

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd tfm
tectonic main.tex
```

## Límites

El resultado es académico y no es una decisión de crédito. El alcance no cubre EAD ni LGD.
