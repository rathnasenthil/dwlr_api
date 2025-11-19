"# dwlr_api" 

Lightweight Flask API for predicting groundwater / domestic water-level risk using pre-trained models (Random Forest & ARIMA) with optional Supabase logging and CORS enabled.

This README was generated specifically from the app.py you provided and documents the exact endpoints, env vars, and behavior implemented there. If you change app.py, update this README.

Features

Flask API serving two model outputs from loaded .pkl files:

random_forest_model.pkl — used via rf.predict(df)

arima_model.pkl — used via arima_model.forecast(steps=1)

Simple decision-support text returned alongside numeric predictions.

Optional persistence of predictions to a Supabase table (predictions) if service key and URL are provided via environment variables.

CORS enabled for all origins (easy to restrict for production).

Health/home route and a history route to list recent predictions saved to Supabase.

Repository layout
.
├── .gitignore
├── .ipynb_checkpoints/
├── app.py
├── arima_model.pkl
├── random_forest_model.pkl
├── DWLR_AI_Prototype.ipynb
├── DWLR_Dataset_2023.csv
├── Procfile
├── requirements.txt
└── README.md        <-- (this file)

Environment variables

Create a .env file (and/or set env vars in your deployment platform) with:

SUPABASE_URL=<your-supabase-url>                # e.g. https://xyzcompany.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>


If both are present, app will create a Supabase client and attempt to save predictions to table predictions.

If missing, the app will still run but will print a warning and skip saving.

For local development you can install python-dotenv (already expected in requirements.txt) and create .env. The code attempts to load_dotenv().

Supabase table schema (recommended)

If you want to store predictions in Supabase, create a table predictions. Suggested columns:

id — bigint / auto-increment primary key

created_at — timestamp with time zone default now()

input_data — jsonb

prediction — jsonb

source — text (e.g. "backend")

This matches how save_prediction(input_data, result) inserts objects.

API Endpoints
GET /

Health / home

Response:

🌊 Groundwater AI API is running with Supabase + CORS enabled!

POST /predict

Make a prediction. The implementation simply converts request.json to a single-row pandas.DataFrame and passes that to the RandomForest model. It also calls arima_model.forecast(steps=1) for a one-step forecast.

Request Content-Type: application/json

Body: any JSON object whose keys match the features the RandomForest expects (i.e. same columns used when training).

Example request (example feature names — adapt to your model's expected features):

curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "hour": 14,
        "temperature": 28.5,
        "day_of_week": 2,
        "is_holiday": false,
        "other_feature": 1.23
      }'


Example response:

{
  "rf_prediction": 2.35,
  "arima_forecast": 2.48,
  "insights": {
    "level_status": "⚠️ Groundwater level is below average, monitor closely.",
    "recharge_status": "🌧️ Groundwater recharge is improving after recent rainfall."
  }
}


Notes:

rf_prediction is produced by rf.predict(df)[0] and rounded to 2 decimals.

arima_forecast is produced by arima_model.forecast(steps=1)[0] and rounded to 2 decimals.

The code expects the RandomForest model to accept a pandas.DataFrame with a single row.

If model input columns don't match the JSON keys, prediction will raise an exception — the app returns {"error": "..."}

save_prediction(data, response) will attempt to insert into Supabase if configured.

GET /history

Returns the last saved predictions from Supabase. If Supabase isn't configured it returns {"predictions": []}.

Request:

curl http://127.0.0.1:5000/history


Response example:

{
  "predictions": [
    {
      "id": 123,
      "input_data": {"hour":14,...},
      "prediction": {"rf_prediction":2.35, ...},
      "source":"backend",
      "created_at":"2025-11-01T12:34:56Z"
    },
    ...
  ]
}

Running locally

Create virtualenv and install deps:

python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt


Ensure models exist in repo root:

random_forest_model.pkl

arima_model.pkl

(Optional) Create .env with Supabase vars.

Start app:

python app.py


By default the app runs with debug=True and will listen on 127.0.0.1:5000.

Production deployment

Procfile is present — typical content for Gunicorn:

web: gunicorn app:app


Ensure requirements.txt includes gunicorn and supabase (and python-dotenv if you rely on .env locally).

Restrict CORS in production:

CORS(app, origins=["https://your-frontend.example.com"])


For ARIMA model compatibility:

arima_model.forecast(steps=1) is supported by pmdarima/statsmodels variants. If your .pkl is a different object ensure it has compatible API or adapt app.py.

Example Python client
import requests

url = "http://127.0.0.1:5000/predict"
payload = {
    "hour": 14,
    "temperature": 28.5,
    "day_of_week": 2,
    "is_holiday": False,
    "other_feature": 1.23
}
r = requests.post(url, json=payload)
print(r.json())

Troubleshooting & tips

Model input mismatch — If you get errors related to missing columns, verify the feature names and order your JSON keys to match the DataFrame columns the model was trained with. Inspect the training notebook (DWLR_AI_Prototype.ipynb) or the original training script to confirm feature names.

ARIMA errors — if arima_model.forecast(steps=1) raises an error, check whether the ARIMA object is from pmdarima or statsmodels; adapt app.py to call the appropriate forecasting method (predict, get_forecast, etc.).

Supabase auth errors — ensure you use the SERVICE_ROLE key if writing to DB from the backend, and that the key is kept secret (use environment variables rather than committing to Git).

CORS — set stricter origins in production to avoid open access.

Large models — if .pkl files are large, consider loading from a remote artifact store (S3, cloud storage, Supabase storage) instead of committing large binaries to the repo.

How to retrain models

Open DWLR_AI_Prototype.ipynb and follow the training cells.

When saving, prefer joblib.dump(model, "random_forest_model.pkl") for sklearn models.

For ARIMA, if using pmdarima:

import joblib
joblib.dump(arima_model, "arima_model.pkl")


Replace .pkl files in the repo or update the code to load from an artifact store.
