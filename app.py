from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# -------------------
# Load environment variables
# -------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------
# Initialize Flask app
# -------------------
app = Flask(__name__)

# Load saved models
rf = joblib.load("random_forest_model.pkl")   # Random Forest
arima_model = joblib.load("arima_model.pkl")  # ARIMA (baseline)

# -------------------
# Supabase helper functions
# -------------------
def save_prediction(input_data, result):
    supabase.table("predictions").insert({
        "input_data": input_data,
        "prediction": result,
        "source": "backend"
    }).execute()

def get_predictions(limit=10):
    response = supabase.table("predictions").select("*").order("created_at", desc=True).limit(limit).execute()
    return response.data

# -------------------
# Routes
# -------------------
@app.route('/')
def home():
    return "🌊 Groundwater AI API is running with Supabase!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON data
        data = request.json
        df = pd.DataFrame([data])  # convert input to dataframe

        # Random Forest prediction
        rf_pred = rf.predict(df)[0]

        # ARIMA forecast
        arima_forecast = arima_model.forecast(steps=1)[0]

        # Decision Support Layer
        if rf_pred < 1:
            level_status = "🚨 Groundwater level is critically low."
        elif 1 <= rf_pred < 3:
            level_status = "⚠️ Groundwater level is below average, monitor closely."
        else:
            level_status = "✅ Groundwater level is stable."

        if arima_forecast > rf_pred:
            recharge_status = "🌧️ Groundwater recharge is improving after recent rainfall."
        else:
            recharge_status = "⚠️ Recharge rate is low, risk of depletion ahead."

        # Response JSON
        response = {
            "rf_prediction": round(float(rf_pred), 2),
            "arima_forecast": round(float(arima_forecast), 2),
            "insights": {
                "level_status": level_status,
                "recharge_status": recharge_status
            }
        }

        # Save prediction to Supabase
        save_prediction(data, response)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/history', methods=['GET'])
def history():
    data = get_predictions(limit=10)
    return jsonify({"predictions": data})

# -------------------
# Run the app
# -------------------
if __name__ == '__main__':
    app.run(debug=True)
