from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load saved models
rf = joblib.load("random_forest_model.pkl")   # Random Forest
arima_model = joblib.load("arima_model.pkl")  # ARIMA (baseline)

@app.route('/')
def home():
    return "🌊 Groundwater AI API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON data
        data = request.json
        df = pd.DataFrame([data])  # convert input to dataframe

        # Random Forest prediction (recharge estimation)
        rf_pred = rf.predict(df)[0]

        # For ARIMA: forecast 1 step ahead
        arima_forecast = arima_model.forecast(steps=1)[0]

        # --- Decision Support Layer ---
        # Groundwater level status
        if rf_pred < 1:
            level_status = "🚨 Groundwater level is critically low."
        elif 1 <= rf_pred < 3:
            level_status = "⚠️ Groundwater level is below average, monitor closely."
        else:
            level_status = "✅ Groundwater level is stable."

        # Recharge status (compare ARIMA forecast vs RF)
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
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
