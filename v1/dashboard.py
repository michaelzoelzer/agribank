
import streamlit as st
import requests
import numpy as np
import pickle

# Load pre-trained models (replace with your actual model paths)
# with open('rf_model.pkl', 'rb') as f:
#     rf_model = pickle.load(f)
# with open('xgb_model.pkl', 'rb') as f:
#     xgb_model = pickle.load(f)

# Simulated model for example
class DummyModel:
    def predict(self, X):
        return [112000 - (X[0][0] * 5) + (X[0][1] * 5000)]

rf_model = DummyModel()
xgb_model = DummyModel()

st.title("Asparagus Yield Forecast Dashboard")
st.subheader("Location: Bruchsal, Germany")

st.markdown("### Step 1: Retrieve Forecasted Weather Data")
if st.button("Get Weather Forecast"):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 49.12,
        "longitude": 8.6,
        "daily": "precipitation_sum,temperature_2m_max",
        "start_date": "2025-06-13",
        "end_date": "2025-09-13",
        "timezone": "Europe/Berlin"
    }
    response = requests.get(url, params=params)
    data = response.json()

    rain_forecast = sum(data['daily']['precipitation_sum'])
    heat_days = sum(t > 30 for t in data['daily']['temperature_2m_max'])

    st.write(f"Total Rainfall Forecast (June–Sept): {rain_forecast:.1f} mm")
    st.write(f"Number of Heat Days (>30°C): {heat_days}")

    st.session_state['rain'] = rain_forecast

st.markdown("### Step 2: NDVI Anomaly (Simulated Input)")
ndvi_anomaly = st.slider("NDVI Anomaly (approximate)", -0.1, 0.1, -0.03, 0.01)
st.session_state['ndvi'] = ndvi_anomaly

st.markdown("### Step 3: Predict Yield")
if 'rain' in st.session_state and 'ndvi' in st.session_state:
    X_input = np.array([[st.session_state['rain'], st.session_state['ndvi']]])
    rf_yield = rf_model.predict(X_input)[0]
    xgb_yield = xgb_model.predict(X_input)[0]

    st.success(f"Random Forest Forecast: {rf_yield:.0f} tons")
    st.success(f"XGBoost Forecast: {xgb_yield:.0f} tons")
else:
    st.info("Please complete steps 1 and 2 first.")
