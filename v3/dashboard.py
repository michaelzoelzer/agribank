import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

# ------------------ Load Trained Model ------------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost_asparagus_yield_model.joblib")

model = load_model()

# ------------------ Constants ------------------
IDEAL = {
    "temp": (18, 30),
    "rain": (400, 800),
    "sun": 6,
    "ph": (6.5, 7.5),
    "sand_frac": 0.4,
}

# ------------------ Fetch Soil Data ------------------
#def get_soil(lat, lon):
#    url = (
#        f"https://rest.isric.org/soilgrids/v2.0/properties/query?lat={lat}&lon={lon}"
#        "&property=phh2o&property=sand&property=clay&depth=0-30cm&value=mean"
#    )
#    r = requests.get(url)
#    r.raise_for_status()
#    d = r.json()["properties"]
#    return {
#        "ph": d["phh2o"]["mean"],
#        "sand": d["sand"]["mean"],
#        "clay": d["clay"]["mean"],
#    }

# ------------------ Fetch Weather Data ------------------
def get_weather(lat, lon):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,sunrise,sunset"
        "&timezone=auto&forecast_days=7"
    )
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["daily"]
    yearly_rain = sum(data["precipitation_sum"]) * 365 / 7
    avg_temp = np.mean([(a + b) / 2 for a, b in zip(data["temperature_2m_max"], data["temperature_2m_min"])])
    sun_hours = np.mean([
        (pd.to_datetime(s_out) - pd.to_datetime(s_in)).total_seconds() / 3600
        for s_in, s_out in zip(data["sunrise"], data["sunset"])
    ])
    return {"avg_temp": avg_temp, "yearly_rain": yearly_rain, "sun_hours": sun_hours}

# ------------------ Yield Suitability Score ------------------
def score(env):
    s = 0
    if IDEAL["temp"][0] <= env["avg_temp"] <= IDEAL["temp"][1]: s += 1
    if IDEAL["rain"][0] <= env["yearly_rain"] <= IDEAL["rain"][1]: s += 1
    if env["sun_hours"] >= IDEAL["sun"]: s += 1
    if IDEAL["ph"][0] <= env["ph"] <= IDEAL["ph"][1]: s += 1
    sand = env["sand"] / (env["sand"] + env["clay"] + 1e-6)
    if abs(sand - IDEAL["sand_frac"]) <= 0.1: s += 1
    return round((s / 5) * 100, 2)

# ------------------ ML Yield Prediction ------------------
def predict_yield_from_features(features_dict):
    input_df = pd.DataFrame([features_dict])[['avg_temp', 'annual_rainfall', 'sun_hours', 'soil_ph', 'sand_percent', 'clay_percent']]
    prediction = model.predict(input_df)[0]
    return round(prediction, 2)

# ------------------ Yield Curve ------------------
def simulate_yield_curve(base_yield):
    years = np.arange(1, 11)
    curve = []
    for year in years:
        if year == 1: factor = 0.3
        elif year == 2: factor = 0.7
        elif 3 <= year <= 5: factor = 1.0
        elif 6 <= year <= 8: factor = 0.95
        else: factor = 0.9
        yield_value = round(base_yield * factor, 2)
        curve.append(yield_value)
    return pd.DataFrame({'Year': years, 'Yield (tons/ha)': curve})

# ------------------ Streamlit App ------------------
st.title("🌱 Asparagus Yield Prediction App")
#loc = st.text_input("Enter location (address or city):")
#if st.button("Estimate"):
#    geo = Nominatim(user_agent="asparagus_app").geocode(loc)
#    if not geo:
#        st.error("Location not found")
#        st.stop()
#    lat, lon = geo.latitude, geo.longitude
st.subheader("📍 Set Your Location")
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", value=51.0, format="%.6f")
with col2:
    lon = st.number_input("Longitude", value=0.0, format="%.6f")

if st.button("Estimate"):
    if lat == 0.0 and lon == 0.0:
        st.warning("Please provide valid coordinates.")
        st.stop()
    st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))

    with st.spinner("Fetching environmental data..."):
        soil = get_soil(lat, lon)
        weather = get_weather(lat, lon)
    env = {**soil, **weather}

    st.subheader("📊 Environmental Data")
    st.json(env)

    prob = score(env)
    st.subheader("🌿 Suitability Score")
    st.metric("Estimated Suitability", f"{prob} %")

    features_for_model = {
        "avg_temp": env["avg_temp"],
        "annual_rainfall": env["yearly_rain"],
        "sun_hours": env["sun_hours"],
        "soil_ph": env["ph"],
        "sand_percent": env["sand"],
        "clay_percent": env["clay"],
    }

    predicted_yield = predict_yield_from_features(features_for_model)
    st.subheader("🤖 ML-Based Yield Prediction")
    st.metric("Estimated Yield (tons/ha)", f"{predicted_yield}")

    st.subheader("📈 Multi-Season Yield Projection")
    df_curve = simulate_yield_curve(predicted_yield)
    fig, ax = plt.subplots()
    ax.plot(df_curve["Year"], df_curve["Yield (tons/ha)"], marker='o')
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (tons/ha)")
    ax.set_title("Asparagus Yield Over 10 Seasons")
    ax.grid(True)
    st.pyplot(fig)
