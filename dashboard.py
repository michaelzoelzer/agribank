import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim

# Constants: ideal asparagus ranges
IDEAL = {
    "temp": (18, 30),
    "rain": (400, 800),
    "sun": 6,
    "ph": (6.5, 7.5),
    "sand_frac": 0.4,  # sandy loam ~40% sand
}

# Fetch soil data from SoilGrids REST API
def get_soil(lat, lon):
    url = (
        "https://rest.isric.org/soilgrids/v2.0/properties/query"
        f"?lat={lat}&lon={lon}"
        "&property=phh2o&property=sand&property=clay"
        "&depth=0-30cm&value=mean"
    )
    r = requests.get(url)
    r.raise_for_status()
    d = r.json()["properties"]
    return {
        "ph": d["phh2o"]["mean"],
        "sand": d["sand"]["mean"],
        "clay": d["clay"]["mean"],
    }

# Fetch weather/climate data from Open-Meteo
def get_weather(lat, lon):
    # 7-day forecast + radiation
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,precipitation,radiation_sum"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,sunrise,sunset"
        "&current_weather=true&timezone=auto&forecast_days=7"
    )
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    # Annualize precipitation & approximate seasonal mean temp
    daily = data["daily"]
    yearly_rain = sum(daily["precipitation_sum"]) * 365/7
    avg_temp = np.mean([(a + b)/2 for a, b in zip(daily["temperature_2m_max"], daily["temperature_2m_min"])])
    # Sunshine hours via daylight durations
    sun_hours = np.mean([
        (np.fromisoformat(s_out) - np.fromisoformat(s_in)).astype('timedelta64[h]').astype(int)
        for s_in, s_out in zip(daily["sunrise"], daily["sunset"])
    ])
    return {"avg_temp": avg_temp, "yearly_rain": yearly_rain, "sun_hours": sun_hours}

# Scoring function
def score(env):
    s = 0
    # Temperature
    if IDEAL["temp"][0] <= env["avg_temp"] <= IDEAL["temp"][1]: s += 1
    # Rain
    if IDEAL["rain"][0] <= env["yearly_rain"] <= IDEAL["rain"][1]: s += 1
    # Sunshine
    if env["sun_hours"] >= IDEAL["sun"]: s += 1
    # Soil pH
    if IDEAL["ph"][0] <= env["ph"] <= IDEAL["ph"][1]: s += 1
    # Texture: sandy loam
    sand = env["sand"] / (env["sand"] + env["clay"] + 1e-6)
    if abs(sand - IDEAL["sand_frac"]) <= 0.1: s += 1
    return round((s/5)*100, 2)

# Streamlit UI
st.title("🌱 Asparagus Yield Suitability Estimator")
loc = st.text_input("Enter location (address or city):")
if st.button("Estimate"):
    if not loc: st.error("Please enter a location"); st.stop()
    geo = Nominatim(user_agent="asparagus_app").geocode(loc)
    if not geo: st.error("Location not found"); st.stop()
    st.success(f"Found: {geo.address}")
    lat, lon = geo.latitude, geo.longitude
    st.map({"lat":[lat],"lon":[lon]})
    with st.spinner("Fetching soil data..."):
        soil = get_soil(lat, lon)
    with st.spinner("Fetching weather data..."):
        weather = get_weather(lat, lon)
    env = {**soil, **weather}
    st.subheader("📊 Environmental Data")
    st.json(env)
    prob = score(env)
    st.subheader("🌿 Yield Probability")
    st.metric("Suitability Score", f"{prob} %")
    if prob >= 80: st.success("Excellent conditions!")
    elif prob >= 50: st.info("Moderate – may need adjustments.")
    else: st.warning("Not ideal – consider amendments.")

