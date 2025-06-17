import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# Define ideal asparagus conditions (based on agronomy research)
# ---------------------------
IDEAL_SOIL_PH = 6.5
IDEAL_TEMPERATURE = 20  # Celsius
IDEAL_RAINFALL = 500    # mm/year

# ---------------------------
# Simulated environmental data (mocked based on lat/lon)
# ---------------------------
def generate_environmental_features(lat, lon):
    np.random.seed(int((lat + lon) * 1000) % 1000)  # Consistent randomness
    soil_ph = np.clip(6 + 0.5 * np.sin(lat * lon), 5.5, 7.5)
    temperature = np.clip(15 + 10 * np.cos(lat / 10), 10, 25)
    rainfall = np.clip(400 + 150 * np.sin(lon / 20), 300, 700)
    return soil_ph, temperature, rainfall

# ---------------------------
# Generate synthetic dataset for ML model
# ---------------------------
def generate_synthetic_dataset(n=1000):
    data = []
    for _ in range(n):
        lat = np.random.uniform(-90, 90)
        lon = np.random.uniform(-180, 180)
        soil_ph, temp, rain = generate_environmental_features(lat, lon)

        score = (
            - abs(soil_ph - IDEAL_SOIL_PH)
            - abs(temp - IDEAL_TEMPERATURE) / 5
            - abs(rain - IDEAL_RAINFALL) / 100
        )
        prob = 1 / (1 + np.exp(-score))
        suitable = int(prob > 0.6)

        data.append([lat, lon, soil_ph, temp, rain, suitable])
    
    df = pd.DataFrame(data, columns=["latitude", "longitude", "soil_ph", "temperature", "rainfall", "suitable"])
    return df

# ---------------------------
# Train the ML model
# ---------------------------
def train_model():
    df = generate_synthetic_dataset()
    X = df[["latitude", "longitude", "soil_ph", "temperature", "rainfall"]]
    y = df["suitable"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Asparagus Suitability Checker")
st.markdown("Check if a location is suitable for growing asparagus based on GPS coordinates.")

lat = st.number_input("Enter Latitude (-90 to 90)", min_value=-90.0, max_value=90.0, value=40.0)
lon = st.number_input("Enter Longitude (-180 to 180)", min_value=-180.0, max_value=180.0, value=-75.0)

if st.button("Check Suitability"):
    with st.spinner("Analyzing environmental conditions and checking suitability..."):
        soil_ph, temp, rain = generate_environmental_features(lat, lon)
        model, accuracy = train_model()

        features = np.array([[lat, lon, soil_ph, temp, rain]])
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][0]

        st.success(f"Suitability Score: {prob:.2%} ({'Suitable' if prediction else 'Suitable'} for Asparagus)")
        st.markdown("### Environmental Conditions")
        st.write(f"- Estimated Soil pH: {soil_ph:.2f}")
        st.write(f"- Average Temperature: {temp:.2f} °C")
        st.write(f"- Annual Rainfall: {rain:.2f} mm")
        st.markdown(f"**Model Accuracy**: {accuracy:.2%} (based on synthetic training data)")
