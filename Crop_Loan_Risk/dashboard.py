import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate a dataset for crop-linked loan risk prediction
def generate_crop_loan_dataset(n=1000):
    np.random.seed(42)
    data = []
    for _ in range(n):
        yield_index = np.random.normal(loc=0.75, scale=0.15)  # proxy for crop yield (0-1)
        rainfall = np.random.normal(500, 150)  # mm
        soil_ph = np.random.normal(6.5, 0.5)
        price_volatility = np.random.normal(0.2, 0.1)  # proxy for market risk
        loan_amount = np.random.uniform(500, 5000)
        land_size = np.random.uniform(1, 10)  # hectares
        past_defaults = np.random.poisson(0.3)

        # Define risk (1 = high risk/default, 0 = low risk)
        risk_score = (
            - 3 * yield_index
            + 0.002 * (500 - rainfall)
            + abs(soil_ph - 6.5)
            + 2 * price_volatility
            + 0.0005 * loan_amount
            + 0.3 * past_defaults
        )
        label = int(risk_score > 1.5)

        data.append([
            yield_index, rainfall, soil_ph, price_volatility,
            loan_amount, land_size, past_defaults, label
        ])

    columns = [
        "yield_index", "rainfall", "soil_ph", "price_volatility",
        "loan_amount", "land_size", "past_defaults", "high_risk"
    ]
    return pd.DataFrame(data, columns=columns)

# Train model on the synthetic data
def train_risk_model():
    df = generate_crop_loan_dataset()
    X = df.drop("high_risk", axis=1)
    y = df["high_risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc, X.columns

# Streamlit App
st.title("Crop-Linked Loan Risk Predictor")
st.markdown("Assess the risk level for crop-linked agricultural loans.")

model, accuracy, feature_names = train_risk_model()

st.sidebar.header("Enter Farm & Loan Details")
yield_index = st.sidebar.slider("Expected Yield Index (0-1)", 0.0, 1.0, 0.75, 0.01)
rainfall = st.sidebar.slider("Annual Rainfall (mm)", 300, 800, 500)
soil_ph = st.sidebar.slider("Soil pH", 5.0, 8.0, 6.5)
price_volatility = st.sidebar.slider("Market Price Volatility", 0.0, 1.0, 0.2, 0.01)
loan_amount = st.sidebar.number_input("Loan Amount ($)", 500, 10000, 2500)
land_size = st.sidebar.number_input("Land Size (hectares)", 0.5, 20.0, 5.0)
past_defaults = st.sidebar.slider("Past Defaults (count)", 0, 10, 0)

if st.sidebar.button("Assess Risk"):
    features = np.array([[
        yield_index, rainfall, soil_ph, price_volatility,
        loan_amount, land_size, past_defaults
    ]])
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.subheader("Loan Risk Assessment")
    st.write(f"**Risk Probability:** {prob:.2%}")
    st.write(f"**Risk Classification:** {'High Risk' if prediction else 'Low Risk'}")
    st.write(f"**Model Accuracy:** {accuracy:.2%}")

    st.markdown("### Explanation")
    st.markdown("- High risk suggests loan may need additional safeguards.")
    st.markdown("- Yield index is the most critical factor influencing risk.")

    st.markdown("---")
    st.subheader("Climate Stress Testing")
    st.markdown("Stress testing loan risk under different rainfall scenarios.")

    stress_rainfalls = np.linspace(200, 800, 20)
    stress_results = []
    for rf in stress_rainfalls:
        test_features = np.array([[
            yield_index, rf, soil_ph, price_volatility,
            loan_amount, land_size, past_defaults
        ]])
        prob = model.predict_proba(test_features)[0][1]
        stress_results.append((rf, prob))

    stress_df = pd.DataFrame(stress_results, columns=["Rainfall", "Risk Probability"])
    st.line_chart(stress_df.set_index("Rainfall"))
    st.write("This graph shows how varying rainfall affects loan risk.")
