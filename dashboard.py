import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Simulate a dataset for crop-linked loan risk prediction
def generate_crop_loan_dataset(n=1000):
    np.random.seed(42)
    data = []
    for _ in range(n):
        yield_index = np.random.normal(loc=0.75, scale=0.15)
        rainfall = np.random.normal(500, 150)
        soil_ph = np.random.normal(6.5, 0.5)
        price_volatility = np.random.normal(0.2, 0.1)
        loan_amount = np.random.uniform(500, 5000)
        land_size = np.random.uniform(1, 10)
        past_defaults = np.random.poisson(0.3)

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

# Train XGBoost model on synthetic data
def train_xgboost_model():
    df = generate_crop_loan_dataset()
    X = df.drop("high_risk", axis=1)
    y = df["high_risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    return model, acc, auc, X.columns, X_train

# Streamlit App
st.title("Crop-Linked Loan Risk Predictor (XGBoost Enhanced)")
st.markdown("Assess the risk level for crop-linked agricultural loans.")
st.markdown("Tap on the arrows (>>) top left to enter details.")

model, accuracy, auc_score, feature_names, X_train = train_xgboost_model()

st.sidebar.header("Enter Details")
st.sidebar.markdown("(Loan, Farm, Crop)")
loan_amount = st.sidebar.number_input("Loan Amount (€)", 10000, 500000, 25000, 1000)
land_size = st.sidebar.number_input("Land Size (hectares)", 5.0, 100.0, 10.0, 0.5)
yield_index = st.sidebar.slider("Expected Yield Index (0-1)", 0.0, 1.0, 0.75, 0.01)
rainfall = st.sidebar.slider("Annual Rainfall (mm)", 300, 800, 500)
soil_ph = st.sidebar.slider("Soil pH", 5.0, 8.0, 6.5)
price_volatility = st.sidebar.slider("Market Price Volatility", 0.0, 1.0, 0.2, 0.01)
past_defaults = st.sidebar.slider("Past Defaults (count)", 0, 10, 0)

if st.sidebar.button("Assess Risk"):
    input_data = pd.DataFrame([[
        yield_index, rainfall, soil_ph, price_volatility,
        loan_amount, land_size, past_defaults
    ]], columns=feature_names)

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Default Risk Assessment")
    st.write(f"**Risk Probability:** {prob:.2%}")
    st.write(f"**Risk Classification:** {'High Risk' if prediction else 'Low Risk'}")
    st.write(f"**Model Accuracy:** {accuracy:.2%}")
    st.write(f"**AUC Score:** {auc_score:.2f}")

    st.markdown("### Annotations")
    st.markdown("- High risk suggests loan may need additional safeguards.")
    st.markdown("- Yield index is the most critical factor influencing risk.")

    st.markdown("---")
    st.markdown("### Feature Importance")
    fig, ax = plt.subplots()
    xgb.plot_importance(model, ax=ax)
    st.pyplot(fig)

    st.markdown("### SHAP Explanation")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(input_data)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Climate and Market Stress Testing")

    # Rainfall Stress Test
    st.markdown("#### Rainfall Stress Test")
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

    # Yield Index Stress Test
    st.markdown("#### Yield Index Stress Test")
    stress_yields = np.linspace(0.4, 1.0, 20)
    yield_results = []
    for yld in stress_yields:
        test_features = np.array([[
            yld, rainfall, soil_ph, price_volatility,
            loan_amount, land_size, past_defaults
        ]])
        prob = model.predict_proba(test_features)[0][1]
        yield_results.append((yld, prob))
    yield_df = pd.DataFrame(yield_results, columns=["Yield Index", "Risk Probability"])
    st.line_chart(yield_df.set_index("Yield Index"))

    # Market Volatility Stress Test
    st.markdown("#### Market Volatility Stress Test")
    stress_volatility = np.linspace(0.0, 1.0, 20)
    vol_results = []
    for vol in stress_volatility:
        test_features = np.array([[
            yield_index, rainfall, soil_ph, vol,
            loan_amount, land_size, past_defaults
        ]])
        prob = model.predict_proba(test_features)[0][1]
        vol_results.append((vol, prob))
    vol_df = pd.DataFrame(vol_results, columns=["Market Volatility", "Risk Probability"])
    st.line_chart(vol_df.set_index("Market Volatility"))

    # Combined Multi-Factor Stress Test
    st.markdown("#### Combined Multi-Factor Stress Test")
    combined_results = []
    for rf in np.linspace(300, 700, 5):
        for yld in np.linspace(0.5, 0.9, 5):
            for vol in np.linspace(0.1, 0.5, 5):
                test_features = np.array([[
                    yld, rf, soil_ph, vol,
                    loan_amount, land_size, past_defaults
                ]])
                prob = model.predict_proba(test_features)[0][1]
                combined_results.append((rf, yld, vol, prob))
    combined_df = pd.DataFrame(combined_results, columns=["Rainfall", "Yield Index", "Volatility", "Risk Probability"])
    pivot_table = combined_df.pivot_table(index="Yield Index", columns="Rainfall", values="Risk Probability")
    st.dataframe(pivot_table.style.format("{:.2%}"))

    # Heatmap visualization
    st.markdown("#### Heatmap of Risk Probabilities")
    fig, ax = plt.subplots()
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title("Loan Risk Probability by Yield Index and Rainfall")
    st.pyplot(fig)

    st.write("This heatmap visualizes the combined effect of rainfall and yield on loan risk.")

    # Future Climate Scenario Simulation with custom inputs
    st.markdown("#### Future Climate Scenario Projection")
    st.write("Simulated projection based on user-defined future climate change parameters.")
    end_year = st.slider("Projection Horizon (years from now)", 10, 50, 30)
    rainfall_drop = st.slider("Total Rainfall Drop over Time (mm)", 0, 200, 100)
    volatility_rise = st.slider("Total Market Volatility Increase", 0.0, 0.5, 0.2, 0.01)

    years = list(range(2025, 2025 + end_year + 1))
    future_rainfall = np.linspace(rainfall, rainfall - rainfall_drop, len(years))
    future_volatility = np.linspace(price_volatility, price_volatility + volatility_rise, len(years))
    future_risk = []
    for i in range(len(years)):
        features_future = np.array([[
            yield_index, future_rainfall[i], soil_ph, future_volatility[i],
            loan_amount, land_size, past_defaults
        ]])
        risk_prob = model.predict_proba(features_future)[0][1]
        future_risk.append((years[i], risk_prob))
    future_df = pd.DataFrame(future_risk, columns=["Year", "Risk Probability"])
    st.line_chart(future_df.set_index("Year"))
    st.write("Projected increase in loan risk over time under custom climate scenario.")
