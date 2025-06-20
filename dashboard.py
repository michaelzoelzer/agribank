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

model, accuracy, auc_score, feature_names, X_train = train_xgboost_model()

st.sidebar.header("Enter Farm & Loan Details")
yield_index = st.sidebar.slider("Expected Yield Index (0-1)", 0.0, 1.0, 0.75, 0.01)
rainfall = st.sidebar.slider("Annual Rainfall (mm)", 300, 800, 500)
soil_ph = st.sidebar.slider("Soil pH", 5.0, 8.0, 6.5)
price_volatility = st.sidebar.slider("Market Price Volatility", 0.0, 1.0, 0.2, 0.01)
loan_amount = st.sidebar.number_input("Loan Amount ($)", 500, 10000, 2500)
land_size = st.sidebar.number_input("Land Size (hectares)", 0.5, 20.0, 5.0)
past_defaults = st.sidebar.slider("Past Defaults (count)", 0, 10, 0)

if st.sidebar.button("Assess Risk"):
    input_data = pd.DataFrame([[
        yield_index, rainfall, soil_ph, price_volatility,
        loan_amount, land_size, past_defaults
    ]], columns=feature_names)

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Loan Risk Assessment")
    st.write(f"**Risk Probability:** {prob:.2%}")
    st.write(f"**Risk Classification:** {'High Risk' if prediction else 'Low Risk'}")
    st.write(f"**Model Accuracy:** {accuracy:.2%}")
    st.write(f"**AUC Score:** {auc_score:.2f}")

    st.markdown("### Feature Importance")
    fig, ax = plt.subplots()
    xgb.plot_importance(model, ax=ax)
    st.pyplot(fig)

    st.markdown("### SHAP Explanation")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(input_data)
#    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.plots.waterfall(shap_values[0], max_display=10)
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [1, 2, 3])
    # other plotting actions...
    st.pyplot(fig)
#    st.pyplot(bbox_inches='tight')
