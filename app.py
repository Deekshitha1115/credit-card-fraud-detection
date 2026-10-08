# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# Title
st.title("💳 Credit Card Fraud Detection")

st.write(
    "This application uses Machine Learning to predict whether a "
    "credit card transaction is normal or fraudulent."
)

st.divider()

# Basic transaction details
st.subheader("📋 Transaction Details")

col1, col2 = st.columns(2)

with col1:
    Time = st.number_input(
        "Transaction Time",
        value=0.0
    )

with col2:
    Amount = st.number_input(
        "Transaction Amount",
        value=0.0,
        min_value=0.0
    )

# V1 to V28
st.subheader("🔢 Transaction Features")

st.write("Enter the values for V1 to V28:")

features = []

col1, col2 = st.columns(2)

for i in range(1, 29):

    if i <= 14:
        with col1:
            val = st.number_input(
                f"V{i}",
                value=0.0,
                key=f"v{i}"
            )
    else:
        with col2:
            val = st.number_input(
                f"V{i}",
                value=0.0,
                key=f"v{i}"
            )

    features.append(val)

st.divider()

# Prediction button
if st.button("🔍 Predict Transaction", use_container_width=True):

    # Combine all inputs
    input_data = [Time] + features + [Amount]

    # Feature names
    columns = [
        "Time",
        "V1", "V2", "V3", "V4", "V5",
        "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15",
        "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25",
        "V26", "V27", "V28",
        "Amount"
    ]

    # Convert input into DataFrame
    df = pd.DataFrame(
        [input_data],
        columns=columns
    )

    # Scale input
    df_scaled = scaler.transform(df)

    # Prediction
    prediction = model.predict(df_scaled)

    # Prediction probability
    probability = model.predict_proba(df_scaled)

    fraud_probability = probability[0][1] * 100

    # Display result
    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:

        st.error("⚠️ Fraudulent Transaction Detected!")

        st.write(
            f"Fraud Probability: **{fraud_probability:.2f}%**"
        )

    else:

        st.success("✅ Normal Transaction")

        st.write(
            f"Fraud Probability: **{fraud_probability:.2f}%**"
        )