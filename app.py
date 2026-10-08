# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details below:")

# Create input fields
Time = st.number_input("Time", value=0.0)
Amount = st.number_input("Amount", value=0.0)

# V1 to V28 inputs
features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    features.append(val)

# Combine all inputs
if st.button("Predict"):
    input_data = [Time] + features + [Amount]
    
    # Convert to DataFrame
    df = pd.DataFrame([input_data], columns=[
        "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
        "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
        "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
    ])

    # Scale input
    df_scaled = scaler.transform(df)

    # Prediction
    prediction = model.predict(df_scaled)

    # Output
    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Normal Transaction")