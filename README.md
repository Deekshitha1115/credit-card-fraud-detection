# 💳 Credit Card Fraud Detection Using Machine Learning

A machine learning project that detects whether a credit card transaction is **legitimate or potentially fraudulent** using Logistic Regression.

The project includes data preprocessing, feature scaling, model training, evaluation, model saving, and a Streamlit-based interface for making individual transaction predictions.

---

## 📌 Project Overview

Credit card fraud is an important problem in the financial sector. A large number of transactions are legitimate, while only a small portion may be fraudulent.

This project uses Machine Learning to classify transactions into two categories:

- **0 → Normal Transaction**
- **1 → Fraudulent Transaction**

The project uses **Logistic Regression** as the classification algorithm and **StandardScaler** for feature scaling.

A Streamlit web application is also included so that users can enter transaction details and receive a prediction.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze credit card transaction data
- Separate transaction features from the target variable
- Preprocess and scale the input features
- Train a Logistic Regression classification model
- Evaluate the model using a confusion matrix and classification report
- Save the trained model and scaler for future predictions
- Build an interactive Streamlit application for fraud prediction

---

## 🗂️ Dataset

The project uses a credit card transaction dataset containing anonymized transaction features.

The dataset contains the following main columns:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

### Target Variable

The `Class` column is the target variable:

| Class | Meaning |
|------:|---------|
| 0 | Normal Transaction |
| 1 | Fraudulent Transaction |

The `V1` to `V28` features are anonymized numerical features from the dataset.

---

## 🔄 Machine Learning Workflow

The project follows this workflow:

```text
Credit Card Transaction Dataset
            ↓
       Load Dataset
            ↓
   Separate Features & Target
            ↓
       Train-Test Split
            ↓
      Feature Scaling
            ↓
    Logistic Regression
            ↓
       Model Prediction
            ↓
       Model Evaluation
            ↓
   Save Model & Scaler
            ↓
      Streamlit Application
            ↓
     Fraud / Normal Result
---

## 🧠 Model Used

### Logistic Regression

Logistic Regression is used as the classification algorithm to predict whether a transaction is normal or fraudulent.

The model is trained using the transaction features after applying feature scaling.

### Feature Scaling

`StandardScaler` is used to standardize the input features before training and prediction.

---

## 📊 Model Evaluation

The model is evaluated using:

- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-Score

These metrics are useful for fraud detection because fraudulent transactions are usually much fewer than normal transactions.

---

## 💾 Model Saving

After training, the trained model and scaler are saved using Joblib.

```text
fraud_model.pkl
scaler.pkl