import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("Loading dataset...")
df = pd.read_csv("creditcard.csv")

# Features & target
X = df.drop("Class", axis=1)
y = df["Class"]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Scaling data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 🔥 SAVE FILES (VERY IMPORTANT)
joblib.dump(model, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model and scaler saved successfully!")