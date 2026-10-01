
import os
import joblib
import pandas as pd

print("=" * 60)
print("LAB 5 - OUTPUT VALIDATION")
print("=" * 60)

required_files = [
    "models/preprocessor.pkl",
    "outputs/X_train_processed.csv",
    "outputs/X_test_processed.csv"
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Missing file: {file}")

print("[SUCCESS] Required files exist!")

X_train = pd.read_csv("outputs/X_train_processed.csv")
X_test = pd.read_csv("outputs/X_test_processed.csv")

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError("Feature dimensions do not match!")

if X_train.isnull().values.any():
    raise ValueError("Missing values found in training output!")

if X_test.isnull().values.any():
    raise ValueError("Missing values found in testing output!")

preprocessor = joblib.load("models/preprocessor.pkl")

print("\nPreprocessor loaded successfully!")
print("[SUCCESS] Output Validation PASSED!")
