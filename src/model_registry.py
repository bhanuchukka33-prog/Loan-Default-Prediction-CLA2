import os
import json
import joblib
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score

# Project paths
MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Loan_Default_Classification")

MODEL_NAME = "Loan_Default_Model"

# Load trained model
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Register model
with mlflow.start_run(run_name="Lab6_Model_Registry") as run:

    model_info = mlflow.sklearn.log_model(
    sk_model=model,
    name="loan_model",
    serialization_format="pickle"
)

    model_uri = model_info.model_uri

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print("Model registered successfully!")
    print("Model Name:", MODEL_NAME)
    print("Model Version:", registered_model.version)

# Assign production alias
client = MlflowClient()

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="production",
    version=registered_model.version
)

print("Production alias assigned successfully!")

# Load production model
production_model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}@production"
)

print("Production model loaded successfully!")
print("Lab 6 Registry Completed!")