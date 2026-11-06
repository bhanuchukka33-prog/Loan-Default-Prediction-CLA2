from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "best_model.pkl"

model = joblib.load(MODEL_PATH)

FEATURES = list(model.feature_names_in_)

app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting loan default status",
    version="1.0"
)


class LoanRequest(BaseModel):
    features: dict


@app.get("/")
def home():
    return {
        "message": "Loan Default Prediction API is running",
        "features_required": len(FEATURES)
    }


@app.get("/features")
def get_features():
    return {"features": FEATURES}


@app.post("/predict")
def predict_loan(request: LoanRequest):

    missing_features = [
        feature for feature in FEATURES
        if feature not in request.features
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required features",
                "missing_features": missing_features
            }
        )

    try:
        input_data = pd.DataFrame(
            [request.features],
            columns=FEATURES
        )

        prediction = model.predict(input_data)[0]

        result = {
            "prediction": str(prediction),
            "message": "Prediction completed successfully"
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            result["probabilities"] = {
                str(label): float(probability)
                for label, probability in zip(
                    model.classes_,
                    probabilities
                )
            }

        return result

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(error)}"
        )