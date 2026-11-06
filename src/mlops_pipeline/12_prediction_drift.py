
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = ROOT / "data" / "Loan_Default.csv"
EVOLVED_PATH = ROOT / "data" / "monitoring" / "drift" / "evolved_data.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
REPORT_PATH = ROOT / "outputs" / "drift" / "prediction_drift_report.csv"

RANDOM_SEED = 42
KS_THRESHOLD = 0.10


def load_data():
    df = pd.read_csv(SOURCE_PATH)
    df = df.dropna(subset=["Status"])

    X = df.drop(columns=["ID", "Status"])
    y = df["Status"].astype(int)

    _, baseline, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    baseline = baseline.reset_index(drop=True)
    evolved = pd.read_csv(EVOLVED_PATH)

    if baseline.shape != evolved.shape:
        raise ValueError(
            f"Dataset shape mismatch: "
            f"baseline={baseline.shape}, evolved={evolved.shape}"
        )

    return baseline, evolved


def predict(model, data):
    predictions = model.predict(data)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)[:, 1]
    else:
        raise ValueError(
            "The saved model does not support predict_proba."
        )

    return predictions, probabilities


def summarize_predictions(name, predictions, probabilities):
    total = len(predictions)

    default_count = int((predictions == 1).sum())
    non_default_count = int((predictions == 0).sum())

    print()
    print(f"{name} PREDICTIONS")
    print("-" * 60)

    print(f"Total predictions : {total}")
    print(
        f"Default (1)       : {default_count} "
        f"({default_count / total * 100:.2f}%)"
    )
    print(
        f"Non-default (0)   : {non_default_count} "
        f"({non_default_count / total * 100:.2f}%)"
    )
    print(f"Mean probability  : {probabilities.mean():.6f}")
    print(f"Median probability: {np.median(probabilities):.6f}")
    print(f"Std probability   : {probabilities.std():.6f}")
    print(f"Min probability   : {probabilities.min():.6f}")
    print(f"Max probability   : {probabilities.max():.6f}")


def analyze_prediction_drift(
    baseline_predictions,
    evolved_predictions,
    baseline_probabilities,
    evolved_probabilities,
):
    prediction_ks, prediction_p = ks_2samp(
        baseline_predictions,
        evolved_predictions,
    )

    probability_ks, probability_p = ks_2samp(
        baseline_probabilities,
        evolved_probabilities,
    )

    return {
        "prediction_distribution_ks": float(prediction_ks),
        "prediction_distribution_p_value": float(prediction_p),
        "prediction_distribution_status": (
            "DRIFT" if prediction_ks >= KS_THRESHOLD else "OK"
        ),
        "probability_distribution_ks": float(probability_ks),
        "probability_distribution_p_value": float(probability_p),
        "probability_distribution_status": (
            "DRIFT" if probability_ks >= KS_THRESHOLD else "OK"
        ),
    }


def save_report(results):
    report = pd.DataFrame([results])

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(REPORT_PATH, index=False)

    return report


def main():
    print("=" * 70)
    print("LOAN DEFAULT PREDICTION DRIFT ANALYSIS")
    print("=" * 70)

    baseline, evolved = load_data()

    print(f"Baseline dataset : {baseline.shape}")
    print(f"Evolved dataset  : {evolved.shape}")
    print(f"Model path       : {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}"
        )

    # Load existing trained model without retraining
    model = joblib.load(MODEL_PATH)

    print()
    print("Loaded existing Loan Default model successfully.")

    baseline_predictions, baseline_probabilities = predict(
        model,
        baseline,
    )

    evolved_predictions, evolved_probabilities = predict(
        model,
        evolved,
    )

    summarize_predictions(
        "BASELINE",
        baseline_predictions,
        baseline_probabilities,
    )

    summarize_predictions(
        "EVOLVED",
        evolved_predictions,
        evolved_probabilities,
    )

    results = analyze_prediction_drift(
        baseline_predictions,
        evolved_predictions,
        baseline_probabilities,
        evolved_probabilities,
    )

    print()
    print("=" * 70)
    print("PREDICTION DRIFT RESULTS")
    print("=" * 70)

    print(
        f"Prediction distribution KS : "
        f"{results['prediction_distribution_ks']:.6f}"
    )

    print(
        f"Prediction distribution p  : "
        f"{results['prediction_distribution_p_value']:.6f}"
    )

    print(
        f"Prediction distribution    : "
        f"{results['prediction_distribution_status']}"
    )

    print()

    print(
        f"Probability distribution KS : "
        f"{results['probability_distribution_ks']:.6f}"
    )

    print(
        f"Probability distribution p  : "
        f"{results['probability_distribution_p_value']:.6f}"
    )

    print(
        f"Probability distribution    : "
        f"{results['probability_distribution_status']}"
    )

    print()
    print(f"Threshold : KS >= {KS_THRESHOLD}")

    save_report(results)

    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
