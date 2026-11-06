import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Loan_Default.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
OUTPUT_PATH = ROOT / "outputs" / "monitoring" / "performance_monitoring_report.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["ID", "Status"])
    y = df["Status"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    print("=" * 60)
    print("LOAN DEFAULT PERFORMANCE MONITORING")
    print("=" * 60)
    print(f"Test samples: {len(X_test)}")
    print(f"Actual defaults: {int(y_test.sum())}")
    print(f"Actual non-defaults: {int((y_test == 0).sum())}")

    for metric, value in metrics.items():
        print(f"{metric.upper():<12}: {value:.4f}")

    print(f"\nRecall >= 0.70: {metrics['recall'] >= 0.70}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(OUTPUT_PATH, index=False)

    print(f"\nReport saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
