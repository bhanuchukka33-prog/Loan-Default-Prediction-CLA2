import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path.cwd()

TEST_FEATURES_PATH = ROOT / "data/processed/X_test_raw.csv"
TEST_LABELS_PATH = ROOT / "data/processed/y_test.npy"

V1_MODEL_PATH = ROOT / "models/best_model.pkl"
V2_MODEL_PATH = ROOT / "models/retrained_model_v2.pkl"
V2_PREPROCESSOR_PATH = ROOT / "models/retrained_preprocessor_v2.pkl"

REPORT_PATH = ROOT / "outputs/retraining/model_comparison_report.csv"
JSON_REPORT_PATH = ROOT / "outputs/retraining/model_comparison_report.json"

RECALL_THRESHOLD = 0.70


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def main():
    print("=" * 70)
    print("LAB 12.14 - MODEL V1 VS V2 COMPARISON")
    print("=" * 70)

    X_test = pd.read_csv(TEST_FEATURES_PATH)
    y_test = np.load(TEST_LABELS_PATH).astype(int)

    if len(X_test) != len(y_test):
        raise ValueError("Test feature/label count mismatch.")

    print(f"Test features: {X_test.shape}")
    print(f"Test labels  : {y_test.shape}")

    # V1 is the complete sklearn pipeline.
    if not V1_MODEL_PATH.exists():
        raise FileNotFoundError(f"V1 model missing: {V1_MODEL_PATH}")

    v1_model = joblib.load(V1_MODEL_PATH)

    # V2 model and preprocessing are saved separately.
    if not V2_MODEL_PATH.exists():
        raise FileNotFoundError(f"V2 model missing: {V2_MODEL_PATH}")

    if not V2_PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"V2 preprocessor missing: {V2_PREPROCESSOR_PATH}"
        )

    v2_model = joblib.load(V2_MODEL_PATH)
    v2_preprocessor = joblib.load(V2_PREPROCESSOR_PATH)

    print("\nEvaluating V1...")
    v1_metrics = evaluate(v1_model, X_test, y_test)

    print("\nEvaluating V2...")
    X_test_transformed = v2_preprocessor.transform(X_test)
    v2_metrics = evaluate(v2_model, X_test_transformed, y_test)

    comparison = []

    for metric in v1_metrics:
        v1_value = float(v1_metrics[metric])
        v2_value = float(v2_metrics[metric])

        comparison.append({
            "metric": metric,
            "model_v1": v1_value,
            "model_v2": v2_value,
            "change_v2_minus_v1": v2_value - v1_value,
        })

    comparison_df = pd.DataFrame(comparison)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    v1_recall_pass = v1_metrics["recall"] >= RECALL_THRESHOLD
    v2_recall_pass = v2_metrics["recall"] >= RECALL_THRESHOLD
    v2_recall_improved = v2_metrics["recall"] > v1_metrics["recall"]

    comparison_df["model_v1"] = comparison_df["model_v1"].round(6)
    comparison_df["model_v2"] = comparison_df["model_v2"].round(6)
    comparison_df["change_v2_minus_v1"] = comparison_df[
        "change_v2_minus_v1"
    ].round(6)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(REPORT_PATH, index=False)

    report = {
        "stage": "Lab 12.14 - Model Comparison",
        "test_rows": int(len(X_test)),
        "model_v1": {
            "artifact": str(V1_MODEL_PATH),
            "metrics": v1_metrics,
        },
        "model_v2": {
            "artifact": str(V2_MODEL_PATH),
            "metrics": v2_metrics,
        },
        "comparison": comparison,
        "quality_gate": {
            "recall_threshold": RECALL_THRESHOLD,
            "v1_passed": v1_recall_pass,
            "v2_passed": v2_recall_pass,
            "v2_improved_over_v1": v2_recall_improved,
        },
    }

    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\nReports saved:")
    print(f"CSV : {REPORT_PATH}")
    print(f"JSON: {JSON_REPORT_PATH}")


if __name__ == "__main__":
    main()
