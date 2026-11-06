from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Loan_Default.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
OUTPUT_PATH = ROOT / "outputs" / "monitoring" / "performance_trend_report.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["ID", "Status"])
    y = df["Status"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    windows = np.array_split(np.arange(len(X_test)), 3)
    results = []

    for i, indices in enumerate(windows, start=1):
        X_window = X_test.iloc[indices]
        y_window = y_test.iloc[indices]

        predictions = model.predict(X_window)
        probabilities = model.predict_proba(X_window)[:, 1]

        results.append({
            "window": f"Window {i}",
            "samples": len(indices),
            "accuracy": accuracy_score(y_window, predictions),
            "precision": precision_score(y_window, predictions, zero_division=0),
            "recall": recall_score(y_window, predictions, zero_division=0),
            "f1": f1_score(y_window, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_window, probabilities)
        })

    report = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print("\nMODEL PERFORMANCE TREND ANALYSIS")
    print("=" * 65)
    print(report.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    change = report["recall"].iloc[-1] - report["recall"].iloc[0]
    print(f"\nRecall change (Window 1 to 3): {change:+.4f}")
    print(f"Report saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
