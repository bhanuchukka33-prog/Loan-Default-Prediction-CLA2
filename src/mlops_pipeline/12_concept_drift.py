
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = ROOT / "data" / "Loan_Default.csv"
EVOLVED_PATH = ROOT / "data" / "monitoring" / "drift" / "evolved_labeled_data.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
REPORT_PATH = ROOT / "outputs" / "drift" / "concept_drift_report.csv"

RANDOM_SEED = 42


def load_data():
    df = pd.read_csv(SOURCE_PATH)
    df = df.dropna(subset=["Status"])

    X = df.drop(columns=["ID", "Status"])
    y = df["Status"].astype(int)

    _, baseline_X, _, baseline_y = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    baseline_X = baseline_X.reset_index(drop=True)
    baseline_y = baseline_y.reset_index(drop=True)

    evolved = pd.read_csv(EVOLVED_PATH)

    evolved_X = evolved.drop(columns=["Status"])
    evolved_y = evolved["Status"].astype(int)

    if len(baseline_X) != len(baseline_y):
        raise ValueError("Baseline feature/label count mismatch")

    if len(evolved_X) != len(evolved_y):
        raise ValueError("Evolved feature/label count mismatch")

    return baseline_X, baseline_y, evolved_X, evolved_y


def evaluate(model, X, y):
    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def main():
    print("=" * 70)
    print("LOAN DEFAULT CONCEPT DRIFT ANALYSIS")
    print("=" * 70)

    baseline_X, baseline_y, evolved_X, evolved_y = load_data()

    print(f"Baseline dataset : {baseline_X.shape}")
    print(f"Evolved dataset  : {evolved_X.shape}")
    print(f"Model path      : {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print()
    print("Existing Loan Default model loaded successfully.")

    baseline_metrics = evaluate(
        model,
        baseline_X,
        baseline_y,
    )

    evolved_metrics = evaluate(
        model,
        evolved_X,
        evolved_y,
    )

    print()
    print("BASELINE PERFORMANCE")
    print("-" * 60)

    for metric, value in baseline_metrics.items():
        print(f"{metric.upper():10s}: {value:.4f}")

    print()
    print("EVOLVED PERFORMANCE")
    print("-" * 60)

    for metric, value in evolved_metrics.items():
        print(f"{metric.upper():10s}: {value:.4f}")

    results = []

    for metric in baseline_metrics:
        baseline_value = baseline_metrics[metric]
        evolved_value = evolved_metrics[metric]

        results.append({
            "metric": metric,
            "baseline": baseline_value,
            "evolved": evolved_value,
            "change": evolved_value - baseline_value,
        })

    report = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(REPORT_PATH, index=False)

    print()
    print("=" * 70)
    print("CONCEPT DRIFT PERFORMANCE CHANGE")
    print("=" * 70)

    print(report.to_string(index=False))

    print()
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
