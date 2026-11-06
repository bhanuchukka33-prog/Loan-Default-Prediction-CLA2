
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


FEATURES_PATH = Path("data/monitoring/drift/evolved_data.csv")
SOURCE_PATH = Path("data/Loan_Default.csv")
OUTPUT_PATH = Path("data/monitoring/drift/evolved_labeled_data.csv")

RANDOM_SEED = 42


def create_concept_drift_dataset():
    rng = np.random.default_rng(RANDOM_SEED)

    # Load evolved test features
    X = pd.read_csv(FEATURES_PATH)

    # Load original dataset
    df = pd.read_csv(SOURCE_PATH)

    # Remove rows without target labels
    df = df.dropna(subset=["Status"])

    # Remove ID and target from features
    X_original = df.drop(columns=["ID", "Status"])
    y = df["Status"].astype(int)

    # Recreate the original stratified train-test split
    _, X_test, _, y_test = train_test_split(
        X_original,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    y_test = y_test.reset_index(drop=True).to_numpy()

    if len(X) != len(y_test):
        raise ValueError(
            f"Feature/label mismatch: X={len(X)}, y={len(y_test)}"
        )

    y_evolved = y_test.copy()

    # ---------------------------------------------------------
    # Simulate concept drift:
    # Low credit-score borrowers become more likely to default.
    # ---------------------------------------------------------
    if "Credit_Score" in X.columns:

        low_credit_indices = np.where(
            X["Credit_Score"] <= X["Credit_Score"].quantile(0.25)
        )[0]

        eligible_indices = low_credit_indices[
            y_evolved[low_credit_indices] == 0
        ]

        flip_count = int(len(eligible_indices) * 0.20)

        if flip_count > 0:
            selected_indices = rng.choice(
                eligible_indices,
                size=flip_count,
                replace=False
            )

            y_evolved[selected_indices] = 1

    # ---------------------------------------------------------
    # Simulate another relationship change:
    # High loan-to-value borrowers become more likely to default.
    # ---------------------------------------------------------
    if "LTV" in X.columns:

        high_ltv_indices = np.where(
            X["LTV"] > X["LTV"].quantile(0.75)
        )[0]

        eligible_indices = high_ltv_indices[
            y_evolved[high_ltv_indices] == 0
        ]

        flip_count = int(len(eligible_indices) * 0.15)

        if flip_count > 0:
            selected_indices = rng.choice(
                eligible_indices,
                size=flip_count,
                replace=False
            )

            y_evolved[selected_indices] = 1

    # Create final concept drift dataset
    result = X.copy()
    result["Status"] = y_evolved

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("LOAN DEFAULT CONCEPT DRIFT DATASET CREATED")
    print("=" * 70)

    print(f"Feature source : {FEATURES_PATH}")
    print(f"Label source   : {SOURCE_PATH}")
    print(f"Output         : {OUTPUT_PATH}")
    print(f"Rows           : {len(result)}")
    print(f"Columns        : {len(result.columns)}")

    print()
    print("Baseline label distribution:")
    print(pd.Series(y_test).value_counts().sort_index())

    print()
    print("Evolved label distribution:")
    print(pd.Series(y_evolved).value_counts().sort_index())

    print()
    print("Label changes:")
    print(f"Changed labels : {(y_test != y_evolved).sum()}")

    print()
    print("[SUCCESS] Concept drift dataset creation completed!")


if __name__ == "__main__":
    create_concept_drift_dataset()
