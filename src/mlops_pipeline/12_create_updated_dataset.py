
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = ROOT / "data" / "Loan_Default.csv"

NEW_BATCH_PATH = (
    ROOT / "data" / "monitoring" / "drift"
    / "post_drift_training_data.csv"
)

UPDATED_DATASET_PATH = (
    ROOT / "data" / "processed"
    / "updated_training_data.csv"
)

SUMMARY_PATH = (
    ROOT / "outputs" / "retraining"
    / "updated_dataset_summary.csv"
)

RANDOM_SEED = 42
NEW_BATCH_SIZE = 1409


def load_original_training_data():
    df = pd.read_csv(SOURCE_PATH)

    df = df.dropna(subset=["Status"])

    X = df.drop(columns=["Status", "ID"], errors="ignore")
    y = df["Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, y_train, X_test


def create_post_drift_batch(X_train, y_train):
    rng = np.random.default_rng(RANDOM_SEED)

    sample_indices = rng.choice(
        len(X_train),
        size=NEW_BATCH_SIZE,
        replace=True
    )

    X_new = (
        X_train.iloc[sample_indices]
        .copy()
        .reset_index(drop=True)
    )

    y_new = y_train.iloc[sample_indices].copy().reset_index(drop=True)

    # Numerical drift: Loan amount
    if "loan_amount" in X_new.columns:
        values = pd.to_numeric(
            X_new["loan_amount"], errors="coerce"
        )

        X_new["loan_amount"] = (
            values * rng.normal(1.15, 0.08, len(X_new))
        ).clip(lower=0)

    # Numerical drift: Income
    if "income" in X_new.columns:
        values = pd.to_numeric(
            X_new["income"], errors="coerce"
        )

        X_new["income"] = (
            values * rng.normal(1.20, 0.08, len(X_new))
        ).clip(lower=0)

    # Numerical drift: Property value
    if "property_value" in X_new.columns:
        values = pd.to_numeric(
            X_new["property_value"], errors="coerce"
        )

        X_new["property_value"] = (
            values * rng.normal(1.10, 0.08, len(X_new))
        ).clip(lower=0)

    # Numerical drift: Credit score
    if "Credit_Score" in X_new.columns:
        values = pd.to_numeric(
            X_new["Credit_Score"], errors="coerce"
        )

        X_new["Credit_Score"] = (
            values + rng.normal(-40, 20, len(X_new))
        ).clip(lower=300, upper=850)

    # Categorical drift: Loan purpose
    if "loan_purpose" in X_new.columns:
        categories = X_new["loan_purpose"].dropna().unique()

        if len(categories) > 0:
            X_new["loan_purpose"] = rng.choice(
                categories,
                size=len(X_new),
                replace=True
            )

    # Categorical drift: Region
    if "Region" in X_new.columns:
        categories = X_new["Region"].dropna().unique()

        if len(categories) > 0:
            X_new["Region"] = rng.choice(
                categories,
                size=len(X_new),
                replace=True
            )

    return X_new, y_new


def validate_new_batch(X_new, y_new, X_test):
    if list(X_new.columns) != list(X_test.columns):
        raise ValueError("Feature schema mismatch!")

    if len(X_new) != len(y_new):
        raise ValueError("Feature and label count mismatch!")

    if X_new.isnull().all().any():
        raise ValueError("A column contains only null values!")

    if (X_new["loan_amount"] < 0).any():
        raise ValueError("Negative loan amounts detected!")

    return True


def create_updated_dataset(X_train, y_train, X_new, y_new):
    original_train = X_train.copy()
    original_train["Status"] = y_train.values

    new_batch = X_new.copy()
    new_batch["Status"] = y_new.values

    updated = pd.concat(
        [original_train, new_batch],
        ignore_index=True
    )

    return updated, original_train, new_batch


def save_summary(original_train, new_batch, updated):
    rows = []

    for name, dataset in [
        ("original_training", original_train),
        ("post_drift_batch", new_batch),
        ("updated_training", updated)
    ]:
        rows.append({
            "dataset": name,
            "rows": len(dataset),
            "default": int((dataset["Status"] == 1).sum()),
            "non_default": int((dataset["Status"] == 0).sum()),
            "default_rate": round(
                float(dataset["Status"].mean()), 6
            )
        })

    summary = pd.DataFrame(rows)

    SUMMARY_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    summary.to_csv(SUMMARY_PATH, index=False)

    return summary


def main():
    print("=" * 70)
    print("LOAN DEFAULT UPDATED DATASET CREATION")
    print("=" * 70)

    X_train, y_train, X_test = load_original_training_data()

    print("\nOriginal Training Features:", X_train.shape)
    print("Training Labels:", y_train.shape)
    print("Test Features:", X_test.shape)

    X_new, y_new = create_post_drift_batch(
        X_train, y_train
    )

    validate_new_batch(X_new, y_new, X_test)

    updated, original_train, new_batch = create_updated_dataset(
        X_train, y_train, X_new, y_new
    )

    NEW_BATCH_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    UPDATED_DATASET_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    new_batch.to_csv(NEW_BATCH_PATH, index=False)
    updated.to_csv(UPDATED_DATASET_PATH, index=False)

    summary = save_summary(
        original_train, new_batch, updated
    )

    print("\nPost-Drift Batch:", X_new.shape)
    print("Updated Dataset:", updated.shape)

    print("\nDataset Summary:")
    print(summary.to_string(index=False))

    print("\nFiles Created:")
    print(NEW_BATCH_PATH)
    print(UPDATED_DATASET_PATH)
    print(SUMMARY_PATH)

    print("\n[SUCCESS] Updated dataset creation completed!")


if __name__ == "__main__":
    main()
