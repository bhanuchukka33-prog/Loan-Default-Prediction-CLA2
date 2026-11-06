
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = ROOT / "data" / "Loan_Default.csv"
OUTPUT_PATH = ROOT / "data" / "monitoring" / "drift" / "evolved_data.csv"

RANDOM_SEED = 42


def create_evolved_dataset():
    rng = np.random.default_rng(RANDOM_SEED)

    # Load original Loan Default dataset
    df = pd.read_csv(SOURCE_PATH)
    df = df.dropna(subset=["Status"])

    # Separate features and target
    X = df.drop(columns=["Status", "ID"], errors="ignore")
    y = df["Status"]

    # Same train-test split as the original project
    _, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    evolved = X_test.copy()

    # ---------------------------------------------------------
    # 1. Numerical drift: Loan Amount
    # ---------------------------------------------------------
    if "loan_amount" in evolved.columns:
        values = pd.to_numeric(
            evolved["loan_amount"], errors="coerce"
        )
        evolved["loan_amount"] = (
            values * rng.normal(1.15, 0.08, len(evolved))
        ).clip(lower=0)

    # ---------------------------------------------------------
    # 2. Numerical drift: Income
    # ---------------------------------------------------------
    if "income" in evolved.columns:
        values = pd.to_numeric(
            evolved["income"], errors="coerce"
        )
        evolved["income"] = (
            values * rng.normal(1.20, 0.08, len(evolved))
        ).clip(lower=0)

    # ---------------------------------------------------------
    # 3. Numerical drift: Property Value
    # ---------------------------------------------------------
    if "property_value" in evolved.columns:
        values = pd.to_numeric(
            evolved["property_value"], errors="coerce"
        )
        evolved["property_value"] = (
            values * rng.normal(1.10, 0.08, len(evolved))
        ).clip(lower=0)

    # ---------------------------------------------------------
    # 4. Numerical drift: Credit Score
    # ---------------------------------------------------------
    if "Credit_Score" in evolved.columns:
        values = pd.to_numeric(
            evolved["Credit_Score"], errors="coerce"
        )
        evolved["Credit_Score"] = (
            values + rng.normal(-40, 20, len(evolved))
        ).clip(lower=300, upper=850)

    # ---------------------------------------------------------
    # 5. Categorical drift: Loan Purpose
    # ---------------------------------------------------------
    if "loan_purpose" in evolved.columns:
        categories = evolved["loan_purpose"].dropna().unique()

        if len(categories) > 0:
            evolved["loan_purpose"] = rng.choice(
                categories,
                size=len(evolved),
                replace=True
            )

    # ---------------------------------------------------------
    # 6. Categorical drift: Region
    # ---------------------------------------------------------
    if "Region" in evolved.columns:
        categories = evolved["Region"].dropna().unique()

        if len(categories) > 0:
            evolved["Region"] = rng.choice(
                categories,
                size=len(evolved),
                replace=True
            )

    # ---------------------------------------------------------
    # Save evolved dataset
    # ---------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    evolved.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("LOAN DEFAULT EVOLVED DATASET CREATED")
    print("=" * 60)

    print(f"\nSource : {SOURCE_PATH}")
    print(f"Output : {OUTPUT_PATH}")
    print(f"Rows   : {len(evolved)}")
    print(f"Columns: {len(evolved.columns)}")

    print("\nModified features:")
    print("- loan_amount")
    print("- income")
    print("- property_value")
    print("- Credit_Score")
    print("- loan_purpose")
    print("- Region")


if __name__ == "__main__":
    create_evolved_dataset()
