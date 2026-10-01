
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "Loan_Default.csv"
OUTPUT_PATH = ROOT / "outputs"

OUTPUT_PATH.mkdir(exist_ok=True)


def load_data():

    print("\nLoading Loan Default Dataset...")

    df = pd.read_csv(DATA_PATH)

    print("\nDataset Shape:", df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    return df


def preprocess_data():

    df = load_data()

    target = "Status"

    if target not in df.columns:
        raise ValueError("Target column 'Status' not found!")

    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    # Remove identifier columns
    id_cols = [
        col for col in X.columns
        if col.lower() in ["id", "loanid"]
    ]

    X = X.drop(columns=id_cols, errors="ignore")

    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    cat_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Numerical preprocessing
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nPreprocessing Completed!")

    print("Training Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    print("\nTraining Target Distribution:")
    print(y_train.value_counts())

    return X_train, X_test, y_train, y_test, preprocessor
