
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

DATA_PATH = "data/Loan_Default.csv"
TARGET_COLUMN = "Status"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("LAB 5 - PRODUCTION PREPROCESSING PIPELINE")
print("=" * 60)

# Load dataset
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=[TARGET_COLUMN])

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Remove identifier columns
id_columns = [
    col for col in X.columns
    if col.lower() in ["id", "loan_id", "customer_id"]
]

X = X.drop(columns=id_columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Identify feature types
numeric_features = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude="number"
).columns.tolist()

print("Numerical Features:", len(numeric_features))
print("Categorical Features:", len(categorical_features))

# Numerical pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combined preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

print("\nFitting preprocessing pipeline...")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save preprocessing pipeline
joblib.dump(preprocessor, "models/preprocessor.pkl")

# Save processed datasets
X_train_processed = X_train_processed.toarray() if hasattr(
    X_train_processed, "toarray"
) else X_train_processed

X_test_processed = X_test_processed.toarray() if hasattr(
    X_test_processed, "toarray"
) else X_test_processed

pd.DataFrame(X_train_processed).to_csv(
    "outputs/X_train_processed.csv", index=False
)

pd.DataFrame(X_test_processed).to_csv(
    "outputs/X_test_processed.csv", index=False
)

print("\nTraining Processed Shape:", X_train_processed.shape)
print("Testing Processed Shape:", X_test_processed.shape)

print("\n[SUCCESS] Preprocessing completed!")
print("Preprocessor saved as models/preprocessor.pkl")
