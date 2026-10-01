
import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = "data/Loan_Default.csv"
TARGET_COLUMN = "Status"

EXPERIMENT_NAME = "Loan_Default_Classification"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================

print("=" * 60)
print("LOAN DEFAULT - LAB 4 MLFLOW TRACKING")
print("=" * 60)

print("\nLoading Loan Default Dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

df = df.dropna(subset=[TARGET_COLUMN]).copy()

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Encode target labels if categorical
if not pd.api.types.is_numeric_dtype(y):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y.astype(str))

# Remove identifier columns
id_columns = [
    col for col in X.columns
    if col.lower() in ["id", "loan_id", "customer_id"]
]

X = X.drop(columns=id_columns)

print("Features:", X.shape[1])
print("Target Distribution:")
print(pd.Series(y).value_counts())

# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================================
# 4. PREPROCESSING
# ==========================================

numeric_features = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude="number"
).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ==========================================
# 5. DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
}

# ==========================================
# 6. MLFLOW EXPERIMENT
# ==========================================

mlflow.set_experiment(EXPERIMENT_NAME)

results = []
best_f1 = -1
best_model = None
best_model_name = None

print("\nStarting MLflow Experiment Tracking...")

# ==========================================
# 7. TRAIN AND TRACK MODELS
# ==========================================

for name, model in models.items():

    print("\nTraining", name, "...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    with mlflow.start_run(run_name=name):

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )

        # Log parameters
        mlflow.log_param("model_name", name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("features", X.shape[1])

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save model locally
        model_path = f"models/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(pipeline, model_path)

        # Log model file as artifact
        mlflow.log_artifact(model_path)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))

        print("-" * 50)

        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            best_model_name = name

# ==========================================
# 8. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/mlflow_model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

# ==========================================
# 9. SAVE BEST MODEL
# ==========================================

joblib.dump(best_model, "models/best_model.pkl")

print("\nBest Model:", best_model_name)
print("Best F1 Score:", round(best_f1, 4))

# ==========================================
# 10. SAVE EVALUATION REPORT
# ==========================================

best_predictions = best_model.predict(X_test)

report = classification_report(
    y_test,
    best_predictions,
    output_dict=True,
    zero_division=0
)

with open("outputs/evaluation_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_predictions))

print("\n[SUCCESS] Lab 4 MLflow Tracking Completed!")
print("=" * 60)
