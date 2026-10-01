
import joblib
import pandas as pd

from pathlib import Path
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from .process import preprocess_data


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def train_models():

    print("\nLoading and preprocessing data...")

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=500,
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

    results = []
    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        trained_models[name] = pipeline

        print("Accuracy:", round(accuracy, 4))
        print("F1 Score:", round(f1, 4))

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    best_name = results_df.sort_values(
        by="F1 Score",
        ascending=False
    ).iloc[0]["Model"]

    best_model = trained_models[best_name]

    joblib.dump(
        best_model,
        MODEL_DIR / "best_model.pkl"
    )

    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    print(results_df.to_string(index=False))

    print("\nBest Model:", best_name)
    print("Model saved successfully!")

    return best_model, X_test, y_test
