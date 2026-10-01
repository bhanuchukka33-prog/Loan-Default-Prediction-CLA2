
import json
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def evaluate_model(model, X_test, y_test):

    print("\nEvaluating Best Model...")

    y_pred = model.predict(X_test)

    print("\nCLASSIFICATION REPORT")
    print("=" * 50)

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    cm = confusion_matrix(y_test, y_pred)

    print("\nCONFUSION MATRIX")
    print(cm)

    pd.DataFrame(cm).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv",
        index=False
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    with open(
        OUTPUT_DIR / "evaluation_report.json",
        "w"
    ) as f:

        json.dump(report, f, indent=4)

    print("\nEvaluation results saved!")
