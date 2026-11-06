import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "logs" / "inference" / "inference_log.jsonl"
OUTPUT_PATH = ROOT / "outputs" / "monitoring" / "prediction_monitoring_report.csv"

def main():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_PATH}")

    with LOG_PATH.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError("Inference log is empty")

    df = pd.DataFrame(records)

    print("\n" + "=" * 60)
    print("LOAN DEFAULT PREDICTION MONITORING")
    print("=" * 60)
    print(f"\nTotal API requests: {len(df)}")

    if "status" in df.columns:
        print("\nRequest status distribution:")
        print(df["status"].value_counts())

    if "prediction" in df.columns:
        prediction_col = "prediction"
    elif "predicted_status" in df.columns:
        prediction_col = "predicted_status"
    else:
        prediction_col = None

    if prediction_col:
        counts = df[prediction_col].value_counts().sort_index()
        percentages = (counts / len(df) * 100).round(2)
        print("\nPrediction counts:")
        print(counts)
        print("\nPrediction percentages:")
        print(percentages)

    probability_col = next(
        (col for col in ["default_probability", "probability", "probabilities"]
         if col in df.columns), None
    )

    if probability_col:
        print("\nDefault probability statistics:")
        print(df[probability_col].describe())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nReport saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
