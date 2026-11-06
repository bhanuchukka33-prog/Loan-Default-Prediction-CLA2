import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "logs" / "inference" / "inference_log.jsonl"

def main():
    with LOG_PATH.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError("Inference history is empty")

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    print("\n" + "=" * 60)
    print("LOAN DEFAULT REQUEST HISTORY")
    print("=" * 60)
    print(f"\nTotal records: {len(df)}")
    print("\nRecent requests:")
    print(df.tail(10).to_string(index=False))

    failures = df[df["status"] != "success"]
    print(f"\nFailed requests: {len(failures)}")

    if failures.empty:
        print("No failed requests.")
    else:
        print(failures.to_string(index=False))

if __name__ == "__main__":
    main()
