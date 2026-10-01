import subprocess
import sys

print("=" * 60)
print("LOAN DEFAULT - LAB 4 MLFLOW TRACKING")
print("=" * 60)

subprocess.run(
    [sys.executable, "src/train_tracking.py"],
    check=True
)

print("\n[SUCCESS] Lab 4 completed successfully!")
