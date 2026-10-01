
import subprocess
import sys

print("=" * 60)
print("LOAN DEFAULT - LAB 5 PRODUCTION PIPELINE")
print("=" * 60)

steps = [
    "src/validate_data.py",
    "src/preprocess_pipeline.py",
    "src/validate_outputs.py"
]

for step in steps:
    print(f"\n[INFO] Executing {step}...")

    subprocess.run(
        [sys.executable, step],
        check=True
    )

print("\n[SUCCESS] LAB 5 PRODUCTION PIPELINE COMPLETED!")
