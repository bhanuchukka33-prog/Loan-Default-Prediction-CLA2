import subprocess
import sys

print("=" * 50)
print("LAB 6: MODEL REGISTRY AND DEPLOYMENT")
print("=" * 50)

subprocess.run(
    [sys.executable, "src/model_registry.py"],
    check=True
)

print("\nLAB 6 EXECUTED SUCCESSFULLY!")