
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from src.train import train_models
from src.evaluate import evaluate_model

print("\n" + "=" * 55)
print("       LOAN DEFAULT PREDICTION PROJECT")
print("       LAB 3 - BASELINE MODEL TRAINING")
print("=" * 55)

model, X_test, y_test = train_models()

evaluate_model(model, X_test, y_test)

print("\nLAB 3 COMPLETED SUCCESSFULLY!")
