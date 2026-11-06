import json
from pathlib import Path


def run_quality_gate(
    report_path="outputs/evaluation_report.json",
    accuracy_threshold=0.80,
    f1_threshold=0.75,
):
    report_path = Path(report_path)

    # Check whether evaluation report exists
    if not report_path.exists():
        raise FileNotFoundError(
            f"Evaluation report not found: {report_path}"
        )

    # Load evaluation metrics
    with open(report_path, "r") as file:
        metrics = json.load(file)

    accuracy = metrics.get("accuracy", 0)

    macro_f1 = metrics.get("macro avg", {}).get(
        "f1-score", 0
    )

    # Quality checks
    checks = {
        "accuracy": accuracy >= accuracy_threshold,
        "macro_f1": macro_f1 >= f1_threshold,
    }

    passed = all(checks.values())

    # Prepare quality gate report
    result = {
        "quality_gate": "PASSED" if passed else "FAILED",
        "accuracy": accuracy,
        "accuracy_threshold": accuracy_threshold,
        "macro_f1": macro_f1,
        "f1_threshold": f1_threshold,
        "checks": checks,
    }

    # Save report
    output_path = Path("outputs/quality_gate_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(result, file, indent=4)

    # Display results
    print("\n========== QUALITY GATE ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Quality Gate: {result['quality_gate']}")
    print(f"Report saved to: {output_path}")

    return passed


if __name__ == "__main__":
    passed = run_quality_gate()

    if not passed:
        raise SystemExit(1)