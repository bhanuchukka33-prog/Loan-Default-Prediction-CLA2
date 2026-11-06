from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def test_missing_dataset():
    print("\nTest 1: Missing Dataset")

    dataset_path = ROOT / "data" / "missing_dataset.csv"

    try:
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

    except FileNotFoundError as error:
        print("Failure detected:", error)
        print("Recovery: Check dataset path and restore the file.")
        return True

    return False


def test_missing_evaluation_report():
    print("\nTest 2: Missing Evaluation Report")

    report_path = ROOT / "outputs" / "missing_report.json"

    try:
        if not report_path.exists():
            raise FileNotFoundError(
                f"Evaluation report not found: {report_path}"
            )

    except FileNotFoundError as error:
        print("Failure detected:", error)
        print("Recovery: Run model evaluation again.")
        return True

    return False


def test_missing_model():
    print("\nTest 3: Missing Model")

    model_path = ROOT / "models" / "missing_model.pkl"

    try:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

    except FileNotFoundError as error:
        print("Failure detected:", error)
        print("Recovery: Retrain the model or restore its backup.")
        return True

    return False


def generate_report(results):
    report_path = ROOT / "outputs" / "failure_testing_report.json"

    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "lab": "Lab 9 - Failure Injection and Recovery Testing",
        "tests": results,
        "total_tests": len(results),
        "failures_detected": sum(results.values()),
        "status": "COMPLETED"
    }

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    print("\nReport saved to:", report_path)


def run_failure_tests():
    print("=" * 50)
    print("LAB 9: FAILURE INJECTION AND RECOVERY TESTING")
    print("=" * 50)

    results = {
        "missing_dataset": test_missing_dataset(),
        "missing_evaluation_report": test_missing_evaluation_report(),
        "missing_model": test_missing_model()
    }

    print("\n========== TEST SUMMARY ==========")

    for test_name, result in results.items():
        print(f"{test_name}: {'PASSED' if result else 'FAILED'}")

    generate_report(results)


if __name__ == "__main__":
    run_failure_tests()