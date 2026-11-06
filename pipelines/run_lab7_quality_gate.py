from src.quality_gate import run_quality_gate


def main():
    print("Starting Lab 7: Quality Gate")

    passed = run_quality_gate()

    if passed:
        print("Model passed the quality gate.")
    else:
        print("Model failed the quality gate.")

    return passed


if __name__ == "__main__":
    success = main()

    if not success:
        raise SystemExit(1)