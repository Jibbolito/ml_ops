import pandas as pd
import sys
import os
import datetime
import subprocess
from Tests.test_missing import strict_null_check
from Tests.test_distribution import test_income_distribution, test_house_ownership_distribution
from helper_functions import get_data_path
from Model.model_trainer import train_model

def run_all(main_log_handle):
    print("\nRunning model training step...")
    # Replace with explicit parameters if needed
    train_model(n_estimators=100, max_depth=None, flow_version="run_all_tests")

    print("Running all tests on clean data...")
    df_clean = pd.read_csv(get_data_path("up_clean.csv"))
    strict_null_check(df_clean)
    test_income_distribution(df_clean)
    test_house_ownership_distribution(df_clean)

    print("\nRunning all tests on dirty data...")
    df_dirty = pd.read_csv(get_data_path("up_dirty.csv"))
    for test_func in [strict_null_check, test_income_distribution, test_house_ownership_distribution]:
        try:
            test_func(df_dirty)
        except AssertionError as e:
            print(e)

    print("\n\nRunning inference tests... (results in separate log at the same location)")
    inference_log_path = os.path.join("logs", f"inference_log_{timestamp}.txt")
    with open(inference_log_path, "w", encoding="utf-8") as inference_log:
        result = subprocess.run(["python", "Tests/test_inference.py"], stdout=inference_log, stderr=inference_log, text=True)
        print(f"{'✅' if result.returncode == 0 else '❌'} Inference log saved to: {inference_log_path}")

    print("\n\nRunning robustness check:")
    robustness_log_path = os.path.join("logs", f"robustness_log_{timestamp}.txt")
    with open(robustness_log_path, "w", encoding="utf-8") as robustness_log:
        result = subprocess.run(["python", "Model/robustness_check.py"], stdout=robustness_log, stderr=robustness_log, text=True)
        print(f"{'✅' if result.returncode == 0 else '❌'} Robustness log saved to: {robustness_log_path}")

    print("\n\nRunning drift monitoring:")
    drift_log_path = os.path.join("logs", f"drift_log_{timestamp}.txt")
    with open(drift_log_path, "w", encoding="utf-8") as drift_log:
        result = subprocess.run(["python", "Monitoring/monitor_drift.py"], stdout=drift_log, stderr=drift_log, text=True)
        print(f"{'✅' if result.returncode == 0 else '❌'} Drift log saved to: {drift_log_path}")

    print("\n\nGenerating model summary:")
    result = subprocess.run(["python", "Model/generate_summary.py"], text=True)
    print("✅ Summary file written to Model/model_summary.md")

    print("\n\nRunning offline A/B test:")
    ab_log_path = os.path.join("logs", f"ab_test_log_{timestamp}.txt")
    with open(ab_log_path, "w", encoding="utf-8") as ab_log:
        result = subprocess.run(["python", "ab_test_runner.py"], stdout=ab_log, stderr=ab_log, text=True)
        print(f"{'✅' if result.returncode == 0 else '❌'} A/B test log saved to: {ab_log_path}")


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_path = os.path.join("logs", f"data_log_{timestamp}.txt")

    with open(main_log_path, "w", encoding="utf-8") as f:
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = f
        sys.stderr = f

        try:
            run_all(f)
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            print(f"✅ Main test logs saved to: /logs")
