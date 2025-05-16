from evidently.test_suite import TestSuite
from evidently.test_preset import DataQualityTestPreset
import pandas as pd
from helper_functions import get_data_path


# Custom wrapper to test missing values
# _____________________________________
def strict_null_check(df):
    suite = TestSuite(tests=[DataQualityTestPreset()])
    suite.run(current_data=df, reference_data=df)
    result = suite.as_dict()

    print("Strict Null Value Test Results:\n")
    failed = False

    for test in result.get("tests", []):
        name = test.get("name", "")
        description = test.get("description", "")
        column = test.get("parameters", {}).get("column_name", "Unknown")
        null_ratio = test.get("parameters", {}).get("value", 0)

        # Check only tests that are checking null ratios
        if name == "The Share of Missing Values in a Column":
            status_icon = "✅" if null_ratio == 0 else "❌"
            print(f"{status_icon} Column '{column}': {null_ratio:.2%} missing")
            print(f"    → {description}")
            if null_ratio > 0:
                failed = True

    if failed:
        raise AssertionError("❌ Test failed: One or more columns contain null values.")
'''
# Run tests
# -------------------------------
print("Running tests on clean data:")
strict_null_check(pd.read_csv(get_data_path("up_clean.csv")))
print("\n\nRunning tests on dirty data:")
try:
    strict_null_check(pd.read_csv(get_data_path("up_dirty.csv")))
except AssertionError as e:
    print(e)
'''