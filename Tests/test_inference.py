import pandas as pd
import logging
import os

def get_data_path(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "Data", filename)


def test_predictions_binary(df: pd.DataFrame):
    """Check if predictions are only 0 or 1."""
    unique_preds = set(df["Predicted_Risk_Flag"].unique())
    assert unique_preds.issubset({0, 1}), f"❌ Non-binary predictions found: {unique_preds}"
    logging.info("✅ All predictions are binary.")

def test_no_missing_predictions(df: pd.DataFrame):
    """Check for missing prediction values."""
    missing_count = df["Predicted_Risk_Flag"].isnull().sum()
    assert missing_count == 0, f"❌ Found {missing_count} missing prediction values."
    logging.info("✅ No missing predictions.")

def test_prediction_distribution(df: pd.DataFrame):
    """Check that both classes are represented."""
    value_counts = df["Predicted_Risk_Flag"].value_counts()
    logging.info(f"Prediction distribution:\n{value_counts}")
    if len(value_counts) < 2:
        logging.warning("⚠️ Only one class predicted — check for possible bias or input issues.")
    else:
        logging.info("✅ Both classes (0 and 1) are present in predictions.")

# Run all tests
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv(get_data_path("predictions.csv"))
    test_predictions_binary(df)
    test_no_missing_predictions(df)
    test_prediction_distribution(df)
