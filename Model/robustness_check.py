import warnings
import pandas as pd
import json
import os
import logging
import joblib
from pathlib import Path

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)

def get_data_path(filename):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_dir, "Data", filename)

# Resolve model path relative to the script's location
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
project_root = Path(__file__).resolve().parents[1]
model_path = project_root / "Model" / "model_rf.joblib"
json_path = project_root / "Model" / "model_metadata.json"


try:
    model = joblib.load(model_path)
    logging.info("✅ Model loaded successfully.")
except FileNotFoundError:
    logging.error(f"❌ Model file not found at: {model_path}")
    exit(1)  # use exit instead of return (outside function)


def validate_input(df, expected_features):
    problems = []

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        problems.append("Missing values detected.")

    # Check for unexpected categorical values
    known_categories = {
        "House_Ownership": {"owned", "rented", "norent_noown"},
        "Married/Single": {"single", "married"},
    }

    for col, allowed in known_categories.items():
        if col in df.columns:
            invalid = set(df[col].dropna().unique()) - allowed
            if invalid:
                problems.append(f"Unexpected values in '{col}': {invalid}")

    return problems



def simulate_error_data():
    df = pd.read_csv(get_data_path("up_clean.csv"))
    df.loc[0, "Income"] = None
    df.loc[1, "House_Ownership"] = "unknown"
    df.loc[2, "Married/Single"] = 12345
    return df


def run_robustness_check(df):
    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)
        expected_features = metadata["feature_names"]

        problems = validate_input(df, expected_features)
        if problems:
            raise ValueError("⚠️ Input validation failed:\n" + "\n".join(problems))

        df_encoded = pd.get_dummies(df)

        missing_cols = [col for col in expected_features if col not in df_encoded.columns]
        if missing_cols:
            df_encoded = pd.concat([df_encoded, pd.DataFrame(0, index=df_encoded.index, columns=missing_cols)], axis=1)

        df_encoded = df_encoded[expected_features]

        predictions = model.predict(df_encoded)
        logging.info("✅ Prediction succeeded.")
    except Exception as e:
        logging.error(f"❌ Prediction failed due to: {e}")



if __name__ == "__main__":
    logging.info("Running tests on clean data...")
    run_robustness_check(pd.read_csv(get_data_path("up_clean.csv")))

    logging.info("Running tests on dirty data...")
    run_robustness_check(simulate_error_data())
