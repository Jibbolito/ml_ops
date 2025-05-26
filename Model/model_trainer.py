import pandas as pd
import os
import joblib
import logging
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
from Model.version_model import create_model_version

# Setup logging
logging.basicConfig(level=logging.INFO)

def get_data_path(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up to project root
    return os.path.join(base_dir, "Data", filename)

def train_model(data_file, n_estimators, max_depth):
    logging.info(f"📦 Loading training data from: {data_file}")
    df = pd.read_csv(get_data_path(data_file))

    # Features and target
    target = "Risk_Flag"
    features = ["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS", "Married/Single",
                "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE"]

    X = df[features]
    y = df[target]

    # One-hot encode
    X = pd.get_dummies(X, drop_first=True)

    if len(X) < 1000:
        raise ValueError("❌ Not enough training samples (found < 1000 rows).")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info(f"🧠 Training RandomForest with n_estimators={n_estimators}, max_depth={max_depth}")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"✅ Model trained. Accuracy: {accuracy:.2%}")

    # Save model and metadata
    model_path = os.path.join("Model", "model_rf.joblib")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    save_model_metadata(model, X_test, y_test, y_pred, model_path, {
        "data_file": data_file,
        "n_estimators": n_estimators,
        "max_depth": max_depth
    })

    logging.info(f"✅ Model saved to: {model_path}")
    create_model_version()

def save_model_metadata(model, X, y_true, y_pred, model_path, config):
    metadata = {
        "model_type": type(model).__name__,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "n_samples": len(X),
        "accuracy": accuracy_score(y_true, y_pred),
        "model_path": model_path,
        "model_file": os.path.basename(model_path),
        "training_config": config
    }

    output_path = os.path.join("Model", "model_metadata.json")
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"✅ Metadata saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RandomForest model with configurable parameters.")
    parser.add_argument("--data_file", type=str, default="train_data.csv", help="Path to the training dataset (relative to /Data/)")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the tree.")
    args = parser.parse_args()

    train_model(args.data_file, args.n_estimators, args.max_depth)
