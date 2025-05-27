import argparse
import pandas as pd
import os
import joblib
import logging
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Model.version_model import create_model_version

os.makedirs("logs", exist_ok=True)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def get_data_path(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "Data", filename)


def train_model(n_estimators, max_depth, flow_version):
    logging.info(f"📦 Training with flow version: {flow_version}")
    logging.info(f"🧠 Params — n_estimators: {n_estimators}, max_depth: {max_depth}")

    df = pd.read_csv(get_data_path("train_data.csv"))

    # Features and target
    target = "Risk_Flag"
    features = ["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS", "Married/Single",
                "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE"]

    X = pd.get_dummies(df[features], drop_first=True)
    y = df[target]

    if len(X) < 1000:
        raise ValueError("❌ Not enough training samples (found < 1000 rows).")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"✅ Model trained. Accuracy: {accuracy:.2%}")

    # Save model
    model_path = os.path.join("Model", "model_rf.joblib")
    os.makedirs("Model", exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"✅ Model saved to: {model_path}")

    # Save metadata with hyperparams and flow version
    metadata = {
        "model_type": type(model).__name__,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "n_samples": len(X),
        "accuracy": accuracy,
        "model_path": model_path,
        "model_file": os.path.basename(model_path),
        "flow_version": flow_version,
        "n_estimators": n_estimators,
        "max_depth": max_depth
    }

    metadata_path = os.path.join("Model", "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"✅ Metadata saved to: {metadata_path}")

    # Save versioned copy
    create_model_version()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest Model with Versioning")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=None, help="Max tree depth")
    parser.add_argument("--flow_version", type=str, default="default", help="Flow version tag")

    args = parser.parse_args()
    train_model(args.n_estimators, args.max_depth, args.flow_version)
