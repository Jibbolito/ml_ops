import os
import argparse
import pandas as pd
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from helper_functions import get_data_path_root


def get_data_path(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "Data", filename)

def save_model_metadata(model, X, y_true, y_pred, output_path):
    metadata = {
        "model_type": type(model).__name__,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "n_samples": len(X),
        "accuracy": accuracy_score(y_true, y_pred),
        "model_file": output_path
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metadata_path = output_path.replace(".joblib", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Metadata saved to: {metadata_path}")

def train_ab_model(variant, n_estimators=100, max_depth=None):
    print(f"📦 Training variant: {variant}")
    df = pd.read_csv(get_data_path_root("train_data.csv"))

    target = "Risk_Flag"
    features = ["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS",
                "Married/Single", "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE"]

    X = df[features]
    y = df[target]
    X = pd.get_dummies(X, drop_first=True)

    if len(X) < 1000:
        raise ValueError("❌ Not enough training samples.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"🧠 Training RandomForest with n_estimators={n_estimators}, max_depth={max_depth}")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.2%}")

    model_filename = f"Model/{variant}.joblib"
    joblib.dump(model, model_filename)
    print(f"✅ Model saved to: {model_filename}")

    save_model_metadata(model, X_test, y_test, y_pred, model_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model variant for A/B testing.")
    parser.add_argument("--variant", required=True, help="Variant name (e.g., model_a or model_b)")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators for RandomForest")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth for RandomForest")
    args = parser.parse_args()

    train_ab_model(args.variant, args.n_estimators, args.max_depth)
