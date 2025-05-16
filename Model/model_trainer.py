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


# Setup logging
logging.basicConfig(level=logging.INFO)

def get_data_path(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up to project root
    return os.path.join(base_dir, "Data", filename)

def train_model():
    logging.info("Loading data...")
    df = pd.read_csv(get_data_path("master_dataset.csv"))

    # Select features & target
    target = "Risk_Flag"
    features = ["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS", "Married/Single",
                "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE"]

    X = df[features]
    y = df[target]

    # Encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Error handling: too few samples
    if len(X) < 1000:
        raise ValueError("❌ Not enough training samples (found < 1000 rows).")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info("Training model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"✅ Model trained. Accuracy on test set: {accuracy:.2%}")

    # Save model
    model_path = os.path.join("Model", "model_rf.joblib")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Save metadata
    save_model_metadata(model, X_test, y_test, y_pred, output_path="Model/model_metadata.json")
    logging.info(f"Model saved to: {model_path}")

    # Also version the model (creates timestamped folder)
    create_model_version()

def save_model_metadata(model, X, y_true, y_pred, output_path="Model/model_metadata.json"):
    metadata = {
        "model_type": type(model).__name__,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "n_samples": len(X),
        "accuracy": accuracy_score(y_true, y_pred),
        "model_path": output_path,
        "model_file": "Model/model_rf.joblib"
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info(f"✅ Metadata saved to: {output_path}")

if __name__ == "__main__":
    train_model()
