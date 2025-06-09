import pandas as pd
import joblib
import hashlib
import logging
from sklearn.metrics import accuracy_score
from datetime import datetime
import os
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
# Define folders relative to project root
data_dir = project_root / "Data"
model_dir = project_root / "Model" / "Current_Model"
logs_dir = project_root / "logs"
version_dir = model_dir / "versions"

# Setup logging
os.makedirs(logs_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(logs_dir, f"ab_test_log_{timestamp}.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

def deterministic_split(df, id_column="Id"):
    hash_vals = df[id_column].astype(str).apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16))
    return df[hash_vals % 2 == 0], df[hash_vals % 2 == 1]

def load_model_and_metadata(model_name):
    model_dir = project_root / "Model" / "AB_Testing"
    model_path = model_dir / f"{model_name}.joblib"
    meta_path = model_dir / f"{model_name}_metadata.json"
    model = joblib.load(model_path)
    metadata = pd.read_json(meta_path)
    return model, metadata["feature_names"]

def prepare_features(df, feature_names):
    df_encoded = pd.get_dummies(df, drop_first=True)
    missing_cols = [col for col in feature_names if col not in df_encoded.columns]
    if missing_cols:
        filler_df = pd.DataFrame(0, index=df_encoded.index, columns=missing_cols)
        df_encoded = pd.concat([df_encoded, filler_df], axis=1)
    df_encoded = df_encoded[feature_names]
    return df_encoded.copy()

def evaluate(model_name, model, features, labels):
    predictions = model.predict(features)
    accuracy = accuracy_score(labels, predictions)
    logging.info(f"✅ {model_name} Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    logging.info("📊 Starting A/B test on unseen_segment.csv")

    df = pd.read_csv(data_dir / "unseen_segment.csv")
    df_a, df_b = deterministic_split(df)

    model_a, features_a = load_model_and_metadata("model_a")
    model_b, features_b = load_model_and_metadata("model_b")

    Xa = prepare_features(df_a, features_a)
    Xb = prepare_features(df_b, features_b)
    ya = df_a["Risk_Flag"]
    yb = df_b["Risk_Flag"]

    acc_a = evaluate("model_a", model_a, Xa, ya)
    acc_b = evaluate("model_b", model_b, Xb, yb)

    logging.info(f"📊 A/B comparison complete — model_a: {acc_a:.4f}, model_b: {acc_b:.4f}")
