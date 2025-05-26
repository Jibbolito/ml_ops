import pandas as pd
import joblib
import hashlib
import logging
from sklearn.metrics import accuracy_score
from datetime import datetime
import os
from helper_functions import get_data_path_root
import sys


# Setup logging
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join("logs", f"ab_test_log_{timestamp}.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)  # Force stdout encoding support
    ]
)


def deterministic_split(df, id_column="Id"):
    hash_vals = df[id_column].astype(str).apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16))
    return df[hash_vals % 2 == 0], df[hash_vals % 2 == 1]

def load_model_and_metadata(model_name):
    model_path = f"Model/{model_name}.joblib"
    meta_path = f"Model/{model_name}_metadata.json"
    model = joblib.load(model_path)
    with open(meta_path) as f:
        metadata = pd.read_json(f)
    return model, metadata["feature_names"]

def prepare_features(df, feature_names):
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Create a DataFrame with all missing columns in one go (avoids fragmentation)
    missing_cols = [col for col in feature_names if col not in df_encoded.columns]
    if missing_cols:
        filler_df = pd.DataFrame(0, index=df_encoded.index, columns=missing_cols)
        df_encoded = pd.concat([df_encoded, filler_df], axis=1)

    # Reorder columns to match training features
    df_encoded = df_encoded[feature_names]

    # Return a copy to defragment the internal blocks
    return df_encoded.copy()


def evaluate(model_name, model, features, labels):
    predictions = model.predict(features)
    accuracy = accuracy_score(labels, predictions)
    logging.info(f"✅ {model_name} Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    logging.info("📊 Starting A/B test on unseen_segment.csv")

    df = pd.read_csv(get_data_path_root("unseen_segment.csv"))

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
