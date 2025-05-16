import pandas as pd
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_inference(input_path, output_path):
    logging.info("🔍 Loading model...")
    model = joblib.load("Model/model_rf.joblib")

    logging.info(f"📥 Reading input data from {input_path}...")
    df = pd.read_csv(input_path)

    # Apply same features used in training
    features = ["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS", "Married/Single",
                "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE"]

    df_features = df[features].copy()

    df_encoded = pd.get_dummies(df_features)

    model_input_cols = model.feature_names_in_
    df_encoded = df_encoded.reindex(columns=model_input_cols, fill_value=0)

    logging.info("⚙️ Running predictions...")
    predictions = model.predict(df_encoded)
    df["Predicted_Risk_Flag"] = predictions

    logging.info(f"📤 Saving predictions to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    logging.info("✅ Inference complete.")

if __name__ == "__main__":
    run_inference(input_path="../Data/up_clean.csv", output_path="../Data/predictions.csv")
