import os
import shutil
import json
from datetime import datetime
import logging

def create_model_version():
    try:
        print("📦 Versioning script started.")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /app or local
        model_dir = os.path.join(base_dir, "Model")
        current_model_dir = os.path.join(model_dir, "Current_Model")
        version_root = os.path.join(model_dir, "versions")
        os.makedirs(version_root, exist_ok=True)

        latest_model = os.path.join(current_model_dir, "model_rf.joblib")
        latest_metadata = os.path.join(current_model_dir, "model_metadata.json")

        # Check if files exist
        if not os.path.exists(latest_model) or not os.path.exists(latest_metadata):
            raise FileNotFoundError("Model or metadata not found in Current_Model folder")

        # Create new version directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = os.path.join(version_root, f"v_{timestamp}")
        os.makedirs(version_dir, exist_ok=True)

        # Copy files
        shutil.copy(latest_model, os.path.join(version_dir, "model_rf.joblib"))
        shutil.copy(latest_metadata, os.path.join(version_dir, "model_metadata.json"))

        # Update manifest
        manifest = {
            "latest_version": f"v_{timestamp}",
            "model_path": latest_model,
            "metadata_path": latest_metadata
        }

        manifest_path = os.path.join(model_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)

        print(f"✅ Model version saved as: {version_dir}")
        print("✅ Manifest updated.")
    except Exception as e:
        logging.error(f"❌ Failed to version model: {e}")