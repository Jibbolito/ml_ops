import os
import shutil
import datetime
import json
import logging  # ✅ correct import

def create_model_version():
    try:
        print("📦 Versioning script started.")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, "Model")
        version_root = os.path.join(model_dir, "versions")
        latest_model = os.path.join(model_dir, "model_rf.joblib")
        latest_metadata = os.path.join(model_dir, "model_metadata.json")

        print(f"📂 Checking paths:")
        print(f"Model file path: {latest_model}")
        print(f"Metadata file path: {latest_metadata}")
        print(f"Exists? model: {os.path.exists(latest_model)}, metadata: {os.path.exists(latest_metadata)}")


        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version_path = os.path.join(version_root, f"v_{timestamp}")
        os.makedirs(version_path, exist_ok=True)

        shutil.copy(latest_model, os.path.join(version_path, "model.joblib"))
        shutil.copy(latest_metadata, os.path.join(version_path, "metadata.json"))

        # Load flow info from metadata
        with open(latest_metadata) as meta_file:
            metadata = json.load(meta_file)
            flow_version = metadata.get("flow_version", "unknown")
            n_estimators = metadata.get("n_estimators", None)
            max_depth = metadata.get("max_depth", None)

        manifest = {
            "latest_version": f"v_{timestamp}",
            "model_path": os.path.abspath(latest_model),
            "metadata_path": os.path.abspath(latest_metadata),
            "flow_version": flow_version,
            "n_estimators": n_estimators,
            "max_depth": max_depth
        }

        with open(os.path.join(model_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=4)

        print(f"✅ Model version saved as: {version_path}")
        print("✅ Manifest updated.")

    except Exception as e:
        logging.error(f"❌ Failed to version model: {e}")
