import json
import os
from datetime import datetime

def generate_report(metadata_path="Model/model_metadata.json", output_path="Model/model_summary.md"):
    print(f"Reading metadata from: {os.path.abspath(metadata_path)}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "# Model Summary Report",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        f"**Model Path:** {metadata['model_path']}",
        f"**Test Accuracy:** {metadata['accuracy']:.2%}",
        "",
        "## Input Features",
        ", ".join(metadata["feature_names"]),
        "",
        f"**Test Sample Size:** {metadata['n_samples']}",
        f"**Metadata Timestamp:** {metadata['trained_at']}"
    ]

    abs_path = os.path.abspath(output_path)
    print(f"Writing to: {abs_path}")

    with open(abs_path, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ Summary report written to: {abs_path}")

if __name__ == "__main__":
    generate_report()
