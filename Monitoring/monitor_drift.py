import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import logging
from datetime import datetime
from helper_functions import get_data_path_root
import re

# Write logs to the central /logs folder (one level above)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # up to project root
central_log_dir = os.path.join(base_dir, "logs")
os.makedirs(central_log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(central_log_dir, f"drift_log_{timestamp}.txt")

def get_data_path(filename):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_dir, "Data", filename)


# Helper: Remove emojis for console
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        record.msg = remove_emojis(str(record.msg))
        super().emit(record)

# Configure logging: file (emoji), terminal (safe)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
)
logging.getLogger().addHandler(SafeStreamHandler())


def calculate_js_divergence(p, q):
    return jensenshannon(p, q, base=2) ** 2

def bin_and_normalize(series, bins):
    counts, _ = np.histogram(series, bins=bins)
    return counts / counts.sum()

def monitor_income_drift(reference_path, current_path, threshold=0.05, bins=10):
    ref_df = pd.read_csv(reference_path)
    curr_df = pd.read_csv(current_path)

    ref_hist = bin_and_normalize(ref_df["Income"], bins)
    curr_hist = bin_and_normalize(curr_df["Income"], bins)

    js_div = calculate_js_divergence(ref_hist, curr_hist)
    logging.info(f"📊 Jensen-Shannon divergence for 'Income': {js_div:.4f}")

    if js_div > threshold:
        logging.warning(f"⚠️ Drift detected in 'Income' feature! Threshold: {threshold}")
    else:
        logging.info("✅ No significant drift detected in 'Income'.")

if __name__ == "__main__":
    reference = get_data_path("up_clean.csv")
    current = get_data_path("unseen_segment.csv")
    monitor_income_drift(reference, current)

