import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import logging
from datetime import datetime

def get_data_path(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "Data", filename)

# Setup logging to use root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

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
