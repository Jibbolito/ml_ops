#!/bin/bash
set -e  # Exit if any command fails

echo "🚀 Starting full test suite..."

echo "1️⃣ Running run_all_tests.py..."
python run_all_tests.py

echo "2️⃣ Training A/B models..."
python train_ab_model.py --variant model_a --n_estimators 100 --max_depth 10
python train_ab_model.py --variant model_b --n_estimators 150 --max_depth 15

echo "3️⃣ Running A/B test..."
python ab_test_runner.py

echo "✅ All workflows completed."
