#!/bin/bash
set -e

echo "================================================"
echo "  SAM2 Fine-Tuning Container — Starting Up"
echo "================================================"

# Step 1: Download & prepare data if not already present
echo "[entrypoint] Checking training data..."
python /app/scripts/download_data.py --data-dir /app/fine_tuning_SAM2/data/train

# Step 2: Run training (pass through any CMD arguments)
echo "[entrypoint] Starting training..."
cd /app/fine_tuning_SAM2/sam2
exec "$@"
