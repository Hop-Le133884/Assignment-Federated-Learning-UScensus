#!/usr/bin/env bash
# data_splits_prep.sh — Split preprocessed Adult data into client/server sets.
#
# Usage: ./data_splits_prep.sh [NUM_CLIENTS] [WORKSPACE]
#   NUM_CLIENTS  Number of FL clients (default: 5)
#   WORKSPACE    Output directory (default: ./workspace)
#
# Example: ./data_splits_prep.sh 5 ./workspace

set -euo pipefail

NUM_CLIENTS="${1:-5}"
WORKSPACE="${2:-./workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
elif [ -f "venv/bin/activate" ]; then
    source "venv/bin/activate"
fi

# Ensure preprocessed data exists — run preprocessing if missing
PREPROCESSED="data/preprocessed/adult_preprocessed.csv"
STATS="data/preprocessed/preprocessing_stats.json"

if [ ! -f "$PREPROCESSED" ] || [ ! -f "$STATS" ]; then
    echo "Preprocessed data not found — running preprocessing first..."
    python3 utils/data_preprocessing.py \
        --input data/adult/adult.data \
        --output "$PREPROCESSED" \
        --stats "$STATS"
fi

echo ""
echo "Splitting data into $NUM_CLIENTS clients → $WORKSPACE"
echo "======================================================="

python3 utils/data_preparation.py \
    --num-clients "$NUM_CLIENTS" \
    --workspace "$WORKSPACE" \
    --preprocessed-csv "$PREPROCESSED" \
    --stats-path "$STATS" \
    --test-input data/adult/adult.test

echo ""
echo "Data splits ready in $WORKSPACE/"
