#!/usr/bin/env bash
# jobs_gen.sh — Generate Flower app files (client_app.py, server_app.py, etc.)
#
# Usage: ./jobs_gen.sh [WORKSPACE]
#   WORKSPACE  Target directory (default: ./workspace)
#
# Example: ./jobs_gen.sh ./workspace

set -euo pipefail

WORKSPACE="${1:-./workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
elif [ -f "venv/bin/activate" ]; then
    source "venv/bin/activate"
fi

echo "Generating Flower app files → $WORKSPACE"
echo "========================================="

python3 utils/jobs_gen.py \
    --workspace "$WORKSPACE" \
    --num-clients 5

echo ""
echo "Generated files:"
find "$WORKSPACE" \( -name "*.py" -o -name "*.toml" -o -name ".gitignore" \) \
    ! -path "*/__pycache__/*" | sort

# Create workspace venv with Python 3.12 if missing or wrong Python version
VENV="$WORKSPACE/.venv"
NEED_VENV=false
if [ ! -f "$VENV/bin/activate" ]; then
    NEED_VENV=true
elif ! "$VENV/bin/python" --version 2>&1 | grep -q "3\.12"; then
    echo ""
    echo "Workspace venv has wrong Python version — recreating..."
    rm -rf "$VENV"
    NEED_VENV=true
fi

if [ "$NEED_VENV" = true ]; then
    echo ""
    echo "Setting up workspace venv (Python 3.12)..."
    cd "$WORKSPACE"
    uv venv --python 3.12
    uv sync
    cd "$SCRIPT_DIR"
    echo "Workspace venv ready."
fi
