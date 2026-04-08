#!/usr/bin/env bash
# run_fl_experiment.sh — Run the federated learning simulation.
#
# Usage:
#   ./run_fl_experiment.sh              — start simulation
#   ./run_fl_experiment.sh log <run-id> — stream logs for a run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WORKSPACE="./workspace"
ABS_WORKSPACE="$(realpath "$WORKSPACE")"
FLWR="$ABS_WORKSPACE/.venv/bin/flwr"

# Write project-local flwr config so global ~/.flwr/config.toml is never touched.
# address = ":local:" is required by flwr >= 1.28 for in-process simulation.
FLWR_HOME="$SCRIPT_DIR/.flwr"
mkdir -p "$FLWR_HOME"
cat > "$FLWR_HOME/config.toml" << 'EOF'
[superlink]
default = "local-simulation"

[superlink.local-simulation]
address = ":local:"
EOF
export FLWR_HOME

# Kill any flower-superlink process from a previous failed run.
# If the old process is still alive it holds an open file handle on the stale
# state.db; a new run will reuse that broken SuperLink instead of starting fresh.
pkill -f "flower-superlink.*$(basename "$FLWR_HOME")" 2>/dev/null || true
sleep 1

# Clear stale SuperLink state (db + ffs objects) from any previous failed run.
# Must happen AFTER killing the process so the db file descriptor is released.
rm -rf "$FLWR_HOME/local-superlink"

if [ ! -f "$FLWR" ]; then
    echo "Error: workspace venv not found. Run ./jobs_gen.sh first."
    exit 1
fi

# log subcommand: stream logs for a given run-id
if [ "${1:-}" = "log" ]; then
    if [ -z "${2:-}" ]; then
        echo "Usage: ./run_fl_experiment.sh log <run-id>"
        exit 1
    fi
    "$FLWR" log "$2" local-simulation
    exit 0
fi

echo "Running FL simulation → $ABS_WORKSPACE"
echo "======================================="

# Prepend workspace venv to PATH so flower-superlink resolves to the right one
export PATH="$ABS_WORKSPACE/.venv/bin:$PATH"

"$FLWR" run "$WORKSPACE/" local-simulation \
    --federation-config "num-supernodes=5" \
    --run-config "workspace-path=\"$ABS_WORKSPACE\"" \
    --stream
