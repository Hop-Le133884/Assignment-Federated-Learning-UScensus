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

# Write project-local flwr config (num-supernodes=5) so global ~/.flwr/config.toml
# on any machine is never read or modified.
FLWR_HOME="$SCRIPT_DIR/.flwr"
mkdir -p "$FLWR_HOME"
cat > "$FLWR_HOME/config.toml" << 'EOF'
[superlink]
default = "local-simulation"

[superlink.local-simulation]
options.num-supernodes = 5
EOF
export FLWR_HOME

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
    --run-config "workspace-path=\"$ABS_WORKSPACE\""
