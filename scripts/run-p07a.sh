#!/usr/bin/env bash
# Run on the Mac Studio in a prepared private environment. Never starts a service.
set -euo pipefail
if [[ $# -ne 4 ]]; then
  printf 'Usage: %s PRIVATE_PYTHON LIVE_SOURCE SOURCE_MANIFEST CHECKPOINT\n' "$0" >&2
  exit 2
fi
PYTHON_BIN=$1
LIVE_SOURCE=$2
SOURCE_MANIFEST=$3
CHECKPOINT=$4
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || { echo 'Apple Silicon Mac required' >&2; exit 1; }
[[ $(shasum -a 256 "$SOURCE_MANIFEST" | awk '{print $1}') == aacb31df395dbb4fed6bd01929b47e1d3d6040b953e8ce457d2fc984303c016c ]] || exit 1
(cd "$LIVE_SOURCE" && shasum -a 256 -c "$SOURCE_MANIFEST" >/dev/null)
export PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$LIVE_SOURCE"
"$PYTHON_BIN" -c 'import importlib.metadata as m; from packaging.version import Version as V; assert V("0.32.2") <= V(m.version("mlx")) < V("0.32.4"); assert V("0.31.3") <= V(m.version("mlx-lm")) < V("0.32")'
exec "$PYTHON_BIN" "$PROJECT_DIR/experiments/p07a/repeat.py" "$CHECKPOINT"
