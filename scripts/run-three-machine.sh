#!/usr/bin/env bash
set -euo pipefail

# Keep the historical command name, but never open a second MCDMA client beside
# the persistent API.  All operator requests go through the single owner.
exec "$(dirname "$0")/chat-three-machine.sh" \
  "${1:-The capital of France is}" "${2:-8}"
