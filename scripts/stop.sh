#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"

note "Stopping GLM on both Sparks"
head_ssh "cd '$GLM_RECIPE_DIR' && ./stop.sh"

