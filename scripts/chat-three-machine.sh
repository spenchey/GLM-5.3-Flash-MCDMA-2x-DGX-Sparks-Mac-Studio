#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_three_machine_config

message=${1:-Reply with exactly THREE_MACHINE_READY}
max_tokens=${2:-256}
[[ "$max_tokens" =~ ^[1-9][0-9]*$ ]] || die "max tokens must be a positive whole number"

payload=$(python3 - "$message" "$max_tokens" <<'PY'
import json, sys
print(json.dumps({"model": "GLM-5.3-Flash-TensorFold-MCDMA-3Machine",
                  "messages": [{"role": "user", "content": sys.argv[1]}],
                  "max_tokens": int(sys.argv[2]), "temperature": 0,
                  "enable_thinking": False}, separators=(",", ":")))
PY
)
printf '%s' "$payload" | studio_ssh "curl -fsS --max-time 300 -H 'Content-Type: application/json' --data-binary @- 'http://$THREE_MACHINE_API_HOST:$THREE_MACHINE_API_PORT/v1/chat/completions'"
printf '\n'
