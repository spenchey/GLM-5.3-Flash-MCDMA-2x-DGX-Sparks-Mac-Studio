#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"

"$PROJECT_ROOT/scripts/preflight.sh"

if head_ssh "docker ps -a --format '{{.Names}}' | grep -Eq '^(dsv41-exl3-head|dsv41-exl3-worker)$'"; then
  die "DeepSeek containers are still present. Stop them before starting GLM."
fi
if head_ssh "ssh -o BatchMode=yes '$WORKER_SSH' docker ps -a --format '{{.Names}}' | grep -Eq '^(dsv41-exl3-head|dsv41-exl3-worker)$'"; then
  die "DeepSeek containers are still present on the worker. Stop them before starting GLM."
fi

note "Starting pinned GLM-5.3 Flash on both Sparks with the commercial MTP drafter"
head_ssh "cd '$GLM_RECIPE_DIR' && DRAFTER=mtp PREPARE=0 ./start.sh"

note "GLM start completed; running a real answer check"
response=$(head_ssh "curl -fsS --max-time 180 http://127.0.0.1:'$GLM_PORT'/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{\"model\":\"GLM-5.3-Flash-EXL3\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly GLM_READY\"}],\"max_tokens\":256,\"temperature\":0}'")
python3 - "$response" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
text = payload["choices"][0]["message"]["content"].strip()
if "GLM_READY" not in text:
    raise SystemExit(f"GLM returned an unexpected answer: {text!r}")
print(f"PASS: GLM answered: {text}")
PY

note "GLM is serving. MCDMA remains transport-ready but is not yet TensorFold's backend."
