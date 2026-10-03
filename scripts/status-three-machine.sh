#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_three_machine_config

healthy=true
link_busy=false

note "Head Spark"
head_state=$(three_machine_container_state head)
validate_three_machine_container_state "head Spark" "$head_state"
if [[ -n "$head_state" ]]; then
  IFS='|' read -r _name _id head_image head_running head_oom head_exit <<<"$head_state"
  printf 'running=%s oom=%s exit=%s image=%s\n' "$head_running" "$head_oom" "$head_exit" "$head_image"
  [[ "$head_running" == true && "$head_oom" == false ]] || healthy=false
else
  echo running=false
  healthy=false
fi
note "Worker Spark"
worker_state=$(three_machine_container_state worker)
validate_three_machine_container_state "worker Spark" "$worker_state"
if [[ -n "$worker_state" ]]; then
  IFS='|' read -r _name _id worker_image worker_running worker_oom worker_exit <<<"$worker_state"
  printf 'running=%s oom=%s exit=%s image=%s\n' "$worker_running" "$worker_oom" "$worker_exit" "$worker_image"
  [[ "$worker_running" == true && "$worker_oom" == false ]] || healthy=false
else
  echo running=false
  healthy=false
fi
note "Mac Studio to Spark MCDMA"
if link=$(adapter peek 2>/dev/null); then
  printf '%s\n' "$link"
  python3 - "$link" <<'PY' || healthy=false
import json, sys
s = json.loads(sys.argv[1])
raise SystemExit(0 if (
    s.get("link") == "up"
    and s.get("mac", {}).get("alive")
    and s.get("spark", {}).get("alive")
    and s.get("spark", {}).get("service") == "poll"
    and s.get("mac", {}).get("failures") == 0
    and s.get("spark", {}).get("failures") == 0
) else 1)
PY
  mcdma_link_busy "$link" && link_busy=true
else
  echo running=false
  healthy=false
fi
note "Persistent API on the Mac Studio"
api_pid=""
if ! api_pid=$(verified_studio_api_listener_pid 2>/dev/null); then
  healthy=false
  api_pid=""
fi
if [[ -n "$api_pid" ]] \
    && api_health=$(studio_ssh "curl -fsS --max-time 3 'http://$THREE_MACHINE_API_HOST:$THREE_MACHINE_API_PORT/health'" 2>/dev/null); then
  printf '%s\n' "$api_health"
  python3 - "$api_health" <<'PY' || healthy=false
import json, sys
s = json.loads(sys.argv[1])
raise SystemExit(0 if (
    s.get("status") == "ready"
    and s.get("model") == "GLM-5.3-Flash-TensorFold-MCDMA-3Machine"
) else 1)
PY
elif [[ -n "$api_pid" && "$link_busy" == true ]]; then
  printf '{"status":"busy","model":"GLM-5.3-Flash-TensorFold-MCDMA-3Machine"}\n'
else
  echo running=false
  healthy=false
fi

[[ "$healthy" == true ]] || die "three-machine service is not fully healthy"
note "PASS: the Mac API/Metal stage, MCDMA, and both TensorFold CUDA ranks are healthy"
