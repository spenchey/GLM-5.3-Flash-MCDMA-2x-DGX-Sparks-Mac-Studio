#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config

trap release_three_machine_lifecycle_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
acquire_three_machine_lifecycle_lock

container=$THREE_MACHINE_CACHE_REPLAY_CONTAINER
state=$(head_ssh "docker inspect '$container' --format '{{.Id}}|{{.Image}}|{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}|{{index .Config.Labels \"glm.mcdma.mode\"}}' 2>/dev/null || true")
container_id=""
if [[ -n "$state" ]]; then
  IFS='|' read -r container_id image running oom exit_code mode <<<"$state"
  [[ "$container_id" =~ ^[0-9a-f]{64}$ && "$image" == "$THREE_MACHINE_IMAGE_ID" && "$mode" == cache-replay ]] \
    || die "refusing to stop an unverified cache replay container"
  if [[ "$running" == true ]]; then
    link=$(adapter peek)
    mcdma_link_healthy "$link" || die "MCDMA is unhealthy before cache replay shutdown"
    mcdma_link_busy "$link" && die "MCDMA is busy; wait before stopping cache replay"
    command=(env
      "MCDMA_RPC_LIBRARY=$THREE_MACHINE_MCDMA_LIBRARY"
      "PYTHONPATH=$THREE_MACHINE_STUDIO_DIR:$THREE_MACHINE_TENSORFOLD_SOURCE"
      "$THREE_MACHINE_STUDIO_PYTHON" -m experiments.three_machine.cache_control shutdown
      --mailbox "$THREE_MACHINE_MAILBOX" --timeout 30)
    printf -v remote 'cd %q && ' "$THREE_MACHINE_STUDIO_DIR"
    printf -v command_text '%q ' "${command[@]}"
    studio_ssh "$remote$command_text"
    deadline=$((SECONDS + 60))
    while (( SECONDS < deadline )); do
      running=$(head_ssh "docker inspect '$container_id' --format '{{.State.Running}}' 2>/dev/null || true")
      [[ "$running" != true ]] && break
      sleep 1
    done
    [[ "$running" != true ]] || die "cache replay survived its orderly shutdown deadline"
  fi
  stopped=$(head_ssh "docker inspect '$container_id' --format '{{.State.OOMKilled}}|{{.State.ExitCode}}'")
  [[ "$stopped" == false\|0 ]] || die "cache replay records a failed stop: $stopped"
  head_ssh "docker rm '$container_id' >/dev/null"
fi

cleanup=$(adapter cleanup)
validate_mcdma_cleanup "$cleanup"
python3 - "$cleanup" <<'PY'
import json, sys
state = json.loads(sys.argv[1])
if state.get("sigkill_used"):
    raise SystemExit(f"MCDMA cleanup was not orderly: {state}")
PY
note "PASS: cache replay and MCDMA stopped cleanly; native TensorFold was untouched"
