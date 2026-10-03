#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config

trap release_three_machine_lifecycle_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
acquire_three_machine_lifecycle_lock

container=$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER
stop_role() {
  local role=$1 state id image mode command
  case "$role" in
    head) state=$(head_ssh "docker inspect '$container' --format '{{.Id}}|{{.Image}}|{{index .Config.Labels \"glm.mcdma.mode\"}}|{{.State.Running}}|{{.State.OOMKilled}}' 2>/dev/null || true") ;;
    worker) state=$(worker_ssh "docker inspect '$container' --format '{{.Id}}|{{.Image}}|{{index .Config.Labels \"glm.mcdma.mode\"}}|{{.State.Running}}|{{.State.OOMKilled}}' 2>/dev/null || true") ;;
    *) die "unknown role $role" ;;
  esac
  [[ -n "$state" ]] || return 0
  IFS='|' read -r id image mode _running _oom <<<"$state"
  [[ "$id" =~ ^[0-9a-f]{64}$ && "$image" == "$THREE_MACHINE_IMAGE_ID" \
      && "$mode" == native-portable ]] \
    || die "refusing to stop an unowned $role container: $state"
  command="docker stop --time 60 '$id' >/dev/null && docker rm '$id' >/dev/null"
  case "$role" in
    head) head_ssh "$command" ;;
    worker) worker_ssh "$command" ;;
  esac
}

stop_role head
stop_role worker
note "PASS: the temporary same-model two-Spark reference is stopped"
