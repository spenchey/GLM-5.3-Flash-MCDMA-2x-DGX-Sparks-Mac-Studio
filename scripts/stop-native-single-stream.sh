#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config

trap release_three_machine_lifecycle_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
acquire_three_machine_lifecycle_lock

container=${THREE_MACHINE_SINGLE_STREAM_CONTAINER:-glm53-single-stream-native}

stop_role() {
  local role=$1 state id image mode command
  case "$role" in
    head) state=$(head_ssh "docker inspect '$container' --format '{{.Id}}|{{.Image}}|{{index .Config.Labels \"glm.mcdma.mode\"}}|{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}' 2>/dev/null || true") ;;
    worker) state=$(worker_ssh "docker inspect '$container' --format '{{.Id}}|{{.Image}}|{{index .Config.Labels \"glm.mcdma.mode\"}}|{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}' 2>/dev/null || true") ;;
    *) die "unknown role $role" ;;
  esac
  [[ -n "$state" ]] || return 0
  IFS='|' read -r id image mode _running oom _exit <<<"$state"
  [[ "$id" =~ ^[0-9a-f]{64}$ && "$image" == "$THREE_MACHINE_IMAGE_ID" \
      && "$mode" == native-single-stream-v065 ]] \
    || die "refusing to stop an unowned $role container: $state"
  [[ "$oom" == false ]] || die "$role single-stream container recorded an OOM"
  command="docker stop --time 60 '$id' >/dev/null && docker rm '$id' >/dev/null"
  case "$role" in
    head) head_ssh "$command" ;;
    worker) worker_ssh "$command" ;;
  esac
}

stop_role head
stop_role worker
note "PASS: TensorFold 0.6.5 one-request reference is stopped"
