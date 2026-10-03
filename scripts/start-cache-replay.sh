#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config

digest=${1:?usage: start-cache-replay.sh PROMPT_SHA256}
[[ "$digest" =~ ^[0-9a-f]{64}$ ]] || die "prompt digest must be a lowercase SHA-256"

trap release_three_machine_lifecycle_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
acquire_three_machine_lifecycle_lock

container=$THREE_MACHINE_CACHE_REPLAY_CONTAINER
bundle="$THREE_MACHINE_CACHE_BUNDLE_DIR/$digest"
native=$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER

for role in head worker; do
  case "$role" in
    head) running=$(head_ssh "docker inspect '$native' --format '{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}' 2>/dev/null || true") ;;
    worker) running=$(worker_ssh "docker inspect '$native' --format '{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}' 2>/dev/null || true") ;;
  esac
  [[ "$running" == true\|false\|0 ]] || die "$role native TensorFold rank is not cleanly running"
done
[[ -z "$(head_ssh "docker ps -a --filter name=^/$container\$ --format '{{.Names}}'")" ]] \
  || die "$container already exists on the head Spark"
head_ssh "test -d '$bundle' && test ! -L '$bundle'" \
  || die "the verified prompt-cache bundle is missing"

started=false
container_id=""
complete=false
cleanup_failed_start() {
  local rc=$?
  if (( rc != 0 )) && [[ "$complete" != true ]]; then
    [[ -z "$container_id" ]] || head_ssh "docker logs '$container_id' 2>&1" >&2 || true
    [[ -z "$container_id" ]] || head_ssh "docker rm -f '$container_id' >/dev/null 2>&1" || true
    [[ "$started" != true ]] || adapter cleanup >/dev/null 2>&1 || true
  fi
  release_three_machine_lifecycle_lock
  return "$rc"
}
trap cleanup_failed_start EXIT

started=true
link=$(adapter snapshot)
mcdma_transport_healthy "$link" || die "MCDMA transport did not become healthy"
mcdma_link_busy "$link" && die "MCDMA is unexpectedly busy"
head_ssh "test -S '/tmp/mcdma-rpcd.$THREE_MACHINE_MAILBOX.sock'" \
  || die "the head Spark MCDMA control socket is absent"

command=(docker run -d --name "$container" --init --network=host --ipc=host
  --label "glm.mcdma.mode=cache-replay" --label "glm.mcdma.prompt=$digest"
  -e PYTHONPATH=/work -e MCDMA_RPC_LIBRARY=/opt/mcdma/libmcdma-rpc.so
  -v "$THREE_MACHINE_PROJECT_DIR:/work:ro"
  -v "$THREE_MACHINE_SPARK_MCDMA_LIBRARY:/opt/mcdma/libmcdma-rpc.so:ro"
  -v "$THREE_MACHINE_CACHE_BUNDLE_DIR:/cache-bundles:ro"
  --mount "type=bind,src=/tmp/mcdma-rpcd.$THREE_MACHINE_MAILBOX.sock,dst=/run/mcdma/control.sock,readonly"
  "$THREE_MACHINE_IMAGE_ID" python3 -m experiments.three_machine.cache_replay
  --bundle "/cache-bundles/$digest" --mailbox "$THREE_MACHINE_MAILBOX"
  --socket /run/mcdma/control.sock --timeout 600)
printf -v remote '%q ' "${command[@]}"
container_id=$(head_ssh "$remote")
[[ "$container_id" =~ ^[0-9a-f]{64}$ ]] || die "cache replay launch returned no container ID"

deadline=$((SECONDS + 120))
while (( SECONDS < deadline )); do
  state=$(head_ssh "docker inspect '$container_id' --format '{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}'")
  [[ "$state" == true\|false\|0 ]] || die "cache replay stopped during startup: $state"
  link=$(adapter peek 2>/dev/null || true)
  if [[ -n "$link" ]] && python3 - "$link" <<'PY'
import json, sys
state = json.loads(sys.argv[1])
raise SystemExit(0 if (
    state.get("link") == "up"
    and state.get("mac", {}).get("alive")
    and state.get("spark", {}).get("alive")
    and state.get("spark", {}).get("service") == "poll"
    and not state.get("mac", {}).get("failures")
    and not state.get("spark", {}).get("failures")
) else 1)
PY
  then
    complete=true
    note "PASS: verified TensorFold cache replay is ready beside the native service"
    exit 0
  fi
  sleep 1
done
die "cache replay did not become ready"
