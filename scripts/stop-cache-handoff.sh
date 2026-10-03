#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config

trap release_three_machine_lifecycle_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
acquire_three_machine_lifecycle_lock

head_state=$(cache_handoff_container_state head)
worker_state=$(cache_handoff_container_state worker)
validate_cache_handoff_container_state "head Spark" "$head_state"
validate_cache_handoff_container_state "worker Spark" "$worker_state"

head_id=""
worker_id=""
[[ -z "$head_state" ]] || head_id=$(cache_handoff_container_id "$head_state")
[[ -z "$worker_state" ]] || worker_id=$(cache_handoff_container_id "$worker_state")
head_running=false
worker_running=false
cache_handoff_container_running "$head_state" && head_running=true
cache_handoff_container_running "$worker_state" && worker_running=true
[[ "$head_running" == "$worker_running" ]] \
  || die "only one Spark cache-handoff rank is running; refusing an ambiguous shutdown"

if [[ "$head_running" == true ]]; then
  link=$(adapter peek) || die "could not inspect MCDMA before orderly shutdown"
  mcdma_link_healthy "$link" \
    || die "MCDMA is unhealthy; refusing to signal an unverified process"
  mcdma_link_busy "$link" \
    && die "a cache transfer is active; wait for it to finish before stopping"

  note "Stopping both Spark prompt ranks through the verified MCDMA path"
  command=(env
    "MCDMA_RPC_LIBRARY=$THREE_MACHINE_MCDMA_LIBRARY"
    "PYTHONPATH=$THREE_MACHINE_STUDIO_DIR:$THREE_MACHINE_TENSORFOLD_SOURCE"
    "$THREE_MACHINE_STUDIO_PYTHON" -m experiments.three_machine.cache_control shutdown
    --mailbox "$THREE_MACHINE_MAILBOX" --timeout 30)
  printf -v remote 'cd %q && ' "$THREE_MACHINE_STUDIO_DIR"
  printf -v command_text '%q ' "${command[@]}"
  studio_ssh "$remote$command_text"

  deadline=$((SECONDS + 120))
  while (( SECONDS < deadline )); do
    head_state=$(cache_handoff_container_state head)
    worker_state=$(cache_handoff_container_state worker)
    validate_cache_handoff_container_state "head Spark" "$head_state"
    validate_cache_handoff_container_state "worker Spark" "$worker_state"
    [[ -z "$head_state" || "$(cache_handoff_container_id "$head_state")" == "$head_id" ]] \
      || die "head rank identity changed during shutdown"
    [[ -z "$worker_state" || "$(cache_handoff_container_id "$worker_state")" == "$worker_id" ]] \
      || die "worker rank identity changed during shutdown"
    if ! cache_handoff_container_running "$head_state" \
        && ! cache_handoff_container_running "$worker_state"; then
      break
    fi
    sleep 1
  done
  ! cache_handoff_container_running "$head_state" \
    && ! cache_handoff_container_running "$worker_state" \
    || die "a Spark cache-handoff rank survived the orderly shutdown deadline"
fi

for role_state in "$head_state" "$worker_state"; do
  [[ -n "$role_state" ]] || continue
  IFS='|' read -r _name _id _image _running stopped_oom stopped_exit _mode <<<"$role_state"
  [[ "$stopped_oom" == false && "$stopped_exit" == 0 ]] \
    || die "a stopped cache-handoff rank records a failure"
done

if [[ -n "$head_state" || -n "$worker_state" ]]; then
  stop_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
  evidence="$PROJECT_ROOT/results/raw/stopped-cache-handoff-$stop_id"
  umask 077
  mkdir -p "$evidence"
  printf '%s\n' "$head_state" >"$evidence/head-state.txt"
  printf '%s\n' "$worker_state" >"$evidence/worker-state.txt"
  [[ -z "$head_id" ]] || head_ssh "docker inspect '$head_id'" >"$evidence/head-inspect.json"
  [[ -z "$worker_id" ]] || worker_ssh "docker inspect '$worker_id'" >"$evidence/worker-inspect.json"
  [[ -z "$head_id" ]] || head_ssh "docker logs '$head_id' 2>&1" >"$evidence/head.log"
  [[ -z "$worker_id" ]] || worker_ssh "docker logs '$worker_id' 2>&1" >"$evidence/worker.log"
  note "Preserved shutdown evidence at $evidence"
fi

note "Stopping the MCDMA connector, listener, and control tunnel"
cleanup=$(adapter cleanup)
validate_mcdma_cleanup "$cleanup"
python3 - "$cleanup" <<'PY'
import json
import sys

state = json.loads(sys.argv[1])
if state.get("sigkill_used"):
    raise SystemExit(f"MCDMA cleanup was not orderly: {state}")
PY

[[ -z "$head_id" ]] || head_ssh "docker rm '$head_id' >/dev/null"
[[ -z "$worker_id" ]] || worker_ssh "docker rm '$worker_id' >/dev/null"
[[ -z "$(cache_handoff_container_state head)" ]] \
  || die "head cache-handoff container survived removal"
[[ -z "$(cache_handoff_container_state worker)" ]] \
  || die "worker cache-handoff container survived removal"
note "PASS: Spark prompt ranks and MCDMA stopped cleanly with no owned programs left running"
