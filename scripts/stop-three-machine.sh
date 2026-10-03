#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_three_machine_config

trap release_three_machine_lifecycle_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
acquire_three_machine_lifecycle_lock

head_state=$(three_machine_container_state head)
worker_state=$(three_machine_container_state worker)
validate_three_machine_container_state "head Spark" "$head_state"
validate_three_machine_container_state "worker Spark" "$worker_state"
head_id=""
worker_id=""
[[ -z "$head_state" ]] || head_id=$(three_machine_container_id "$head_state")
[[ -z "$worker_state" ]] || worker_id=$(three_machine_container_id "$worker_state")
head_running=false
worker_running=false
three_machine_container_running "$head_state" && head_running=true
three_machine_container_running "$worker_state" && worker_running=true
[[ "$head_running" == "$worker_running" ]] \
  || die "only one Spark stage is running; use recover-three-machine.sh"
if [[ "$head_running" == true ]]; then
  link=$(adapter peek) || die "could not inspect MCDMA before orderly shutdown"
  mcdma_link_healthy "$link" \
    || die "MCDMA is unhealthy; use recover-three-machine.sh instead of an orderly stop"
  mcdma_link_busy "$link" \
    && die "an inference request is active; wait for it to finish before stopping"
fi

api_pid=$(probe_studio_api_pid)
if [[ -z "$api_pid" ]]; then
  if studio_api_port_open; then
    die "the configured API port is active but its owner is not verified; use recover-three-machine.sh"
  fi
  studio_ssh "rm -f '$THREE_MACHINE_API_STATE_DIR/api.identity'"
fi
if [[ -n "$api_pid" ]]; then
  [[ "$(verified_studio_api_listener_pid)" == "$api_pid" ]] \
    || die "the verified Mac API does not own the configured listener"
  note "Stopping the Mac API; it will shut down both TensorFold ranks through MCDMA"
  studio_ssh "kill -TERM '$api_pid'"
  # A request that races the pre-stop busy check is drained by the API signal
  # handler. Allow the verified 2051-position dense ceiling to finish before recovery.
  api_deadline=$((SECONDS + 300))
  api_alive=true
  while (( SECONDS < api_deadline )); do
    api_state=$(studio_process_state "$api_pid") \
      || die "could not inspect the Mac API during orderly shutdown"
    [[ "$api_state" != alive ]] && api_alive=false && break
    sleep 1
  done
  [[ "$api_alive" != true ]] || die "Mac API survived its orderly shutdown deadline; use recover-three-machine.sh"
  ! studio_api_port_open \
    || die "the API listener survived its orderly shutdown deadline; use recover-three-machine.sh"
  studio_ssh "rm -f '$THREE_MACHINE_API_STATE_DIR/api.identity'"
fi

if [[ "$head_running" == true ]]; then
  if [[ -z "$api_pid" ]]; then
    note "Sending the model's orderly shutdown through MCDMA"
    remote="cd $(printf '%q' "$THREE_MACHINE_STUDIO_DIR") && env MCDMA_RPC_LIBRARY=$(printf '%q' "$THREE_MACHINE_MCDMA_LIBRARY") PYTHONPATH=$(printf '%q' "$THREE_MACHINE_STUDIO_DIR:$THREE_MACHINE_TENSORFOLD_SOURCE") $(printf '%q' "$THREE_MACHINE_STUDIO_PYTHON") experiments/three_machine/control.py shutdown --mailbox $(printf '%q' "$THREE_MACHINE_MAILBOX") --timeout 30"
    studio_ssh "$remote"
  fi

  deadline=$((SECONDS + 90))
  while (( SECONDS < deadline )); do
    head_state=$(three_machine_container_state head)
    worker_state=$(three_machine_container_state worker)
    validate_three_machine_container_state "head Spark" "$head_state"
    validate_three_machine_container_state "worker Spark" "$worker_state"
    [[ -z "$head_state" || "$(three_machine_container_id "$head_state")" == "$head_id" ]] \
      || die "head rank identity changed during shutdown"
    [[ -z "$worker_state" || "$(three_machine_container_id "$worker_state")" == "$worker_id" ]] \
      || die "worker rank identity changed during shutdown"
    if ! three_machine_container_running "$head_state" && ! three_machine_container_running "$worker_state"; then
      break
    fi
    sleep 1
  done
  ! three_machine_container_running "$head_state" && ! three_machine_container_running "$worker_state" \
    || die "a TensorFold rank survived orderly shutdown; use recover-three-machine.sh"
  [[ "$(head_ssh "docker inspect '$head_id' --format '{{.State.ExitCode}}'")" == 0 ]] || die "head rank did not exit cleanly"
  [[ "$(worker_ssh "docker inspect '$worker_id' --format '{{.State.ExitCode}}'")" == 0 ]] || die "worker rank did not exit cleanly"
else
  for stopped_state in "$head_state" "$worker_state"; do
    [[ -n "$stopped_state" ]] || continue
    IFS='|' read -r _name _id _image _running stopped_oom stopped_exit <<<"$stopped_state"
    [[ "$stopped_oom" == false && "$stopped_exit" == 0 ]] \
      || die "a stopped TensorFold rank records a failure; use recover-three-machine.sh"
  done
fi

if [[ -n "$head_state" || -n "$worker_state" ]]; then
  stop_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
  stop_evidence_dir="$PROJECT_ROOT/results/raw/stopped-ranks-$stop_id"
  umask 077
  mkdir -p "$stop_evidence_dir"
  printf '%s\n' "$head_state" >"$stop_evidence_dir/head-inspect.txt"
  printf '%s\n' "$worker_state" >"$stop_evidence_dir/worker-inspect.txt"
  [[ -z "$head_state" ]] \
    || head_ssh "docker inspect '$(three_machine_container_id "$head_state")'" >"$stop_evidence_dir/head-full-inspect.json"
  [[ -z "$worker_state" ]] \
    || worker_ssh "docker inspect '$(three_machine_container_id "$worker_state")'" >"$stop_evidence_dir/worker-full-inspect.json"
  [[ -z "$head_state" ]] \
    || head_ssh "docker logs '$(three_machine_container_id "$head_state")' 2>&1" >"$stop_evidence_dir/head.log"
  [[ -z "$worker_state" ]] \
    || worker_ssh "docker logs '$(three_machine_container_id "$worker_state")' 2>&1" >"$stop_evidence_dir/worker.log"
  note "Preserved stopped-rank evidence at $stop_evidence_dir"
fi

note "Stopping MCDMA connector, listener, and control tunnel"
cleanup=$(adapter cleanup)
validate_mcdma_cleanup "$cleanup"
python3 - "$cleanup" <<'PY'
import json, sys
s = json.loads(sys.argv[1])
if s.get("sigkill_used"):
    raise SystemExit(f"MCDMA cleanup was not clean: {s}")
PY
[[ -z "$head_state" ]] || head_ssh "docker rm '$(three_machine_container_id "$head_state")' >/dev/null"
[[ -z "$worker_state" ]] || worker_ssh "docker rm '$(three_machine_container_id "$worker_state")' >/dev/null"
[[ -z "$(three_machine_container_state head)" ]] || die "head rank container survived removal"
[[ -z "$(three_machine_container_state worker)" ]] || die "worker rank container survived removal"
! studio_api_port_open || die "the API listener is still active after shutdown"
note "PASS: TensorFold and MCDMA stopped cleanly with no owned programs left running"
