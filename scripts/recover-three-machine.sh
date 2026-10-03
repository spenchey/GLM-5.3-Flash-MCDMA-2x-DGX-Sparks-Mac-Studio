#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_three_machine_config

trap release_three_machine_lifecycle_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
acquire_three_machine_lifecycle_lock

umask 077
recovery_started_epoch=$(date +%s)
recovery_budget_seconds=$(three_machine_recovery_budget_seconds)
recovery_deadline_epoch=$((recovery_started_epoch + recovery_budget_seconds))
recovery_outcome=failed
recovery_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
evidence_dir="$PROJECT_ROOT/results/raw/recovery-$recovery_id"
mkdir -p "$evidence_dir"
printf 'role\tforced_kill\n' >"$evidence_dir/forced-stops.tsv"
printf 'event\tutc\tepoch_s\telapsed_s\tremaining_budget_s\n' >"$evidence_dir/timing.tsv"
if [[ -n "${THREE_MACHINE_RECOVERY_BUDGET_SECONDS:-}" ]]; then
  recovery_budget_source=configured
elif [[ -n "${THREE_MACHINE_CLEAN_START_SECONDS:-}" ]]; then
  recovery_budget_source=measured-clean-start
else
  recovery_budget_source=safe-initial
fi
printf 'budget_seconds\t%s\nsource\t%s\nclean_start_seconds\t%s\n' \
  "$recovery_budget_seconds" "$recovery_budget_source" \
  "${THREE_MACHINE_CLEAN_START_SECONDS:-unavailable}" >"$evidence_dir/budget.tsv"

record_recovery_event() {
  local event=${1:?event required} now elapsed remaining
  now=$(date +%s)
  elapsed=$((now - recovery_started_epoch))
  remaining=$((recovery_deadline_epoch - now))
  (( remaining >= 0 )) || remaining=0
  printf '%s\t%s\t%s\t%s\t%s\n' "$event" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$now" "$elapsed" "$remaining" >>"$evidence_dir/timing.tsv"
}

recovery_budget_remaining() {
  local remaining=$((recovery_deadline_epoch - $(date +%s)))
  (( remaining > 0 )) || return 1
  printf '%s\n' "$remaining"
}

require_recovery_budget() {
  local phase=${1:?phase required}
  recovery_budget_remaining >/dev/null || {
    record_recovery_event "${phase}_budget_exceeded"
    die "recovery exceeded its ${recovery_budget_seconds}s budget during $phase"
  }
}

# shellcheck disable=SC1091
source "$(dirname "$0")/recovery-budget.sh"

finish_recovery() {
  local rc=$?
  set +e
  record_recovery_event recovery_end
  printf 'outcome\t%s\nexit_code\t%s\n' "$recovery_outcome" "$rc" \
    >>"$evidence_dir/budget.tsv"
  release_three_machine_lifecycle_lock
  return "$rc"
}
trap finish_recovery EXIT
record_recovery_event recovery_start
record_recovery_event detection_start

head_state=$(capture_recovery_command detection_head_state three_machine_container_state head) \
  || die "could not inspect the head Spark within the recovery budget"
worker_state=$(capture_recovery_command detection_worker_state three_machine_container_state worker) \
  || die "could not inspect the worker Spark within the recovery budget"
printf '%s\n' "$head_state" >"$evidence_dir/head-inspect.txt"
printf '%s\n' "$worker_state" >"$evidence_dir/worker-inspect.txt"
validate_three_machine_container_state "head Spark" "$head_state"
validate_three_machine_container_state "worker Spark" "$worker_state"
[[ -z "$head_state" ]] \
  || run_recovery_command detection_head_inspect "$evidence_dir/head-full-inspect.json" \
       "$evidence_dir/head-full-inspect.stderr" head_ssh \
       "docker inspect '$(three_machine_container_id "$head_state")'"
[[ -z "$worker_state" ]] \
  || run_recovery_command detection_worker_inspect "$evidence_dir/worker-full-inspect.json" \
       "$evidence_dir/worker-full-inspect.stderr" worker_ssh \
       "docker inspect '$(three_machine_container_id "$worker_state")'"
[[ -z "$head_state" ]] \
  || run_recovery_command detection_head_log "$evidence_dir/head.log" - head_ssh \
       "docker logs '$(three_machine_container_id "$head_state")' 2>&1"
[[ -z "$worker_state" ]] \
  || run_recovery_command detection_worker_log "$evidence_dir/worker.log" - worker_ssh \
       "docker logs '$(three_machine_container_id "$worker_state")' 2>&1"
set +e
adapter_state=$(capture_recovery_command detection_adapter adapter peek)
adapter_state_rc=$?
set -e
(( adapter_state_rc == 0 )) || adapter_state=
printf '%s\n' "$adapter_state" >"$evidence_dir/mcdma-before.json"
run_recovery_command detection_api_log "$evidence_dir/api.log" - studio_ssh \
  "test -r '$THREE_MACHINE_API_STATE_DIR/api.log' && cat '$THREE_MACHINE_API_STATE_DIR/api.log' || true"

head_running=false
worker_running=false
three_machine_container_running "$head_state" && head_running=true
three_machine_container_running "$worker_state" && worker_running=true
api_ready=false
healthy_api_pid=$(capture_recovery_command detection_api_pid probe_studio_api_pid) \
  || die "could not inspect the managed Mac API within the recovery budget"
api_port_open=false
if [[ -n "$healthy_api_pid" ]]; then
  set +e
  run_recovery_command detection_api_port "$evidence_dir/api-port.stdout" \
    "$evidence_dir/api-port.stderr" studio_api_port_open
  api_port_rc=$?
  set -e
  (( api_port_rc == 0 )) && api_port_open=true
fi
if [[ -n "$healthy_api_pid" && "$api_port_open" == true ]]; then
  listener_pid=$(capture_recovery_command detection_api_listener verified_studio_api_listener_pid) \
    || die "could not verify the Mac API listener within the recovery budget"
  [[ "$listener_pid" == "$healthy_api_pid" ]] \
    || die "the live API listener is not owned by the verified Studio process"
else
  healthy_api_pid=""
fi
if run_recovery_command detection_api_health "$evidence_dir/api-health-before.json" \
  "$evidence_dir/api-health-before.stderr" studio_ssh \
  "curl -fsS --max-time 3 'http://$THREE_MACHINE_API_HOST:$THREE_MACHINE_API_PORT/health'"; then
  if python3 - "$evidence_dir/api-health-before.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    data = json.load(stream)
raise SystemExit(0 if (
    data.get("status") == "ready"
    and data.get("model") == "GLM-5.3-Flash-TensorFold-MCDMA-3Machine"
) else 1)
PY
  then
    api_ready=true
  fi
fi
transport_healthy=false
if [[ -n "$adapter_state" ]] && python3 - "$adapter_state" <<'PY'
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
then
  transport_healthy=true
fi
record_recovery_event detection_end
require_recovery_budget detection
if [[ "$head_running" == true && "$worker_running" == true \
      && -n "$healthy_api_pid" && "$transport_healthy" == true \
      && "$api_ready" == true ]]; then
  die "the three-machine service is healthy; use stop-three-machine.sh for an orderly stop"
fi

stop_api() {
  local pid stale_pid stale_start match_rc deadline listeners process_state
  pid=$(probe_studio_api_pid)
  if [[ -z "$pid" ]]; then
    stale_pid=$(studio_api_pid_file_value)
    if [[ -n "$stale_pid" ]]; then
      stale_start=$(studio_api_start_file_value) \
        || die "could not read the stale Mac API start identity during recovery"
      [[ -n "$stale_start" ]] \
        || die "the stale Mac API identity is missing its process start time"
      if studio_process_start_matches "$stale_pid" "$stale_start"; then
        die "the recorded Mac API still owns its PID/start identity but cannot be verified; refusing to signal it"
      else
        match_rc=$?
      fi
      [[ "$match_rc" == 1 ]] \
        || die "could not inspect the stale Mac API identity during recovery"
    fi
    ! studio_api_port_open \
      || die "the API listener is active but its owner cannot be verified; refusing forced recovery"
    [[ -z "$stale_pid" ]] || clear_studio_api_identity "$stale_pid" "$stale_start"
    return 0
  fi
  note "Stopping the verified Mac API process"
  studio_ssh "kill -TERM '$pid'"
  deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    process_state=$(studio_process_state "$pid") \
      || die "could not inspect the Mac API during recovery"
    [[ "$process_state" != alive ]] && break
    sleep 1
  done
  process_state=$(studio_process_state "$pid") \
    || die "could not inspect the Mac API before forced recovery"
  if [[ "$process_state" == alive ]]; then
    [[ "$(verified_studio_api_pid)" == "$pid" ]] || die "Mac API pid changed identity before forced recovery"
    note "Mac API remained blocked after a rank loss; force-stopping only that verified process"
    printf 'mac-api\ttrue\n' >>"$evidence_dir/forced-stops.tsv"
    studio_ssh "kill -KILL '$pid'"
    deadline=$((SECONDS + 10))
    while (( SECONDS < deadline )); do
      process_state=$(studio_process_state "$pid") \
        || die "could not inspect the Mac API after forced recovery"
      [[ "$process_state" != alive ]] && break
      sleep 1
    done
    process_state=$(studio_process_state "$pid") \
      || die "could not inspect the Mac API after its forced recovery deadline"
    [[ "$process_state" != alive ]] \
      || die "verified Mac API survived the forced recovery stop"
  fi
  listeners=$(studio_ssh "lsof -nP -tiTCP@'$THREE_MACHINE_API_HOST':'$THREE_MACHINE_API_PORT' -sTCP:LISTEN 2>/dev/null || true")
  [[ -z "$listeners" ]] || die "Mac API listener survived recovery stop"
  studio_ssh "rm -f '$THREE_MACHINE_API_STATE_DIR/api.identity'"
}

stop_rank() {
  local role=$1 state id deadline
  state=$(three_machine_container_state "$role")
  validate_three_machine_container_state "$role Spark" "$state"
  [[ -n "$state" ]] || return 0
  three_machine_container_running "$state" || return 0
  id=$(three_machine_container_id "$state")
  note "Sending TERM to the verified $role Spark container"
  if [[ "$role" == head ]]; then
    head_ssh "docker kill --signal=TERM '$id' >/dev/null"
  else
    worker_ssh "docker kill --signal=TERM '$id' >/dev/null"
  fi
  deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    state=$(three_machine_container_state "$role")
    validate_three_machine_container_state "$role Spark" "$state"
    [[ -z "$state" || "$(three_machine_container_id "$state")" == "$id" ]] \
      || die "$role Spark container identity changed during recovery"
    three_machine_container_running "$state" || return 0
    sleep 1
  done
  note "$role Spark remained blocked in NCCL; force-stopping only $THREE_MACHINE_CONTAINER"
  printf '%s\ttrue\n' "$role-rank" >>"$evidence_dir/forced-stops.tsv"
  state=$(three_machine_container_state "$role")
  validate_three_machine_container_state "$role Spark" "$state"
  three_machine_container_running "$state" || return 0
  [[ "$(three_machine_container_id "$state")" == "$id" ]] \
    || die "$role Spark container identity changed before forced recovery"
  if [[ "$role" == head ]]; then
    head_ssh "docker kill '$id' >/dev/null"
  else
    worker_ssh "docker kill '$id' >/dev/null"
  fi
  deadline=$((SECONDS + 10))
  while (( SECONDS < deadline )); do
    state=$(three_machine_container_state "$role")
    validate_three_machine_container_state "$role Spark" "$state"
    [[ -z "$state" || "$(three_machine_container_id "$state")" == "$id" ]] \
      || die "$role Spark container identity changed after forced recovery"
    three_machine_container_running "$state" || return 0
    sleep 1
  done
  die "$role Spark container survived the forced recovery stop"
}

set +e
run_recovery_command api_stop "$evidence_dir/api-stop.log" - stop_api
api_stop_rc=$?
set -e
(( api_stop_rc == 0 )) \
  || die "Mac API stop failed or exceeded the recovery budget"
set +e
run_recovery_command head_stop "$evidence_dir/head-stop.log" - stop_rank head
head_stop_rc=$?
set -e
(( head_stop_rc == 0 )) \
  || die "head Spark stop failed or exceeded the recovery budget"
set +e
run_recovery_command worker_stop "$evidence_dir/worker-stop.log" - stop_rank worker
worker_stop_rc=$?
set -e
(( worker_stop_rc == 0 )) \
  || die "worker Spark stop failed or exceeded the recovery budget"

note "Cleaning the owned MCDMA connector, listener, and control tunnel"
set +e
run_recovery_command mcdma_cleanup "$evidence_dir/mcdma-cleanup-initial.json" \
  "$evidence_dir/mcdma-cleanup.stderr" adapter cleanup
cleanup_rc=$?
set -e
(( cleanup_rc == 0 )) \
  || die "MCDMA cleanup command failed; evidence: $evidence_dir/mcdma-cleanup.stderr"
cleanup=$(cat "$evidence_dir/mcdma-cleanup-initial.json")
socket_cleanup_needed=false
if [[ -n "$adapter_state" ]] && python3 - "$cleanup" <<'PY'
import json
import sys

cleanup = json.loads(sys.argv[1])
raise SystemExit(0 if any("RPC socket remains:" in item
                          for item in cleanup.get("problems", [])) else 1)
PY
then
  socket_cleanup_needed=true
fi
if [[ "$socket_cleanup_needed" == true ]]; then
  set +e
  run_recovery_command owned_socket_cleanup "$evidence_dir/socket-cleanup.tsv" \
    "$evidence_dir/socket-cleanup.stderr" clear_owned_mcdma_socket_paths "$adapter_state"
  socket_cleanup_rc=$?
  set -e
  (( socket_cleanup_rc == 0 )) \
    || die "owned MCDMA socket cleanup failed or exceeded the recovery budget"
  set +e
  run_recovery_command mcdma_cleanup_retry "$evidence_dir/mcdma-cleanup-retry.json" \
    "$evidence_dir/mcdma-cleanup-retry.stderr" adapter cleanup
  cleanup_rc=$?
  set -e
  (( cleanup_rc == 0 )) \
    || die "MCDMA cleanup retry failed; evidence: $evidence_dir/mcdma-cleanup.stderr"
  cleanup=$(cat "$evidence_dir/mcdma-cleanup-retry.json")
fi
printf '%s\n' "$cleanup" >"$evidence_dir/mcdma-cleanup.json"
validate_mcdma_cleanup "$cleanup"

note "Restarting the complete Mac Studio plus two-Spark service"
record_recovery_event restart_start
set +e
run_recovery_command restart "$evidence_dir/restart.log" - \
  "$PROJECT_ROOT/scripts/start-three-machine.sh"
restart_rc=$?
set -e
record_recovery_event restart_end
cat "$evidence_dir/restart.log"
(( restart_rc == 0 )) \
  || die "recovery restart failed or exceeded its ${recovery_budget_seconds}s budget"
record_recovery_event canary_start
set +e
run_recovery_command canary "$evidence_dir/canary.json" "$evidence_dir/canary.stderr" \
  "$PROJECT_ROOT/scripts/chat-three-machine.sh" "Reply with exactly THREE_MACHINE_READY" 64
canary_rc=$?
set -e
if (( canary_rc != 0 )); then
  record_recovery_event canary_end
  die "recovery canary failed or exceeded its ${recovery_budget_seconds}s budget; evidence: $evidence_dir/canary.stderr"
fi
set +e
python3 - "$evidence_dir/canary.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    data = json.load(stream)
text = data["choices"][0]["message"]["content"]
if text != "THREE_MACHINE_READY":
    raise SystemExit(f"recovery canary was not exact: {text!r}")
PY
canary_validation_rc=$?
set -e
record_recovery_event canary_end
(( canary_validation_rc == 0 )) || die "recovery canary answer was not exact"
set +e
run_recovery_command status "$evidence_dir/status.log" - \
  "$PROJECT_ROOT/scripts/status-three-machine.sh"
status_rc=$?
set -e
cat "$evidence_dir/status.log"
(( status_rc == 0 )) \
  || die "recovery status failed or exceeded its ${recovery_budget_seconds}s budget"
recovery_outcome=pass
note "PASS: the full three-machine service recovered and answered exactly; evidence: $evidence_dir"
