#!/usr/bin/env bash
# Shared, source-only recovery command supervision.

terminate_recovery_tree() {
  local signal=${1:?signal required} owner_pid=${2:?owner PID required}
  local child
  [[ "$owner_pid" =~ ^[0-9]+$ && "$owner_pid" -gt 1 ]] || return 1
  while IFS= read -r child; do
    [[ -n "$child" ]] || continue
    terminate_recovery_tree "$signal" "$child" || true
  done < <(pgrep -P "$owner_pid" 2>/dev/null || true)
  kill "-$signal" "$owner_pid" 2>/dev/null || true
}

run_recovery_command() {
  local label=${1:?label required} stdout_path=${2:?stdout path required}
  local stderr_path=${3:?stderr path required}
  shift 3
  local child_pid rc timed_out=false stop_deadline restore_errexit=false
  local term_grace=${THREE_MACHINE_RECOVERY_TERM_GRACE_SECONDS:-30}
  [[ "$term_grace" =~ ^[0-9]+$ && "$term_grace" -ge 1 ]] \
    || die "THREE_MACHINE_RECOVERY_TERM_GRACE_SECONDS must be a positive integer"
  recovery_budget_remaining >/dev/null \
    || die "recovery exceeded its ${recovery_budget_seconds}s budget before $label"
  if [[ "$stderr_path" == - ]]; then
    "$@" >"$stdout_path" 2>&1 &
  else
    "$@" >"$stdout_path" 2>"$stderr_path" &
  fi
  child_pid=$!
  while kill -0 "$child_pid" 2>/dev/null; do
    if ! recovery_budget_remaining >/dev/null; then
      timed_out=true
      record_recovery_event "${label}_budget_exceeded"
      terminate_recovery_tree TERM "$child_pid"
      stop_deadline=$((SECONDS + term_grace))
      while (( SECONDS < stop_deadline )) && kill -0 "$child_pid" 2>/dev/null; do
        sleep 1
      done
      if kill -0 "$child_pid" 2>/dev/null; then
        terminate_recovery_tree KILL "$child_pid"
      fi
      break
    fi
    sleep 1
  done
  [[ $- != *e* ]] || restore_errexit=true
  set +e
  wait "$child_pid"
  rc=$?
  [[ "$restore_errexit" != true ]] || set -e
  [[ "$timed_out" != true ]] || return 124
  return "$rc"
}

capture_recovery_command() {
  local label=${1:?label required}
  shift
  local stdout_path="$evidence_dir/$label.stdout"
  local stderr_path="$evidence_dir/$label.stderr"
  run_recovery_command "$label" "$stdout_path" "$stderr_path" "$@" || return $?
  cat "$stdout_path"
}
