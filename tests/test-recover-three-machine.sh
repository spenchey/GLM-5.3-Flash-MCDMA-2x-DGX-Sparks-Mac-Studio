#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
live_pid=

cleanup() {
  if [[ -n "$live_pid" ]]; then
    kill "$live_pid" 2>/dev/null || true
    wait "$live_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT INT TERM

config=$tmp/config.env
printf '%s\n' \
  'HEAD_SSH=head' \
  'WORKER_SSH=worker' \
  'STUDIO_SSH=studio' \
  'GLM_RECIPE_DIR=/recipe' \
  'MCDMA_STUDIO_DIR=/mcdma' \
  'GLM_PORT=1' \
  'STUDIO_RDMA_DEVICE=studio-device' \
  'STUDIO_RDMA_INTERFACE=studio-interface' \
  'SPARK_RDMA_DEVICE=spark-device' \
  'SPARK_RDMA_INTERFACE=spark-interface' >"$config"
CONFIG_FILE=$config
# shellcheck disable=SC1091
source "$root/scripts/lib.sh"

unset THREE_MACHINE_RECOVERY_BUDGET_SECONDS THREE_MACHINE_CLEAN_START_SECONDS
[[ $(three_machine_recovery_budget_seconds) == 600 ]]
THREE_MACHINE_CLEAN_START_SECONDS=200
[[ $(three_machine_recovery_budget_seconds) == 480 ]]
THREE_MACHINE_CLEAN_START_SECONDS=201
[[ $(three_machine_recovery_budget_seconds) == 482 ]]
THREE_MACHINE_RECOVERY_BUDGET_SECONDS=725
[[ $(three_machine_recovery_budget_seconds) == 725 ]]

studio_ssh() {
  /bin/sh -c "$1"
}

dead_socket=$tmp/dead.sock
python3 - "$dead_socket" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_UNIX)
sock.bind(sys.argv[1])
sock.close()
PY
(exit 0) &
dead_pid=$!
wait "$dead_pid"
cleared=$(clear_owned_socket_after_dead_pid studio_ssh "$dead_pid" "$dead_socket")
[[ "$cleared" == $'studio_ssh\t'"$dead_pid"$'\t'"$dead_socket" ]]
[[ ! -e "$dead_socket" ]]

live_socket=$tmp/live.sock
python3 - "$live_socket" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_UNIX)
sock.bind(sys.argv[1])
sock.close()
PY
sleep 30 &
live_pid=$!
set +e
live_output=$(clear_owned_socket_after_dead_pid studio_ssh "$live_pid" "$live_socket" 2>&1)
live_rc=$?
set -e
[[ $live_rc -ne 0 ]]
grep -q 'while its exact owner PID .* is alive' <<<"$live_output"
[[ -S "$live_socket" ]]
kill "$live_pid"
wait "$live_pid" 2>/dev/null || true
live_pid=

non_socket=$tmp/not-a-socket
: >"$non_socket"
set +e
non_socket_output=$(clear_owned_socket_after_dead_pid studio_ssh "$dead_pid" "$non_socket" 2>&1)
non_socket_rc=$?
set -e
[[ $non_socket_rc -ne 0 ]]
grep -q 'because it is not a socket' <<<"$non_socket_output"
[[ -f "$non_socket" ]]

# A symlink to a socket is not an owned socket pathname.
socket_target=$tmp/target.sock
python3 - "$socket_target" <<'PY'
import socket
import sys
sock = socket.socket(socket.AF_UNIX)
sock.bind(sys.argv[1])
sock.close()
PY
ln -s "$socket_target" "$tmp/link.sock"
set +e
link_output=$(clear_owned_socket_after_dead_pid studio_ssh "$dead_pid" "$tmp/link.sock" 2>&1)
link_rc=$?
set -e
[[ $link_rc -ne 0 ]]
[[ -L "$tmp/link.sock" && -S "$socket_target" ]]

# Signal permission failure must not be mistaken for proof of a dead owner.
sleep 30 &
live_pid=$!
studio_ssh() {
  /bin/sh -c 'kill() { return 1; }; eval "$1"' sh "$1"
}
set +e
denied_output=$(clear_owned_socket_after_dead_pid studio_ssh "$live_pid" "$socket_target" 2>&1)
denied_rc=$?
set -e
[[ $denied_rc -ne 0 ]]
grep -q 'while its exact owner PID .* is alive' <<<"$denied_output"
[[ -S "$socket_target" ]]
kill "$live_pid"
wait "$live_pid" 2>/dev/null || true
live_pid=

set +e
missing_pid_output=$(clear_owned_mcdma_socket_paths '{"mac":{"pid":0},"spark":{"pid":0}}' 2>&1)
missing_pid_rc=$?
set -e
[[ $missing_pid_rc -ne 0 ]]
grep -q 'could not prove the owned MCDMA socket PIDs' <<<"$missing_pid_output"

# The command supervisor must turn a budget expiry into exit 124 and kill the
# exact owned process tree, including a child that ignores TERM.
# shellcheck disable=SC1091
source "$root/scripts/recovery-budget.sh"
recovery_budget_seconds=1
THREE_MACHINE_RECOVERY_TERM_GRACE_SECONDS=1
budget_checks=0
recovery_budget_remaining() {
  budget_checks=$((budget_checks + 1))
  (( budget_checks <= 2 ))
}
recovery_event=$tmp/recovery-event.txt
record_recovery_event() {
  printf '%s\n' "$1" >>"$recovery_event"
}
stubborn_parent=$tmp/stubborn-parent.pid
stubborn_child=$tmp/stubborn-child.pid
stubborn_code='import os,signal,sys,time,pathlib
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
child=os.fork()
if child == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pathlib.Path(sys.argv[2]).write_text(str(os.getpid()))
    time.sleep(30)
    raise SystemExit(0)
time.sleep(30)'
set +e
run_recovery_command forced-timeout "$tmp/forced.stdout" "$tmp/forced.stderr" \
  python3 -c "$stubborn_code" "$stubborn_parent" "$stubborn_child"
timeout_rc=$?
set -e
[[ $timeout_rc -eq 124 ]]
grep -qx 'forced-timeout_budget_exceeded' "$recovery_event"
[[ -s "$stubborn_parent" && -s "$stubborn_child" ]]
for pid_file in "$stubborn_parent" "$stubborn_child"; do
  owned_pid=$(<"$pid_file")
  for _ in {1..20}; do
    kill -0 "$owned_pid" 2>/dev/null || break
    sleep 0.1
  done
  ! kill -0 "$owned_pid" 2>/dev/null
done

printf 'PASS: recovery budgets, owned-tree timeout, and dead-owner-only socket cleanup\n'
