#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
helper=$root/scripts/mcdma-control-tunnel.sh
tmp=$(mktemp -d)
state=$tmp/state
remote_dir=
remote_pid=
local_pid=
test_port=28620
test_host=${MCDMA_TUNNEL_TEST_HOST:-}
if [[ -z "$test_host" && -f "$root/config.local.env" ]]; then
  # shellcheck disable=SC1091
  source "$root/config.local.env"
  test_host=${HEAD_SSH:-}
fi
[[ -n "$test_host" ]] || { printf 'MCDMA_TUNNEL_TEST_HOST or HEAD_SSH is required\n' >&2; exit 2; }

helper_run() {
  MCDMA_CONTROL_STATE_DIR=$state MCDMA_CONTROL_PORT=$test_port \
    MCDMA_CONTROL_SSH_TARGET=$test_host "$helper" "$@"
}

cleanup() {
  helper_run stop >/dev/null 2>&1 || true
  if [[ -n "$local_pid" ]]; then kill "$local_pid" 2>/dev/null || true; wait "$local_pid" 2>/dev/null || true; fi
  if [[ -n "$remote_pid" ]]; then
    ssh -o BatchMode=yes "$test_host" "kill '$remote_pid' 2>/dev/null || true; wait '$remote_pid' 2>/dev/null || true; rm -rf '$remote_dir'" >/dev/null 2>&1 || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT INT TERM

dry=$(helper_run dry-run)
grep -q '^DRY-RUN: no SSH connection or listener will be created\.$' <<<"$dry"
grep -q 'ExitOnForwardFailure=yes and GatewayPorts=no' <<<"$dry"
[[ ! -e "$state" ]]

status=$(helper_run status)
grep -q "^stopped address=127\\.0\\.0\\.1 port=$test_port " <<<"$status"

# A process already using the fixed Mac port must be rejected before SSH starts.
python3 -m http.server "$test_port" --bind 127.0.0.1 --directory "$tmp" >"$tmp/local-http.log" 2>&1 &
local_pid=$!
for _ in 1 2 3 4 5; do lsof -nP -iTCP@127.0.0.1:"$test_port" -sTCP:LISTEN -t >/dev/null 2>&1 && break; sleep 1; done
set +e
conflict_output=$(helper_run start 2>&1)
conflict_rc=$?
set -e
[[ $conflict_rc -ne 0 ]]
grep -q "port conflict on 127.0.0.1:$test_port" <<<"$conflict_output"
kill "$local_pid"
wait "$local_pid" 2>/dev/null || true
local_pid=

# A pre-existing control path is never silently replaced or removed.
mkdir -m 700 "$state"
: >"$state/control.sock"
set +e
path_output=$(helper_run start 2>&1)
path_rc=$?
set -e
[[ $path_rc -ne 0 ]]
grep -q 'control path conflict' <<<"$path_output"
rm "$state/control.sock"
rmdir "$state"

# Run a bounded, unprivileged Spark fixture on loopback only.
fixture_info=$(ssh -o BatchMode=yes "$test_host" bash -s -- "$test_port" <<'REMOTE'
set -eu
port=$1
test -z "$(ss -ltnH "sport = :$port" 2>/dev/null || true)"
d=$(mktemp -d /tmp/mcdma-tunnel-fixture.XXXXXX)
chmod 700 "$d"
printf 'MCDMA_TUNNEL_ROUND_TRIP\n' >"$d/proof"
nohup python3 -m http.server "$port" --bind 127.0.0.1 --directory "$d" >"$d/server.log" 2>&1 </dev/null &
p=$!
for _ in 1 2 3 4 5; do ss -ltnH "sport = :$port" | grep -q "127.0.0.1:$port" && break; sleep 1; done
ss -ltnH "sport = :$port" | grep -q "127.0.0.1:$port"
printf '%s|%s\n' "$d" "$p"
REMOTE
)
IFS='|' read -r remote_dir remote_pid <<<"$fixture_info"
[[ -n "$remote_dir" && -n "$remote_pid" ]]

start=$(helper_run start)
grep -q "^started address=127\\.0\\.0\\.1 port=$test_port " <<<"$start"
[[ $(stat -f '%Lp' "$state") == 700 ]]
[[ -S "$state/control.sock" ]]
[[ $(stat -f '%Su' "$state/control.sock") == "$(id -un)" ]]
socket_mode=$(stat -f '%Lp' "$state/control.sock")
[[ "$socket_mode" == 600 || "$socket_mode" == 700 ]]
running=$(helper_run status)
grep -q "^running address=127\\.0\\.0\\.1 port=$test_port " <<<"$running"

proof=$(curl --noproxy '*' -fsS --max-time 5 "http://127.0.0.1:$test_port/proof")
[[ "$proof" == MCDMA_TUNNEL_ROUND_TRIP ]]

# Both listeners must be loopback-only; every non-loopback Mac address must fail.
[[ -z "$(lsof -nP -iTCP:"$test_port" -sTCP:LISTEN | awk -v expected="127.0.0.1:$test_port" 'NR > 1 && $9 != expected')" ]]
ssh -o BatchMode=yes "$test_host" bash -s -- "$test_port" <<'REMOTE'
set -eu
port=$1
lines=$(ss -ltnH "sport = :$port")
test -n "$lines"
test -z "$(printf '%s\n' "$lines" | awk -v port="$port" '$4 !~ ("(^|\\])127\\.0\\.0\\.1:" port "$") { print }')"
REMOTE
while IFS= read -r address; do
  [[ -n "$address" ]] || continue
  if curl --noproxy '*' -fsS --connect-timeout 1 --max-time 2 "http://$address:$test_port/proof" >/dev/null 2>&1; then
    printf 'non-loopback address unexpectedly reached tunnel: %s\n' "$address" >&2
    exit 1
  fi
done < <(ifconfig | awk '/inet / && $2 != "127.0.0.1" { print $2 }')

stop=$(helper_run stop)
grep -q 'cleanup=complete' <<<"$stop"
[[ ! -e "$state/control.sock" ]]
[[ ! -d "$state" ]]
! lsof -nP -iTCP@127.0.0.1:"$test_port" -sTCP:LISTEN -t >/dev/null 2>&1
if curl --noproxy '*' -fsS --connect-timeout 1 --max-time 2 "http://127.0.0.1:$test_port/proof" >/dev/null 2>&1; then
  printf 'tunnel did not fail closed after stop\n' >&2
  exit 1
fi

ssh -o BatchMode=yes "$test_host" "kill '$remote_pid'; for i in 1 2 3 4 5; do ! ps -p '$remote_pid' >/dev/null 2>&1 && break; sleep 1; done; rm -rf '$remote_dir'; test -z \"\$(ss -ltnH 'sport = :$test_port' 2>/dev/null || true)\"; ! ps -p '$remote_pid' >/dev/null 2>&1"
remote_pid=
remote_dir=

printf 'PASS: authenticated loopback tunnel round-trip, isolation, fail-closed stop, conflicts, and cleanup\n'
