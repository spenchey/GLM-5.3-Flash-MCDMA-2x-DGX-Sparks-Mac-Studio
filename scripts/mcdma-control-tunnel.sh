#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=scripts/lib.sh
source "$root/scripts/lib.sh"

action=${1:-status}
[[ $# -eq 1 ]] || { printf 'Usage: %s {status|dry-run|start|stop}\n' "$0" >&2; exit 2; }
case "$action" in status|dry-run|start|stop) ;; *) printf 'Usage: %s {status|dry-run|start|stop}\n' "$0" >&2; exit 2 ;; esac

bind_address=127.0.0.1
port=${MCDMA_CONTROL_PORT:-18620}
[[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || die "invalid MCDMA_CONTROL_PORT: $port"
state_dir=${MCDMA_CONTROL_STATE_DIR:-$root/.state/mcdma-control-tunnel}
control_path=$state_dir/control.sock
ssh_target=${MCDMA_CONTROL_SSH_TARGET:-$HEAD_SSH}

ssh_base=(
  -o BatchMode=yes
  -o ExitOnForwardFailure=yes
  -o GatewayPorts=no
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=2
)

mode_of() {
  if stat -f '%Lp' "$1" >/dev/null 2>&1; then stat -f '%Lp' "$1"; else stat -c '%a' "$1"; fi
}

owner_of() {
  if stat -f '%Su' "$1" >/dev/null 2>&1; then stat -f '%Su' "$1"; else stat -c '%U' "$1"; fi
}

master_alive() {
  [[ -S "$control_path" ]] && ssh "${ssh_base[@]}" -S "$control_path" -O check "$ssh_target" >/dev/null 2>&1
}

assert_private_control_path() {
  [[ -S "$control_path" ]] || die "SSH control socket was not created"
  [[ "$(owner_of "$control_path")" == "$(id -un)" ]] || die "control socket is not owned by $(id -un)"
  local mode
  mode=$(mode_of "$control_path")
  [[ "$mode" == 600 || "$mode" == 700 ]] || die "control socket is not owner-only: mode $mode"
}

local_port_busy() {
  lsof -nP -iTCP@${bind_address}:${port} -sTCP:LISTEN -t 2>/dev/null | grep -q .
}

assert_private_state() {
  [[ ! -L "$state_dir" ]] || die "state directory must not be a symlink: $state_dir"
  [[ "$(owner_of "$state_dir")" == "$(id -un)" ]] || die "state directory is not owned by $(id -un): $state_dir"
  [[ "$(mode_of "$state_dir")" == 700 ]] || die "state directory mode is not 0700: $state_dir"
}

remote_exposure_check="set -eu
lines=\$(ss -ltnH \"sport = :$port\" 2>/dev/null || true)
test -n \"\$lines\" || { echo \"no Spark loopback listener on $bind_address:$port\" >&2; exit 1; }
bad=\$(printf \"%s\\n\" \"\$lines\" | awk '\$4 !~ /(^|\\])127\\.0\\.0\\.1:$port\$/ { print }')
test -z \"\$bad\" || { echo \"Spark port $port is exposed beyond loopback\" >&2; printf \"%s\\n\" \"\$bad\" >&2; exit 1; }"

case "$action" in
  dry-run)
    cat <<EOF
DRY-RUN: no SSH connection or listener will be created.
1. Require Spark control service only on $bind_address:$port via $ssh_target.
2. Refuse an existing local listener or SSH control path.
3. Create owner-only 0700 state at $state_dir.
4. Start authenticated SSH with ExitOnForwardFailure=yes and GatewayPorts=no.
5. Forward Mac $bind_address:$port to Spark $bind_address:$port.
6. Stop with the SSH control command and confirm listener/socket removal.
EOF
    ;;
  status)
    if master_alive; then
      [[ "$(mode_of "$state_dir")" == 700 ]] || die "active state directory mode is not 0700"
      assert_private_control_path
      local_port_busy || die "SSH master exists but $bind_address:$port is not listening"
      printf 'running address=%s port=%s target=%s state_mode=0700\n' "$bind_address" "$port" "$ssh_target"
    else
      [[ ! -e "$control_path" ]] || die "stale or conflicting control path: $control_path"
      local_port_busy && die "port conflict on $bind_address:$port"
      printf 'stopped address=%s port=%s target=%s\n' "$bind_address" "$port" "$ssh_target"
    fi
    ;;
  start)
    local_port_busy && die "port conflict on $bind_address:$port"
    mkdir -p -m 700 "$state_dir"
    chmod 700 "$state_dir"
    assert_private_state
    [[ ! -e "$control_path" ]] || die "control path conflict: $control_path"
    ssh "${ssh_base[@]}" "$ssh_target" "$remote_exposure_check"
    ssh "${ssh_base[@]}" -M -S "$control_path" -fN \
      -L "${bind_address}:${port}:${bind_address}:${port}" "$ssh_target"
    master_alive || die "SSH tunnel did not remain running"
    assert_private_control_path
    local_port_busy || die "SSH tunnel did not bind $bind_address:$port"
    printf 'started address=%s port=%s target=%s state_mode=0700\n' "$bind_address" "$port" "$ssh_target"
    ;;
  stop)
    if master_alive; then
      ssh "${ssh_base[@]}" -S "$control_path" -O exit "$ssh_target" >/dev/null
      for _ in 1 2 3 4 5; do
        ! local_port_busy && [[ ! -e "$control_path" ]] && break
        sleep 1
      done
    elif [[ -e "$control_path" ]]; then
      die "refusing to remove stale or conflicting control path: $control_path"
    fi
    local_port_busy && die "port $bind_address:$port is still listening after stop"
    [[ ! -e "$control_path" ]] || die "control path remains after stop: $control_path"
    rmdir "$state_dir" 2>/dev/null || true
    printf 'stopped address=%s port=%s target=%s cleanup=complete\n' "$bind_address" "$port" "$ssh_target"
    ;;
esac
