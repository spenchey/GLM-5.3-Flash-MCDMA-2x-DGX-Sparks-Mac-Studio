#!/usr/bin/env bash

# Record one crash-consistent process identity before replacing this launcher
# with the long-lived Python API. The PID and start time remain unchanged across
# exec, so lifecycle scripts can prove ownership without trusting a stale pid.
set -euo pipefail

if (( $# < 3 )); then
  printf 'usage: %s STATE_DIR COMMAND ARG...\n' "$0" >&2
  exit 64
fi

state_dir=$1
shift
umask 077
mkdir -p "$state_dir"
identity_tmp="$state_dir/.api.identity.$$"
trap 'rm -f "$identity_tmp"' EXIT
started=$(LC_ALL=C TZ=UTC ps -p "$$" -o lstart=)
[[ -n "$started" ]]
printf '%s\t%s\n' "$$" "$started" >"$identity_tmp"
mv -f "$identity_tmp" "$state_dir/api.identity"
trap - EXIT
exec "$@"
