#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
state_dir=$(mktemp -d "${TMPDIR:-/tmp}/three-machine-launcher.XXXXXX")
child_pid=""
cleanup() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  rm -rf "$state_dir"
}
trap cleanup EXIT

"$project_root/scripts/studio-api-launcher.sh" "$state_dir" sleep 30 &
child_pid=$!
for _ in {1..100}; do
  [[ -s "$state_dir/api.identity" ]] && break
  sleep 0.01
done
[[ -s "$state_dir/api.identity" ]]
IFS=$'\t' read -r recorded_pid recorded_start <"$state_dir/api.identity"
[[ "$recorded_pid" == "$child_pid" ]]
[[ "$recorded_start" == "$(LC_ALL=C TZ=UTC ps -p "$child_pid" -o lstart=)" ]]
[[ "$(ps -p "$child_pid" -o command=)" == "sleep 30" ]]

printf 'PASS: Studio API launcher recorded one exact pre-exec identity\n'
