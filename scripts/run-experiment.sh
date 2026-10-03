#!/usr/bin/env bash
set -uo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  printf 'Usage: %s ID "short label" command [arg ...]\n' "$0" >&2
  exit 2
}

[[ $# -ge 3 ]] || usage
id=$1
label=$2
shift 2

[[ "$id" =~ ^[A-Z][A-Z0-9_-]*-[0-9]{3,}$ ]] || {
  printf 'ERROR: invalid experiment ID: %s\n' "$id" >&2
  exit 2
}

raw_dir=${RESULTS_RAW_DIR:-$root/results/raw}
mkdir -p "$raw_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log=$raw_dir/$stamp-$id.log

set +e
{
  printf 'experiment_id=%s\n' "$id"
  printf 'label=%s\n' "$label"
  printf 'started_utc=%s\n' "$stamp"
  "$@"
  rc=$?
  printf 'finished_utc=%s\n' "$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'exit_code=%d\n' "$rc"
  exit "$rc"
} 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e

if command -v shasum >/dev/null 2>&1; then
  digest=$(shasum -a 256 "$log" | awk '{print $1}')
else
  digest=$(sha256sum "$log" | awk '{print $1}')
fi

printf 'raw_log=%s\n' "$log"
printf 'raw_log_sha256=%s\n' "$digest"
exit "$rc"
