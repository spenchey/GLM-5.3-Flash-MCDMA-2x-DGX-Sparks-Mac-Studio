#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
helper=$root/scripts/prepare-mcdma-safe-benchmark.sh

if [[ $# -ne 1 ]]; then
  printf 'usage: %s EXACT_MCDMA_BASE_CHECKOUT\n' "$0" >&2
  exit 2
fi
source_dir=$(cd "$1" && pwd)

tmp=$(mktemp -d)
cleanup() {
  if [[ -d "$tmp" ]]; then
    rm -rf -- "$tmp"
  fi
}
trap cleanup EXIT INT TERM HUP

# A nearby repository at any other revision must be rejected before copying.
set +e
wrong_output=$($helper "$root" "$tmp/wrong-base" 2>&1)
wrong_code=$?
set -e
[[ $wrong_code -eq 1 ]]
grep -q 'base commit mismatch' <<<"$wrong_output"
[[ ! -e "$tmp/wrong-base" ]]

before=$(git -C "$source_dir" status --porcelain --untracked-files=no)
$helper "$source_dir" "$tmp/patched"
after=$(git -C "$source_dir" status --porcelain --untracked-files=no)
[[ "$before" == "$after" ]]

test -f "$tmp/patched/.mcdma-safe-benchmark.json"
! grep -Eq 'self\.process\.kill\(' "$tmp/patched/tools/native_cross_host.py"

PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  "$tmp/patched/tools/native_cross_host.py" \
  "$tmp/patched/benchmarks/run_bw.py" \
  "$tmp/patched/tests/test_native_endpoint.py" \
  "$tmp/patched/tests/test_bw_tools.py"
(
  cd "$tmp/patched"
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
    tests/test_native_endpoint.py tests/test_bw_tools.py
)

# The offline suite must leave no child of this harness alive.
if jobs -pr | grep -q .; then
  printf 'offline test child survived cleanup\n' >&2
  exit 1
fi

printf 'PASS: exact-base patch, hashes, signals, nonzero cleanup failure, and no forced kill\n'
