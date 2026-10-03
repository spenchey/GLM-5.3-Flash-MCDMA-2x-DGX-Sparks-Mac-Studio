#!/usr/bin/env bash
set -euo pipefail

project=$(cd "$(dirname "$0")/.." && pwd)
if [[ $# -ne 1 ]]; then
  printf 'usage: %s EXACT_MCDMA_BASE_CHECKOUT\n' "$0" >&2
  exit 2
fi
base=$(cd "$1" && pwd)
temporary=$(mktemp -d "${TMPDIR:-/tmp}/mcdma-exact-verification-test.XXXXXX")
cleanup() { [[ ! -d "$temporary" ]] || find "$temporary" -depth -delete; }
trap cleanup EXIT INT TERM HUP

if "$project/scripts/prepare-mcdma-exact-verification.sh" "$project" "$temporary/wrong" >/dev/null 2>&1; then
  printf 'wrong base was accepted\n' >&2
  exit 1
fi
prepared=$temporary/prepared
"$project/scripts/prepare-mcdma-exact-verification.sh" "$base" "$prepared"

python3 - "$prepared" "$project/patches/mcdma/exact-verification-manifest.json" <<'PY'
import hashlib,json
from pathlib import Path
import sys
tree=Path(sys.argv[1]); expected=json.loads(Path(sys.argv[2]).read_text())
marker=json.loads((tree/'.mcdma-exact-verification.json').read_text())
assert marker==expected
for relative,hashes in expected['files'].items():
    actual=hashlib.sha256((tree/relative).read_bytes()).hexdigest()
    assert actual==hashes['patched_sha256'],(relative,actual)
PY

(
  cd "$prepared"
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
    tests.test_verify_windows tests.test_verify_receiver tests.test_bw_pattern_trial tests.test_native_endpoint tests.test_bw_tools tests.test_bw_guard \
    tests.test_bw_payload tests.test_bw_payload_trial tests.test_ndp_neighbor tests.test_native_observer_marker
)

if pgrep -f "$temporary" >/dev/null 2>&1; then
  printf 'test child survived cleanup scope: %s\n' "$temporary" >&2
  exit 1
fi
printf 'exact verification materialization tests passed\n'
