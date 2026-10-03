#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
manifest=$root/patches/mcdma/exact-verification-manifest.json
cleanup_patch=$root/patches/mcdma/0001-safe-benchmark-cleanup.patch
verification_patch=$root/patches/mcdma/0002-exact-verification-windows.patch

if [[ $# -ne 2 ]]; then
  printf 'usage: %s EXACT_BASE_CHECKOUT NEW_ISOLATED_DIRECTORY\n' "$0" >&2
  exit 2
fi

source_dir=$(cd "$1" && pwd)
destination=$2
parent=$(cd "$(dirname "$destination")" && pwd)
destination=$parent/$(basename "$destination")
if [[ -e "$destination" || "$source_dir" == "$destination" ]]; then
  printf 'destination must be new and differ from source: %s\n' "$destination" >&2
  exit 2
fi

temporary=$(mktemp -d "$parent/.mcdma-exact-verification.XXXXXX")
cleanup() {
  if [[ -n "${temporary:-}" && -d "$temporary" ]]; then
    find "$temporary" -depth -delete
  fi
}
trap cleanup EXIT INT TERM HUP

verify_phase() {
  local tree=$1 phase=$2
  python3 - "$tree" "$manifest" "$cleanup_patch" "$verification_patch" "$phase" <<'PY'
import hashlib,json
from pathlib import Path
import sys

tree=Path(sys.argv[1]); manifest_path=Path(sys.argv[2])
cleanup_patch=Path(sys.argv[3]); verification_patch=Path(sys.argv[4]); phase=sys.argv[5]
manifest=json.loads(manifest_path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
for expected,path in zip(manifest['patches'],(cleanup_patch,verification_patch)):
    if path.name!=expected['path'] or digest(path)!=expected['sha256']:
        raise SystemExit(f'patch identity mismatch: {path}')
for relative,hashes in manifest['files'].items():
    path=tree/relative
    expected=hashes[f'{phase}_sha256']
    if expected is None:
        if path.exists(): raise SystemExit(f'{phase} expected absent: {relative}')
    elif not path.is_file() or digest(path)!=expected:
        actual=digest(path) if path.is_file() else 'absent'
        raise SystemExit(f'{phase} hash mismatch for {relative}: expected {expected}, got {actual}')
PY
}

base_commit=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_commit"])' "$manifest")
actual_commit=$(git -C "$source_dir" rev-parse HEAD)
if [[ "$actual_commit" != "$base_commit" ]]; then
  printf 'base commit mismatch: expected %s, got %s\n' "$base_commit" "$actual_commit" >&2
  exit 1
fi
if [[ -n "$(git -C "$source_dir" status --porcelain --untracked-files=no)" ]]; then
  printf 'base checkout has tracked changes; refusing to materialize\n' >&2
  exit 1
fi
verify_phase "$source_dir" base

prepared=$temporary/tree
"$root/scripts/prepare-mcdma-safe-benchmark.sh" "$source_dir" "$prepared"
verify_phase "$prepared" after_cleanup
(
  cd "$prepared"
  git apply --check "$verification_patch"
  git apply "$verification_patch"
)
verify_phase "$prepared" patched
python3 - "$prepared" "$manifest" <<'PY'
import json
from pathlib import Path
import sys
tree=Path(sys.argv[1]); manifest=json.loads(Path(sys.argv[2]).read_text())
(tree/'.mcdma-exact-verification.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
PY

mv "$prepared" "$destination"
find "$temporary" -depth -delete
temporary=
trap - EXIT INT TERM HUP
printf 'prepared base=%s cleanup_patch=%s verification_patch=%s destination=%s\n' \
  "$base_commit" \
  "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["patches"][0]["sha256"])' "$manifest")" \
  "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["patches"][1]["sha256"])' "$manifest")" \
  "$destination"
