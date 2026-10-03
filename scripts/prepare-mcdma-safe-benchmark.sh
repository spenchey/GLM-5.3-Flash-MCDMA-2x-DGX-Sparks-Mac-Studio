#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
patch=$root/patches/mcdma/0001-safe-benchmark-cleanup.patch
manifest=$root/patches/mcdma/safe-benchmark-cleanup-manifest.json

if [[ $# -ne 2 ]]; then
  printf 'usage: %s EXACT_BASE_CHECKOUT NEW_ISOLATED_DIRECTORY\n' "$0" >&2
  exit 2
fi

source_dir=$(cd "$1" && pwd)
destination=$2
parent=$(cd "$(dirname "$destination")" && pwd)
destination=$parent/$(basename "$destination")

if [[ -e "$destination" ]]; then
  printf 'refusing existing destination: %s\n' "$destination" >&2
  exit 2
fi
if [[ "$source_dir" == "$destination" ]]; then
  printf 'source and destination must differ\n' >&2
  exit 2
fi

tmp=$(mktemp -d "$parent/.mcdma-safe-benchmark.XXXXXX")
cleanup() {
  if [[ -n "${tmp:-}" && -d "$tmp" ]]; then
    rm -rf -- "$tmp"
  fi
}
trap cleanup EXIT INT TERM HUP

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

verify_hashes() {
  local tree=$1 phase=$2
  python3 - "$tree" "$patch" "$manifest" "$phase" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

tree = Path(sys.argv[1])
patch = Path(sys.argv[2])
manifest = json.loads(Path(sys.argv[3]).read_text())
phase = sys.argv[4]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if digest(patch) != manifest['patch']['sha256']:
    raise SystemExit('patch hash mismatch')
for relative, hashes in manifest['files'].items():
    actual = digest(tree / relative)
    expected = hashes[f'{phase}_sha256']
    if actual != expected:
        raise SystemExit(f'{phase} hash mismatch for {relative}: expected {expected}, got {actual}')
PY
}

verify_hashes "$source_dir" base
git -C "$source_dir" archive "$base_commit" | tar -x -C "$tmp"
git -C "$tmp" init -q
(
  cd "$tmp"
  git apply --check "$patch"
  git apply "$patch"
)
verify_hashes "$tmp" patched

python3 - "$tmp" "$manifest" <<'PY'
import json
from pathlib import Path
import sys

destination = Path(sys.argv[1]) / '.mcdma-safe-benchmark.json'
manifest = json.loads(Path(sys.argv[2]).read_text())
destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
PY

mv "$tmp" "$destination"
tmp=
trap - EXIT INT TERM HUP
printf 'prepared base=%s patch_sha256=%s destination=%s\n' \
  "$base_commit" \
  "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["patch"]["sha256"])' "$manifest")" \
  "$destination"
