#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
patch=$root/patches/tensorfold/0001-metal-read-exl3-dense-stage.patch
manifest=$root/patches/tensorfold/metal-stage-manifest.json

if [[ $# -ne 2 ]]; then
  printf 'usage: %s EXACT_TENSORFOLD_SOURCE DESTINATION\n' "$0" >&2
  exit 2
fi
source_dir=$(cd "$1" && pwd)
destination=$(python3 - "$2" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().absolute())
PY
)
[[ "$source_dir" != "$destination" ]] || { echo "source and destination must differ" >&2; exit 1; }
[[ ! -e "$destination" ]] || { echo "destination already exists: $destination" >&2; exit 1; }

verify_tree() {
  local tree=$1 key=$2 expected_tree actual_tree
  python3 - "$tree" "$manifest" "$patch" "$key" <<'PY'
import hashlib, json, sys
from pathlib import Path

tree = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
patch_path = Path(sys.argv[3])
key = sys.argv[4]
expected = json.loads(manifest_path.read_text())

patch_hash = hashlib.sha256(patch_path.read_bytes()).hexdigest()
if patch_hash != expected["patch_sha256"]:
    raise SystemExit(f"stage patch hash mismatch: {patch_hash}")
for relative, hashes in expected["files"].items():
    path = tree / relative
    if not path.is_file():
        raise SystemExit(f"missing required TensorFold source file: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != hashes[key]:
        raise SystemExit(f"{relative} {key} mismatch: {actual}")
PY
  expected_tree=$(python3 - "$manifest" "$key" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))[f"{sys.argv[2].removesuffix('_sha256')}_tree_sha256"])
PY
)
  actual_tree=$(python3 "$root/scripts/tree-manifest.py" \
    --exclude-relative .tensorfold-metal-stage.json "$tree")
  [[ "$actual_tree" == "$expected_tree" ]] \
    || { printf '%s full-tree mismatch: %s\n' "$key" "$actual_tree" >&2; exit 1; }
}

verify_tree "$source_dir" base_sha256
parent=$(dirname "$destination")
mkdir -p "$parent"
temporary=$(mktemp -d "$parent/.tensorfold-metal-stage.XXXXXXXX")
cleanup() { [[ ! -d "$temporary" ]] || find "$temporary" -depth -delete; }
trap cleanup EXIT INT TERM HUP
tree=$temporary/tree
mkdir "$tree"
tar -cf - \
  --exclude=.git --exclude=.mypy_cache --exclude=.pytest_cache \
  --exclude=__pycache__ --exclude='*.pyc' --exclude=.DS_Store \
  --exclude=.tensorfold-metal-stage.json \
  -C "$source_dir" . | tar -xf - -C "$tree"
git -C "$tree" init -q
git -C "$tree" apply --check "$patch"
git -C "$tree" apply "$patch"
find "$tree/.git" -depth -delete
verify_tree "$tree" patched_sha256
cp "$manifest" "$tree/.tensorfold-metal-stage.json"
mv "$tree" "$destination"
trap - EXIT INT TERM HUP
find "$temporary" -depth -delete
printf 'PASS: exact TensorFold Metal stage source materialized at %s\n' "$destination"
