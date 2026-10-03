#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config

head_stage=${HEAD_STAGE:-$THREE_MACHINE_PORTABLE_DOWNLOAD_STAGE}
worker_stage=${WORKER_STAGE:-$head_stage}
head_state=${HEAD_DOWNLOAD_STATE:-$THREE_MACHINE_HEAD_DOWNLOAD_STATE}
worker_state=${WORKER_DOWNLOAD_STATE:-$THREE_MACHINE_WORKER_DOWNLOAD_STATE}
direct_worker=${DIRECT_WORKER_SSH:-$WORKER_SSH}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
head_assembled="${THREE_MACHINE_PORTABLE_MODEL_DIR}.assembled-$stamp"
worker_assembled="${THREE_MACHINE_PORTABLE_MODEL_DIR}.assembled-$stamp"
tmp_manifest=/tmp/glm53-portable-model-content-manifest.py

check_download() {
  local role=$1 state=$2 start=$3 end=$4 command
  printf -v command 'set -eu
state=%q
stage=%q
test -r "$state/exit"
test "$(cat "$state/exit")" = 0
if test -r "$state/active.pid" && kill -0 "$(cat "$state/active.pid")" 2>/dev/null; then
  echo "download wrapper is still running" >&2
  exit 1
fi
for i in $(seq %d %d); do
  file=$(printf "model-%%05d-of-00043.safetensors" "$i")
  test -s "$stage/$file" || { echo "missing $file" >&2; exit 1; }
done' "$state" "$([[ "$role" == head ]] && printf '%s' "$head_stage" || printf '%s' "$worker_stage")" "$start" "$end"
  case "$role" in
    head) head_ssh "$command" ;;
    worker) worker_ssh "$command" ;;
    *) die "unknown role $role" ;;
  esac
}

check_download head "$head_state" 1 22
check_download worker "$worker_state" 23 43

note "Exchanging the verified model halves over the private Spark-to-Spark link"
head_ssh "set -eu
list=\$(mktemp /tmp/glm53-head-half.XXXXXX)
trap 'rm -f \"\$list\"' EXIT
for i in \$(seq 1 22); do printf 'model-%05d-of-00043.safetensors\\n' \"\$i\"; done >\"\$list\"
rsync -a --checksum --files-from=\"\$list\" '$head_stage/' '$direct_worker:$worker_stage/'"
head_ssh "set -eu
list=\$(mktemp /tmp/glm53-worker-half.XXXXXX)
trap 'rm -f \"\$list\"' EXIT
for i in \$(seq 23 43); do printf 'model-%05d-of-00043.safetensors\\n' \"\$i\"; done >\"\$list\"
rsync -a --checksum --files-from=\"\$list\" '$direct_worker:$worker_stage/' '$head_stage/'"
head_ssh "rsync -a --checksum --exclude='.cache/' --exclude='model-*.safetensors' '$head_stage/' '$direct_worker:$worker_stage/'"

note "Building clean model directories without downloader metadata"
head_ssh "set -eu; test ! -e '$head_assembled'; mkdir -p '$head_assembled'; rsync -a --exclude='.cache/' '$head_stage/' '$head_assembled/'"
worker_ssh "set -eu; test ! -e '$worker_assembled'; mkdir -p '$worker_assembled'; rsync -a --exclude='.cache/' '$worker_stage/' '$worker_assembled/'"

scp -o BatchMode=yes "$PROJECT_ROOT/scripts/model_content_manifest.py" "$HEAD_SSH:$tmp_manifest"
scp -o BatchMode=yes "$PROJECT_ROOT/scripts/model_content_manifest.py" "$WORKER_MANAGEMENT_SSH:$tmp_manifest"
head_result=$(mktemp)
worker_result=$(mktemp)
cleanup() {
  rm -f "$head_result" "$worker_result"
  head_ssh "rm -f '$tmp_manifest'" >/dev/null 2>&1 || true
  worker_ssh "rm -f '$tmp_manifest'" >/dev/null 2>&1 || true
}
trap cleanup EXIT
head_ssh "python3 '$tmp_manifest' '$head_assembled'" >"$head_result" &
head_hash_pid=$!
worker_ssh "python3 '$tmp_manifest' '$worker_assembled'" >"$worker_result" &
worker_hash_pid=$!
wait "$head_hash_pid"
wait "$worker_hash_pid"

python3 - "$THREE_MACHINE_PORTABLE_MODEL_CONTENT_SHA256" \
  "$THREE_MACHINE_PORTABLE_MODEL_MANIFEST_SHA256" "$head_result" "$worker_result" <<'PY'
import json
import sys
from pathlib import Path

expected_content, expected_sizes = sys.argv[1:3]
facts = [json.loads(Path(path).read_text()) for path in sys.argv[3:]]
if facts[0] != facts[1]:
    raise SystemExit(f"Spark model content differs: {facts}")
fact = facts[0]
if fact.get("sha256") != expected_content:
    raise SystemExit(f"Spark model content differs from the verified Mac: {fact}")
if fact.get("size_sha256") != expected_sizes:
    raise SystemExit(f"Spark filename/size inventory differs from the pin: {fact}")
if fact.get("files") != 54 or fact.get("bytes") != 181741759037:
    raise SystemExit(f"Spark model inventory is incomplete: {fact}")
PY

promote() {
  local role=$1 assembled=$2 command
  printf -v command 'set -eu
final=%q
assembled=%q
backup="${final}.pre-verified-%s"
test -d "$assembled"
if test -e "$final"; then
  test ! -e "$backup"
  mv "$final" "$backup"
fi
mkdir -p "$(dirname "$final")"
mv "$assembled" "$final"' "$THREE_MACHINE_PORTABLE_MODEL_DIR" "$assembled" "$stamp"
  case "$role" in
    head) head_ssh "$command" ;;
    worker) worker_ssh "$command" ;;
  esac
}

promote head "$head_assembled"
promote worker "$worker_assembled"
note "PASS: both Sparks now have the exact model content already verified on the Mac"
