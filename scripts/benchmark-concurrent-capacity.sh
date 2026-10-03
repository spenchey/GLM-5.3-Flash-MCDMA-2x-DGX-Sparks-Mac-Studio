#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

digest=${1:?usage: benchmark-concurrent-capacity.sh PROMPT_SHA256 [TOTAL_PARALLEL] [MAC_PARALLEL] [TOKENS]}
reference="$PROJECT_ROOT/.state/concurrent-reference-$digest.json"
[[ -r "$reference" ]] || die "the frozen prompt/output reference is missing; run prepare-concurrent-cache.sh first"
read -r expected_hash expected_text_hash reference_tokens expected_frame_count < <(
  python3 - "$reference" <<'PY'
import json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text())
if (
    value.get("schema") != 2
    or value.get("event") != "glm_concurrent_reference"
    or value.get("prompt_sha256") != Path(sys.argv[1]).stem.removeprefix("concurrent-reference-")
):
    raise SystemExit("frozen reference identity differs from its filename")
print(value.get("token_sha256", ""), value.get("text_sha256", ""), value.get("max_new_tokens", ""), value.get("cache_frame_count", ""))
PY
)
total_parallel=${2:-16}
mac_parallel=${3:-8}
tokens=${4:-$reference_tokens}
expected_mcdma_calls=$((expected_frame_count + 2))
[[ "$digest" =~ ^[0-9a-f]{64}$ ]] || die "prompt digest must be a lowercase SHA-256"
[[ "$expected_hash" =~ ^[0-9a-f]{64}$ ]] || die "output digest must be a lowercase SHA-256"
[[ "$expected_text_hash" =~ ^[0-9a-f]{64}$ ]] || die "output byte digest must be a lowercase SHA-256"
[[ "$total_parallel" =~ ^(1|2|4|8|16)$ ]] || die "total width must be 1, 2, 4, 8, or 16"
[[ "$mac_parallel" =~ ^(1|2|4|8)$ ]] || die "Mac width must be 1, 2, 4, or 8"
(( mac_parallel <= total_parallel )) || die "Mac width cannot exceed total width"
(( total_parallel - mac_parallel <= 8 )) || die "candidate Spark width cannot exceed eight"
[[ "$tokens" =~ ^[1-9][0-9]*$ ]] || die "token count must be positive"
[[ "$tokens" == "$reference_tokens" ]] || die "requested token count differs from the frozen reference"
[[ "$expected_frame_count" == 3 ]] || die "frozen cache does not contain the expected three frames"
[[ "$expected_mcdma_calls" == 5 ]] || die "three cache frames must make exactly five MCDMA calls"

prompt='Output the integers from 1 through 2000 in order, separated only by commas. Do not explain and do not stop early.'
stamp=$(date -u +%Y%m%dT%H%M%SZ | tr '[:upper:]' '[:lower:]')
run_id="paired-capacity-p${total_parallel}-s$((total_parallel-mac_parallel))-m${mac_parallel}-t${tokens}-${stamp}-$$"
evidence="$PROJECT_ROOT/results/raw/$run_id"
umask 077
mkdir -p "$evidence/rounds"
cp "$reference" "$evidence/frozen-reference.json"
native_started=false
replay_started=false
worker_started=false
worker_pid=''
worker_id="paired-capacity-$run_id"
worker_pid_file="$THREE_MACHINE_API_STATE_DIR/$worker_id.json"
wait_local_pid() {
  local pid timeout deadline
  pid=${1:?pid required}
  timeout=${2:-120}
  deadline=$((SECONDS + timeout))
  while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}
stop_worker_exact() {
  studio_ssh "'$THREE_MACHINE_STUDIO_PYTHON' '$THREE_MACHINE_STUDIO_DIR/scripts/stop-concurrent-worker.py' --pid-file '$worker_pid_file' --worker-id '$worker_id' --timeout 30" \
    >"$evidence/worker-stop.stdout" 2>"$evidence/worker-stop.stderr"
}
cleanup() {
  local rc=$?
  if [[ "$worker_started" == true ]]; then
    printf '%s\n' '{"command":"stop"}' >&7 2>/dev/null || true
    exec 7>&- 8<&- || true
    stop_worker_exact || rc=$?
    [[ -z "$worker_pid" ]] || wait_local_pid "$worker_pid" 60
    rm -f -- "$evidence/worker-command.fifo" "$evidence/worker-response.fifo"
  fi
  if [[ "$replay_started" == true ]]; then
    "$PROJECT_ROOT/scripts/stop-cache-replay.sh" \
      >"$evidence/replay-stop.stdout" 2>"$evidence/replay-stop.stderr" || rc=$?
  fi
  if [[ "$native_started" == true ]]; then
    "$PROJECT_ROOT/scripts/stop-native-portable.sh" \
      >"$evidence/native-stop.stdout" 2>"$evidence/native-stop.stderr" || rc=$?
  fi
  release_three_machine_lifecycle_lock
  exit "$rc"
}
trap cleanup EXIT INT TERM HUP
acquire_three_machine_lifecycle_lock

"$PROJECT_ROOT/scripts/start-native-portable.sh" \
  >"$evidence/native-start.stdout" 2>"$evidence/native-start.stderr"
native_started=true
"$PROJECT_ROOT/scripts/start-cache-replay.sh" "$digest" \
  >"$evidence/replay-start.stdout" 2>"$evidence/replay-start.stderr"
replay_started=true

head_ssh "docker inspect '$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER'" >"$evidence/head-before.json"
worker_ssh "docker inspect '$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER'" >"$evidence/worker-before.json"
head_ssh "docker inspect '$THREE_MACHINE_CACHE_REPLAY_CONTAINER'" >"$evidence/replay-before.json"
native_url="http://$THREE_MACHINE_STUDIO_TO_HEAD_SSH:$THREE_MACHINE_NATIVE_PORTABLE_PORT"
studio_ssh "curl -fsS --max-time 10 '$native_url/health'" >"$evidence/studio-native-health-before.json"

command=(env
  "MCDMA_RPC_LIBRARY=$THREE_MACHINE_MCDMA_LIBRARY"
  "PYTHONPATH=$THREE_MACHINE_STUDIO_DIR:$THREE_MACHINE_TENSORFOLD_SOURCE"
  "TENSORFOLD_NO_UPDATE_CHECK=1"
  "$THREE_MACHINE_STUDIO_PYTHON" -m experiments.three_machine.concurrent_capacity
  --worker --model "$THREE_MACHINE_PORTABLE_MODEL_STUDIO_DIR"
  --native-url "$native_url" --prompt "$prompt"
  --mailbox "$THREE_MACHINE_MAILBOX" --max-new-tokens "$tokens"
  --timeout 900 --expected-prompt-sha256 "$digest"
  --expected-token-sha256 "$expected_hash" --expected-text-sha256 "$expected_text_hash"
  --tensorfold-source "$THREE_MACHINE_TENSORFOLD_SOURCE"
  --expected-tensorfold-tree-sha256 "$THREE_MACHINE_TENSORFOLD_SOURCE_MANIFEST_SHA256"
  --expected-python-sha256 "$THREE_MACHINE_STUDIO_PYTHON_SHA256"
  --expected-python-env "$THREE_MACHINE_STUDIO_PYTHON_ENV"
  --pid-file "$worker_pid_file" --worker-id "$worker_id")
printf -v remote 'cd %q && ' "$THREE_MACHINE_STUDIO_DIR"
printf -v command_text '%q ' "${command[@]}"
mkfifo "$evidence/worker-command.fifo" "$evidence/worker-response.fifo"
studio_ssh "$remote$command_text" \
  <"$evidence/worker-command.fifo" >"$evidence/worker-response.fifo" \
  2>"$evidence/worker.stderr" &
worker_pid=$!
# Bash 3.2 has no coproc or dynamic FD allocation. Fixed private descriptors
# keep the remote JSON-lines worker alive across the complete A/B sequence.
exec 7>"$evidence/worker-command.fifo"
exec 8<"$evidence/worker-response.fifo"
worker_started=true

IFS= read -r ready <&8 || die "paired worker stopped before readiness"
printf '%s\n' "$ready" >"$evidence/worker-ready.json"
python3 - "$evidence/worker-ready.json" "$digest" "$expected_hash" \
  "$expected_text_hash" "$tokens" "$worker_id" \
  "$THREE_MACHINE_TENSORFOLD_SOURCE_MANIFEST_SHA256" \
  "$THREE_MACHINE_STUDIO_PYTHON_SHA256" "$THREE_MACHINE_STUDIO_PYTHON_ENV" \
  "$THREE_MACHINE_TENSORFOLD_SOURCE" "$TENSORFOLD_MAC_PACKAGE_VERSION" <<'PY' \
  || die "paired worker readiness evidence is invalid"
import json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text())
if value.get("event") != "glm_paired_capacity_ready":
    raise SystemExit("wrong worker readiness event")
if (
    value.get("prompt_sha256") != sys.argv[2]
    or value.get("token_sha256") != sys.argv[3]
    or value.get("text_sha256") != sys.argv[4]
    or value.get("max_new_tokens") != int(sys.argv[5])
    or value.get("worker_id") != sys.argv[6]
):
    raise SystemExit("worker readiness differs from the frozen workload")
runtime = value.get("runtime") or {}
if (
    runtime.get("tensorfold_tree_sha256") != sys.argv[7]
    or runtime.get("python_sha256") != sys.argv[8]
    or runtime.get("python_env") != sys.argv[9]
    or runtime.get("tensorfold_version") != sys.argv[11]
):
    raise SystemExit("worker readiness differs from the frozen Mac runtime")
module_path = str(runtime.get("tensorfold_module_path", ""))
source_path = str(Path(sys.argv[10]))
try:
    Path(module_path).relative_to(Path(source_path))
except ValueError as exc:
    raise SystemExit("worker imported TensorFold outside the frozen source tree") from exc
PY

rounds_file="$evidence/rounds.jsonl"
: >"$rounds_file"
run_arm() {
  local phase arm ordinal tag
  phase=$1
  arm=$2
  ordinal=$3
  tag="$phase-$ordinal-$arm"
  head_ssh "docker inspect '$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER'" >"$evidence/rounds/$tag-head-before.json"
  worker_ssh "docker inspect '$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER'" >"$evidence/rounds/$tag-worker-before.json"
  head_ssh "docker inspect '$THREE_MACHINE_CACHE_REPLAY_CONTAINER'" >"$evidence/rounds/$tag-replay-before.json"
  studio_ssh "curl -fsS --max-time 10 '$native_url/health'" >"$evidence/rounds/$tag-health-before.json"
  adapter peek >"$evidence/rounds/$tag-before.json"
  printf '{"arm":"%s","run":%d,"total_parallel":%d,"mac_parallel":%d}\n' \
    "$arm" "$ordinal" "$total_parallel" "$mac_parallel" >&7
  IFS= read -r result <&8 || die "paired worker stopped during $tag"
  printf '%s\n' "$result" >"$evidence/rounds/$tag-result.json"
  adapter peek >"$evidence/rounds/$tag-after.json"
  head_ssh "docker inspect '$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER'" >"$evidence/rounds/$tag-head-after.json"
  worker_ssh "docker inspect '$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER'" >"$evidence/rounds/$tag-worker-after.json"
  head_ssh "docker inspect '$THREE_MACHINE_CACHE_REPLAY_CONTAINER'" >"$evidence/rounds/$tag-replay-after.json"
  studio_ssh "curl -fsS --max-time 10 '$native_url/health'" >"$evidence/rounds/$tag-health-after.json"
  python3 -m experiments.three_machine.native_attestation \
    --anchor-head "$evidence/head-before.json" --anchor-worker "$evidence/worker-before.json" \
    --before-head "$evidence/rounds/$tag-head-before.json" --after-head "$evidence/rounds/$tag-head-after.json" \
    --before-worker "$evidence/rounds/$tag-worker-before.json" --after-worker "$evidence/rounds/$tag-worker-after.json" \
    --anchor-replay "$evidence/replay-before.json" \
    --before-replay "$evidence/rounds/$tag-replay-before.json" \
    --after-replay "$evidence/rounds/$tag-replay-after.json" \
    --health-before "$evidence/rounds/$tag-health-before.json" --health-after "$evidence/rounds/$tag-health-after.json" \
    --expected-image "$THREE_MACHINE_IMAGE_ID" --expected-parallel 4 \
    --expected-model "$THREE_MACHINE_PORTABLE_MODEL_CONTAINER_DIR" \
    --expected-drafter incoai/GLM-5.3-Flash-DFlash2 \
    --expected-prompt "$digest" \
    >"$evidence/rounds/$tag-native-attestation.json"
  python3 - "$phase" "$arm" "$ordinal" \
    "$evidence/rounds/$tag-before.json" "$evidence/rounds/$tag-result.json" \
    "$evidence/rounds/$tag-after.json" \
    "$evidence/rounds/$tag-native-attestation.json" >>"$rounds_file" <<'PY'
import json, sys
from pathlib import Path
phase, arm, ordinal = sys.argv[1], sys.argv[2], int(sys.argv[3])
before, result, after, attestation = (
    json.loads(Path(path).read_text()) for path in sys.argv[4:8]
)
if result.get("arm") != arm or result.get("run") != ordinal:
    raise SystemExit("worker returned the wrong round identity")
print(json.dumps({
    "phase": phase, "arm": arm, "ordinal": ordinal,
    "benchmark": result, "mcdma_before": before, "mcdma_after": after,
    "native_attestation": attestation,
}, sort_keys=True))
PY
}

# Warm each arm once. The candidate warmup is separately bracketed and must
# already make the exact same five hardware calls as every measured B round.
run_arm warmup A -2
run_arm warmup B -1

sequence=ABBABAAB
for (( index=0; index<${#sequence}; index++ )); do
  run_arm measured "${sequence:index:1}" "$((index + 1))"
done

printf '%s\n' '{"command":"stop"}' >&7
IFS= read -r -t 120 stopped <&8 || die "paired worker stopped without bounded final health"
printf '%s\n' "$stopped" >"$evidence/worker-stopped.json"
exec 7>&- 8<&-
deadline=$((SECONDS + 120))
while kill -0 "$worker_pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
kill -0 "$worker_pid" 2>/dev/null && die "paired worker did not exit within two minutes"
wait "$worker_pid" || die "paired worker exited unsuccessfully"
studio_ssh "test ! -e '$worker_pid_file'" || die "paired worker left its owned PID record"
worker_started=false
rm -f -- "$evidence/worker-command.fifo" "$evidence/worker-response.fifo"

head_ssh "docker inspect '$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER'" >"$evidence/head-after.json"
worker_ssh "docker inspect '$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER'" >"$evidence/worker-after.json"
head_ssh "docker inspect '$THREE_MACHINE_CACHE_REPLAY_CONTAINER'" >"$evidence/replay-after.json"
python3 - "$evidence/worker-stopped.json" >"$evidence/studio-native-health-after.json" <<'PY'
import json, sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text())
if value.get("event") != "glm_paired_capacity_stopped":
    raise SystemExit("worker stopped event is invalid")
print(json.dumps(value.get("health_after") or {}, sort_keys=True))
PY
python3 -m experiments.three_machine.native_attestation \
  --anchor-head "$evidence/head-before.json" --anchor-worker "$evidence/worker-before.json" \
  --before-head "$evidence/head-before.json" --after-head "$evidence/head-after.json" \
  --before-worker "$evidence/worker-before.json" --after-worker "$evidence/worker-after.json" \
  --anchor-replay "$evidence/replay-before.json" \
  --before-replay "$evidence/replay-before.json" --after-replay "$evidence/replay-after.json" \
  --health-before "$evidence/studio-native-health-before.json" \
  --health-after "$evidence/studio-native-health-after.json" \
  --expected-image "$THREE_MACHINE_IMAGE_ID" --expected-parallel 4 \
  --expected-model "$THREE_MACHINE_PORTABLE_MODEL_CONTAINER_DIR" \
  --expected-drafter incoai/GLM-5.3-Flash-DFlash2 \
  --expected-prompt "$digest" \
  >"$evidence/native-final-attestation.json" \
  || die "native TensorFold health, start identity, restart count, or configuration changed"
python3 -m experiments.three_machine.paired_capacity "$rounds_file" \
  --reference "$reference" \
  --parallel "$total_parallel" --mac-parallel "$mac_parallel" \
  --expected-mcdma-calls "$expected_mcdma_calls" \
  --expected-frame-count "$expected_frame_count" \
  --native-cuda-lanes 4 \
  >"$evidence/acceptance.json"

"$PROJECT_ROOT/scripts/stop-cache-replay.sh" \
  >"$evidence/replay-stop.stdout" 2>"$evidence/replay-stop.stderr"
replay_started=false
"$PROJECT_ROOT/scripts/stop-native-portable.sh" \
  >"$evidence/native-stop.stdout" 2>"$evidence/native-stop.stderr"
native_started=false
release_three_machine_lifecycle_lock
trap - EXIT INT TERM HUP
note "PASS: paired same-width capacity gate passed; evidence is at $evidence"
cat "$evidence/acceptance.json"
