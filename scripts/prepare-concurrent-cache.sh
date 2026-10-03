#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config

prompt=${1:-'Output the integers from 1 through 2000 in order, separated only by commas. Do not explain and do not stop early.'}
tokens=${2:-256}
[[ "$tokens" =~ ^[1-9][0-9]*$ ]] || die "token count must be positive"
proof_id="concurrent-cache-$(date -u +%Y%m%dT%H%M%SZ)-$$"
evidence="$PROJECT_ROOT/results/raw/$proof_id"
umask 077
mkdir -p "$evidence"
started=false
cleanup() {
  local rc=$?
  if [[ "$started" == true ]]; then
    "$PROJECT_ROOT/scripts/stop-cache-handoff.sh" \
      >"$evidence/stop.stdout" 2>"$evidence/stop.stderr" || rc=$?
  fi
  release_three_machine_lifecycle_lock
  exit "$rc"
}
trap cleanup EXIT INT TERM HUP
acquire_three_machine_lifecycle_lock

"$PROJECT_ROOT/scripts/start-cache-handoff.sh" \
  >"$evidence/start.stdout" 2>"$evidence/start.stderr"
started=true
before=$(adapter peek)
printf '%s\n' "$before" >"$evidence/mcdma-before.json"

preparation_started_ns=$(python3 -c 'import time; print(time.time_ns())')
command=(env
  "MCDMA_RPC_LIBRARY=$THREE_MACHINE_MCDMA_LIBRARY"
  "PYTHONPATH=$THREE_MACHINE_STUDIO_DIR:$THREE_MACHINE_TENSORFOLD_SOURCE"
  "$THREE_MACHINE_STUDIO_PYTHON" -m experiments.three_machine.mac_decode
  --model "$THREE_MACHINE_PORTABLE_MODEL_STUDIO_DIR"
  --prompt "$prompt" --max-new-tokens "$tokens" --parallel 1 --compare-local
  --mailbox "$THREE_MACHINE_MAILBOX" --timeout 600
  --chunk-mib "$THREE_MACHINE_HANDOFF_CHUNK_MIB")
printf -v remote 'cd %q && ' "$THREE_MACHINE_STUDIO_DIR"
printf -v command_text '%q ' "${command[@]}"
studio_ssh "$remote$command_text" >"$evidence/result.json" 2>"$evidence/mac.stderr"
preparation_finished_ns=$(python3 -c 'import time; print(time.time_ns())')
head_ssh "docker logs '$THREE_MACHINE_HANDOFF_CONTAINER'" \
  >"$evidence/prefill-head.log" 2>&1
worker_ssh "docker logs '$THREE_MACHINE_HANDOFF_CONTAINER'" \
  >"$evidence/prefill-worker.log" 2>&1

after=$(adapter peek)
printf '%s\n' "$after" >"$evidence/mcdma-after.json"
mkdir -p "$PROJECT_ROOT/.state"
digest=$(python3 - "$evidence/result.json" "$evidence/mcdma-before.json" \
  "$evidence/mcdma-after.json" "$PROJECT_ROOT/.state" "$tokens" \
  "$preparation_started_ns" "$preparation_finished_ns" \
  "$evidence/prefill-head.log" "$evidence/prefill-worker.log" <<'PY'
import hashlib, json, os, re, sys, tempfile
from pathlib import Path
result, before, after = (json.loads(Path(path).read_text()) for path in sys.argv[1:4])
state_dir, tokens = Path(sys.argv[4]), int(sys.argv[5])
preparation_seconds = (int(sys.argv[7]) - int(sys.argv[6])) / 1_000_000_000

def prefill_seconds(path, rank):
    records = []
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("event") == "glm_cache_prefill" and value.get("rank") == rank:
            records.append(value)
    if len(records) != 1:
        raise SystemExit(f"Spark rank {rank} did not emit exactly one prefill record")
    value = records[0]
    seconds = value.get("seconds")
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds <= 0:
        raise SystemExit(f"Spark rank {rank} prefill time is invalid")
    return float(seconds)

head_prefill_seconds = prefill_seconds(sys.argv[8], 0)
worker_prefill_seconds = prefill_seconds(sys.argv[9], 1)
digest = result.get("prompt_sha256", "")
if result.get("event") != "glm_mcdma_decode" or not re.fullmatch(r"[0-9a-f]{64}", digest):
    raise SystemExit("cache preparation returned no verified prompt digest")
if result.get("cached_tokens") != result.get("prompt_tokens", 0) - 1:
    raise SystemExit("cache preparation stopped at the wrong prompt position")
reference = result.get("local_reference") or {}
completion = reference.get("tokens")
if not isinstance(completion, list) or len(completion) != tokens:
    raise SystemExit("frozen local reference did not complete the requested token count")
if result.get("exact_token_match") is not True or reference.get("finish_reason") != "length":
    raise SystemExit("Spark cache and local reference are not an exact fixed-length match")
frame_lengths = result.get("cache_frame_lengths")
cache_bytes = result.get("cache_bytes_total")
if (
    result.get("cache_frame_count") != 3
    or not isinstance(frame_lengths, list)
    or len(frame_lengths) != 3
    or any(type(value) is not int or value <= 0 or value > 64 * 1024 * 1024 for value in frame_lengths)
    or type(cache_bytes) is not int
    or sum(frame_lengths) != cache_bytes
):
    raise SystemExit("cache preparation did not produce the expected three authenticated frames")
for side in ("mac", "spark"):
    if after.get(side, {}).get("calls", 0) - before.get(side, {}).get("calls", 0) != 5:
        raise SystemExit(f"MCDMA {side} counter did not advance by exactly five calls")
    if after.get(side, {}).get("failures") != 0:
        raise SystemExit(f"MCDMA {side} reports a failure")
text = str(reference.get("text", ""))
transfer_id = result.get("transfer_id")
if not isinstance(transfer_id, str) or not transfer_id:
    raise SystemExit("cache preparation returned no transfer identity")
transfer_seconds = result.get("transfer_seconds")
import_seconds = result.get("import_seconds")
for label, value in (("transfer", transfer_seconds), ("import", import_seconds)):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise SystemExit(f"cache {label} timing is invalid")
record = {
    "schema": 2,
    "event": "glm_concurrent_reference",
    "prompt_sha256": digest,
    "token_sha256": str(reference.get("token_sha256", "")),
    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    "max_new_tokens": tokens,
    "prompt_tokens": result.get("prompt_tokens"),
    "cache_bytes_total": cache_bytes,
    "cache_frame_count": 3,
    "cache_frame_lengths": frame_lengths,
    "transfer_id": transfer_id,
    "preparation_timings": {
        "reference_wall_seconds": preparation_seconds,
        "cuda_prefill_head_seconds": head_prefill_seconds,
        "cuda_prefill_worker_seconds": worker_prefill_seconds,
        "mcdma_transfer_seconds": float(transfer_seconds),
        "mac_import_seconds": float(import_seconds),
    },
}
if not re.fullmatch(r"[0-9a-f]{64}", record["token_sha256"]):
    raise SystemExit("frozen local reference has no valid token hash")
target = state_dir / f"concurrent-reference-{digest}.json"
fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=state_dir)
try:
    with os.fdopen(fd, "w") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
print(digest)
PY
)
head_ssh "test -d '$THREE_MACHINE_CACHE_BUNDLE_DIR/$digest' && test ! -L '$THREE_MACHINE_CACHE_BUNDLE_DIR/$digest'" \
  || die "the TensorFold prompt cache was not persisted"

"$PROJECT_ROOT/scripts/stop-cache-handoff.sh" \
  >"$evidence/stop.stdout" 2>"$evidence/stop.stderr"
started=false
trap - EXIT INT TERM HUP
release_three_machine_lifecycle_lock
printf '%s\n' "$digest" >"$evidence/prompt-sha256.txt"
cp "$PROJECT_ROOT/.state/concurrent-reference-$digest.json" "$evidence/frozen-reference.json"
note "PASS: reusable two-Spark prompt cache and frozen ${tokens}-token reference $digest are prepared; evidence is at $evidence"
printf '%s\n' "$digest"
