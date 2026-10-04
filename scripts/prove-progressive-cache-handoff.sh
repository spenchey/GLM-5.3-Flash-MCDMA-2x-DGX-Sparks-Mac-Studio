#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config

(( $# <= 1 )) || die "usage: $0 [frame-mib]"
frame_mib=${1:-32}
[[ "$frame_mib" =~ ^[0-9]+$ ]] \
  && (( frame_mib >= 1 && frame_mib <= THREE_MACHINE_HANDOFF_CHUNK_MIB )) \
  || die "frame-mib must be between 1 and $THREE_MACHINE_HANDOFF_CHUNK_MIB"

trap release_three_machine_lifecycle_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
acquire_three_machine_lifecycle_lock

proof_id="progressive-cache-canary-${frame_mib}m-$(date -u +%Y%m%dT%H%M%SZ)-$$"
evidence="$PROJECT_ROOT/results/raw/$proof_id"
umask 077
mkdir -p "$evidence"
proof_complete=false
cleanup_failed_proof() {
  local rc=$?
  if (( rc != 0 )) && [[ "$proof_complete" != true ]]; then
    note "Progressive canary failed; evidence is preserved at $evidence" >&2
    "$PROJECT_ROOT/scripts/stop-cache-handoff.sh" \
      >"$evidence/cleanup.stdout" 2>"$evidence/cleanup.stderr" || true
  fi
  release_three_machine_lifecycle_lock
  return "$rc"
}
trap cleanup_failed_proof EXIT

head_before=$(cache_handoff_container_state head)
worker_before=$(cache_handoff_container_state worker)
validate_cache_handoff_container_state "head Spark" "$head_before"
validate_cache_handoff_container_state "worker Spark" "$worker_before"
cache_handoff_container_running "$head_before" || die "head Spark prompt rank is not running"
cache_handoff_container_running "$worker_before" || die "worker Spark prompt rank is not running"
head_id=$(cache_handoff_container_id "$head_before")
worker_id=$(cache_handoff_container_id "$worker_before")
link_before=$(adapter peek)
mcdma_link_healthy "$link_before" || die "MCDMA is not healthy before the canary"
mcdma_link_busy "$link_before" && die "MCDMA is busy before the canary"
printf '%s\n' "$head_before" >"$evidence/head-before.txt"
printf '%s\n' "$worker_before" >"$evidence/worker-before.txt"
printf '%s\n' "$link_before" >"$evidence/mcdma-before.json"

prompt='Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough.'
command=(env
  "MCDMA_RPC_LIBRARY=$THREE_MACHINE_MCDMA_LIBRARY"
  "PYTHONPATH=$THREE_MACHINE_STUDIO_DIR:$THREE_MACHINE_TENSORFOLD_SOURCE"
  "$THREE_MACHINE_STUDIO_PYTHON" -m experiments.three_machine.mac_decode
  --model "$THREE_MACHINE_PORTABLE_MODEL_STUDIO_DIR"
  --prompt "$prompt" --max-new-tokens 16 --force-length
  --mailbox "$THREE_MACHINE_MAILBOX" --timeout 600
  --compare-local --no-copy-drafts --parallel 1 --mtp-drafts 1
  --chunk-mib "$frame_mib"
  --progressive-handoff)
printf -v remote 'cd %q && ' "$THREE_MACHINE_STUDIO_DIR"
printf -v command_text '%q ' "${command[@]}"
note "Running a strict 16-token protocol-v4 canary before the full benchmark"
studio_ssh "$remote$command_text" \
  >"$evidence/result.json" 2>"$evidence/mac.stderr"

head_after=$(cache_handoff_container_state head)
worker_after=$(cache_handoff_container_state worker)
validate_cache_handoff_container_state "head Spark" "$head_after"
validate_cache_handoff_container_state "worker Spark" "$worker_after"
cache_handoff_container_running "$head_after" || die "head Spark stopped during the canary"
cache_handoff_container_running "$worker_after" || die "worker Spark stopped during the canary"
[[ "$(cache_handoff_container_id "$head_after")" == "$head_id" ]] \
  || die "head Spark identity changed during the canary"
[[ "$(cache_handoff_container_id "$worker_after")" == "$worker_id" ]] \
  || die "worker Spark identity changed during the canary"
[[ "$head_after" == *'|true|false|0|cache-handoff' ]] \
  || die "head Spark reported an unhealthy or OOM state"
[[ "$worker_after" == *'|true|false|0|cache-handoff' ]] \
  || die "worker Spark reported an unhealthy or OOM state"
link_after=$(adapter peek)
mcdma_link_healthy "$link_after" || die "MCDMA is not healthy after the canary"
mcdma_link_busy "$link_after" && die "MCDMA remained busy after the canary"
printf '%s\n' "$head_after" >"$evidence/head-after.txt"
printf '%s\n' "$worker_after" >"$evidence/worker-after.txt"
printf '%s\n' "$link_after" >"$evidence/mcdma-after.json"
head_ssh "docker logs '$head_id' 2>&1" >"$evidence/head.log"
worker_ssh "docker logs '$worker_id' 2>&1" >"$evidence/worker.log"
head_ssh "docker inspect '$head_id'" >"$evidence/head-inspect.json"
worker_ssh "docker inspect '$worker_id'" >"$evidence/worker-inspect.json"

python3 - "$evidence/result.json" "$evidence/mcdma-before.json" \
  "$evidence/mcdma-after.json" "$evidence/head.log" "$evidence/worker.log" \
  "$frame_mib" <<'PY' \
  | tee "$evidence/summary.json"
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
before = json.loads(Path(sys.argv[2]).read_text())
after = json.loads(Path(sys.argv[3]).read_text())
frame_mib = int(sys.argv[6])


def last_event(path, event, rank):
    matches = []
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == event and record.get("rank") == rank:
            matches.append(record)
    if not matches:
        raise SystemExit(f"{path} has no {event} record for rank {rank}")
    return matches[-1]


if result.get("event") != "glm_mcdma_decode":
    raise SystemExit("unexpected canary record")
if result.get("cache_protocol") != 4:
    raise SystemExit("the canary did not use cache protocol 4")
if result.get("exact_token_match") is not True:
    raise SystemExit("Spark-prefilled and Mac-prefilled output tokens differ")
if result.get("parallel") != 1 or result.get("completion_tokens") != 16:
    raise SystemExit("the canary did not preserve its frozen single-stream shape")
if result.get("cached_tokens") != result.get("prompt_tokens", 0):
    raise SystemExit("the Sparks did not prepare the complete prompt prefix")
if result.get("mtp_cached_tokens") != result.get("prompt_tokens", 0) - 1:
    raise SystemExit("the Spark MTP prefix stopped at the wrong position")
if result.get("result", {}).get("first_token_source") != "mac_head_from_spark_final_hidden":
    raise SystemExit("the Mac did not generate the first reply token from Spark state")
if result.get("cache_wire_transfers") != 1 or result.get("cache_reuse_copies") != 0:
    raise SystemExit("the canary did not perform exactly one live cache transfer")
cache_bytes = int(result.get("cache_bytes_per_request", 0))
frame_lengths = result.get("cache_frame_lengths") or []
if cache_bytes <= 0 or sum(frame_lengths) != cache_bytes:
    raise SystemExit("the progressive frames do not cover the complete cache")

telemetry = result.get("handoff_telemetry") or {}
required = (
    "prepare_call_seconds", "plan_decode_seconds", "frame_call_seconds",
    "frame_assemble_seconds", "seal_call_seconds", "verify_seconds",
    "release_call_seconds", "total_seconds", "job_id", "plan_id",
    "transfer_id",
)
if any(name not in telemetry for name in required):
    raise SystemExit("the progressive transfer is missing phase or identity evidence")
if telemetry.get("protocol") != 4:
    raise SystemExit("the progressive telemetry did not attest protocol 4")
requested_frame_bytes = frame_mib * 1024 * 1024
effective_frame_bytes = int(telemetry.get("effective_frame_bytes", 0))
if not 0 < effective_frame_bytes <= requested_frame_bytes:
    raise SystemExit("the progressive transfer used an invalid effective frame limit")
expected_frame_count = (cache_bytes + effective_frame_bytes - 1) // effective_frame_bytes
if len(frame_lengths) != expected_frame_count:
    raise SystemExit("the progressive frame count does not match the effective mailbox limit")
if telemetry.get("frame_strategy") != "execution_order_tensor_groups":
    raise SystemExit("the progressive transfer used the wrong frame order")
if telemetry.get("archive_bytes") != cache_bytes:
    raise SystemExit("the progressive archive byte count differs from the imported cache")
if telemetry.get("frame_payload_bytes") != cache_bytes:
    raise SystemExit("the progressive frame byte count differs from the imported cache")
if telemetry.get("transfer_id") != result.get("transfer_id"):
    raise SystemExit("the Mac imported a different transfer identity")

head_event = last_event(sys.argv[4], "glm_progressive_prefill", 0)
worker_event = last_event(sys.argv[5], "glm_progressive_prefill", 1)
for event in (head_event, worker_event):
    if event.get("cached_tokens") != result.get("cached_tokens"):
        raise SystemExit("a Spark rank prepared the wrong prefix length")
    if event.get("cache_bytes") != cache_bytes:
        raise SystemExit("a Spark rank produced the wrong cache byte count")
if head_event.get("job_id") != telemetry.get("job_id"):
    raise SystemExit("the head Spark job identity differs from the Mac request")
if head_event.get("plan_id") != telemetry.get("plan_id"):
    raise SystemExit("the head Spark plan identity differs from the Mac request")
if head_event.get("transfer_id") != telemetry.get("transfer_id"):
    raise SystemExit("the head Spark transfer identity differs from the Mac import")

for side in ("mac", "spark"):
    if after.get(side, {}).get("calls", 0) <= before.get(side, {}).get("calls", 0):
        raise SystemExit(f"MCDMA {side} call counter did not advance")
    if after.get(side, {}).get("failures") != 0:
        raise SystemExit(f"MCDMA {side} reports a failure")

print(json.dumps({
    "cache_bytes": cache_bytes,
    "cache_protocol": 4,
    "exact_token_match": True,
    "effective_frame_bytes": effective_frame_bytes,
    "frame_count": len(frame_lengths),
    "frame_mib": frame_mib,
    "job_id": telemetry["job_id"],
    "output_token_sha256": result["result"]["token_sha256"],
    "plan_id": telemetry["plan_id"],
    "prompt_tokens": result["prompt_tokens"],
    "transfer_id": telemetry["transfer_id"],
    "transfer_seconds": result["transfer_seconds"],
}, sort_keys=True))
PY

proof_complete=true
note "PASS: protocol 4 transferred the complete Spark cache, matched the Mac tokens, and left both ranks healthy; evidence is at $evidence"
