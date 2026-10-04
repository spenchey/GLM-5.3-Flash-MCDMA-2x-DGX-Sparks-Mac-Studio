#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config

trap release_three_machine_lifecycle_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
acquire_three_machine_lifecycle_lock

proof_id="cache-handoff-proof-$(date -u +%Y%m%dT%H%M%SZ)-$$"
evidence="$PROJECT_ROOT/results/raw/$proof_id"
umask 077
mkdir -p "$evidence"
proof_complete=false
cleanup_failed_proof() {
  local rc=$?
  if (( rc != 0 )) && [[ "$proof_complete" != true ]]; then
    note "Proof failed; evidence is preserved at $evidence"
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
mcdma_link_healthy "$link_before" || die "MCDMA is not healthy before the proof"
mcdma_link_busy "$link_before" && die "MCDMA is busy before the proof"
printf '%s\n' "$head_before" >"$evidence/head-before.txt"
printf '%s\n' "$worker_before" >"$evidence/worker-before.txt"
printf '%s\n' "$link_before" >"$evidence/mcdma-before.json"

prompt='Output the integers from 1 through 2000 in order, separated only by commas. Do not explain and do not stop early.'
command=(env
  "MCDMA_RPC_LIBRARY=$THREE_MACHINE_MCDMA_LIBRARY"
  "PYTHONPATH=$THREE_MACHINE_STUDIO_DIR:$THREE_MACHINE_TENSORFOLD_SOURCE"
  "$THREE_MACHINE_STUDIO_PYTHON" -m experiments.three_machine.mac_decode
  --model "$THREE_MACHINE_PORTABLE_MODEL_STUDIO_DIR"
  --prompt "$prompt" --max-new-tokens 128
  --mailbox "$THREE_MACHINE_MAILBOX" --timeout 600 --compare-local --parallel 4
  --chunk-mib "$THREE_MACHINE_HANDOFF_CHUNK_MIB")
printf -v remote 'cd %q && ' "$THREE_MACHINE_STUDIO_DIR"
printf -v command_text '%q ' "${command[@]}"
note "Running the four-request proof: both Sparks prepare every prompt; the Mac decodes one shared batch"
started_epoch=$(date +%s)
studio_ssh "$remote$command_text" \
  >"$evidence/result.json" 2>"$evidence/mac.stderr"
finished_epoch=$(date +%s)
printf '%s\n' "$((finished_epoch - started_epoch))" >"$evidence/wall-seconds.txt"

head_after=$(cache_handoff_container_state head)
worker_after=$(cache_handoff_container_state worker)
validate_cache_handoff_container_state "head Spark" "$head_after"
validate_cache_handoff_container_state "worker Spark" "$worker_after"
cache_handoff_container_running "$head_after" || die "head Spark stopped during the proof"
cache_handoff_container_running "$worker_after" || die "worker Spark stopped during the proof"
[[ "$(cache_handoff_container_id "$head_after")" == "$head_id" ]] \
  || die "head Spark identity changed during the proof"
[[ "$(cache_handoff_container_id "$worker_after")" == "$worker_id" ]] \
  || die "worker Spark identity changed during the proof"
link_after=$(adapter peek)
mcdma_link_healthy "$link_after" || die "MCDMA is not healthy after the proof"
mcdma_link_busy "$link_after" && die "MCDMA remained busy after the proof"
printf '%s\n' "$head_after" >"$evidence/head-after.txt"
printf '%s\n' "$worker_after" >"$evidence/worker-after.txt"
printf '%s\n' "$link_after" >"$evidence/mcdma-after.json"

python3 - "$evidence/result.json" "$evidence/mcdma-before.json" \
  "$evidence/mcdma-after.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
before = json.loads(Path(sys.argv[2]).read_text())
after = json.loads(Path(sys.argv[3]).read_text())
if result.get("event") != "glm_mcdma_decode":
    raise SystemExit(f"unexpected proof record: {result}")
if result.get("exact_token_match") is not True:
    raise SystemExit("Spark-prefilled and Mac-prefilled output tokens differ")
if result.get("parallel") != 4:
    raise SystemExit("the proof did not run the required four simultaneous Mac replies")
if result.get("cached_tokens") != result.get("prompt_tokens", 0):
    raise SystemExit("the Sparks did not prepare the complete prompt prefix")
if result.get("mtp_cached_tokens") != result.get("prompt_tokens", 0) - 1:
    raise SystemExit("the Spark MTP prefix stopped at the wrong position")
if result.get("spark_reference_first_token") is None:
    raise SystemExit("the Spark prefill did not bind its first reply token")
if result.get("result", {}).get("first_token_source") != "mac_head_from_spark_final_hidden":
    raise SystemExit("the Mac did not generate the first reply token from the Spark prompt state")
cache_bytes = result.get("cache_bytes_per_request", 0)
if cache_bytes <= 0:
    raise SystemExit("no prompt state crossed MCDMA")
if result.get("cache_bytes_total") != cache_bytes:
    raise SystemExit("the single MCDMA prompt-state transfer has an invalid byte total")
if result.get("cache_wire_transfers") != 1 or result.get("cache_reuse_copies") != 3:
    raise SystemExit("the transferred prompt state was not reused across four Mac lanes")
remote = result.get("result", {})
local = result.get("local_reference", {})
if not remote.get("tokens") or remote.get("token_sha256") != local.get("token_sha256"):
    raise SystemExit("equal-answer proof is incomplete")
batch = result.get("batch", {})
batch_results = batch.get("results", [])
if batch.get("parallel") != 4 or len(batch_results) != 4:
    raise SystemExit("the Mac did not report four decoded replies")
if any(item.get("token_sha256") != local.get("token_sha256") for item in batch_results):
    raise SystemExit("at least one Mac batch reply differs from the local reference")
decode_rate = float(batch.get("aggregate_decode_tokens_per_second", 0))
if decode_rate <= 0:
    raise SystemExit("the Mac returned no positive aggregate decode rate")
pipeline_seconds = (
    float(result.get("transfer_seconds", 0))
    + float(result.get("import_seconds", 0))
    + float(batch.get("total_seconds", 0))
)
completion_tokens = sum(len(item.get("tokens", [])) for item in batch_results)
if pipeline_seconds <= 0 or completion_tokens <= 0:
    raise SystemExit("the full prompt-transfer-decode pipeline timing is incomplete")
for side in ("mac", "spark"):
    if after.get(side, {}).get("calls", 0) <= before.get(side, {}).get("calls", 0):
        raise SystemExit(f"MCDMA {side} call counter did not advance")
    if after.get(side, {}).get("failures") != 0:
        raise SystemExit(f"MCDMA {side} reports a failure")
print(json.dumps({
    "aggregate_decode_tokens_per_second": decode_rate,
    "aggregate_pipeline_tokens_per_second": completion_tokens / pipeline_seconds,
    "cache_bytes_per_request": cache_bytes,
    "cache_bytes_total": result["cache_bytes_total"],
    "cache_reuse_copies": result["cache_reuse_copies"],
    "cache_wire_transfers": result["cache_wire_transfers"],
    "cached_tokens": result["cached_tokens"],
    "exact_token_match": True,
    "output_token_sha256": remote["token_sha256"],
    "parallel": 4,
    "pipeline_seconds": pipeline_seconds,
    "prompt_tokens": result["prompt_tokens"],
    "transfer_seconds": result["transfer_seconds"],
}, sort_keys=True))
PY

head_ssh "docker logs '$head_id' 2>&1" >"$evidence/head.log"
worker_ssh "docker logs '$worker_id' 2>&1" >"$evidence/worker.log"
head_ssh "docker inspect '$head_id'" >"$evidence/head-inspect.json"
worker_ssh "docker inspect '$worker_id'" >"$evidence/worker-inspect.json"
proof_complete=true
note "PASS: equal tokens, one verified MCDMA cache transfer reused by four Mac replies, and measured batch decode; evidence is at $evidence"
