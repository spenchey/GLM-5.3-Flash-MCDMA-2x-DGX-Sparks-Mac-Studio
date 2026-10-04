#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config

mode=warm
decoder=mtp
handoff=sealed
progressive_frame_mib=32
while (( $# )); do
  case "$1" in
    --cold) mode=cold ;;
    --dflash) decoder=dflash ;;
    --progressive) handoff=progressive ;;
    --progressive-frame-mib)
      (( $# >= 2 )) || die "--progressive-frame-mib needs a value"
      progressive_frame_mib=$2
      shift
      ;;
    *) die "usage: $0 [--cold] [--dflash] [--progressive] [--progressive-frame-mib MIB]" ;;
  esac
  shift
done
[[ "$progressive_frame_mib" =~ ^[0-9]+$ ]] \
  && (( progressive_frame_mib >= 1 && progressive_frame_mib <= THREE_MACHINE_HANDOFF_CHUNK_MIB )) \
  || die "--progressive-frame-mib must be between 1 and $THREE_MACHINE_HANDOFF_CHUNK_MIB"

trap release_three_machine_lifecycle_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
acquire_three_machine_lifecycle_lock

handoff_label=$handoff
[[ "$handoff" != progressive ]] || handoff_label="progressive-${progressive_frame_mib}m"
benchmark_id="mia-c1-three-machine-${mode}-${decoder}-${handoff_label}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
evidence="$PROJECT_ROOT/results/raw/$benchmark_id"
umask 077
mkdir -p "$evidence"
benchmark_complete=false
cleanup_failed_benchmark() {
  local rc=$?
  if (( rc != 0 )) && [[ "$benchmark_complete" != true ]]; then
    note "Benchmark did not pass; evidence is preserved at $evidence" >&2
    "$PROJECT_ROOT/scripts/stop-cache-handoff.sh" \
      >"$evidence/cleanup.stdout" 2>"$evidence/cleanup.stderr" || true
  fi
  release_three_machine_lifecycle_lock
  return "$rc"
}
trap cleanup_failed_benchmark EXIT

head_before=$(cache_handoff_container_state head)
worker_before=$(cache_handoff_container_state worker)
validate_cache_handoff_container_state "head Spark" "$head_before"
validate_cache_handoff_container_state "worker Spark" "$worker_before"
cache_handoff_container_running "$head_before" || die "head Spark prompt rank is not running"
cache_handoff_container_running "$worker_before" || die "worker Spark prompt rank is not running"
head_id=$(cache_handoff_container_id "$head_before")
worker_id=$(cache_handoff_container_id "$worker_before")
link_before=$(adapter peek)
mcdma_link_healthy "$link_before" || die "MCDMA is not healthy before the benchmark"
mcdma_link_busy "$link_before" && die "MCDMA is busy before the benchmark"
printf '%s\n' "$head_before" >"$evidence/head-before.txt"
printf '%s\n' "$worker_before" >"$evidence/worker-before.txt"
printf '%s\n' "$link_before" >"$evidence/mcdma-before.json"

prompt='Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough.'
command=(env
  "MCDMA_RPC_LIBRARY=$THREE_MACHINE_MCDMA_LIBRARY"
  "PYTHONPATH=$THREE_MACHINE_STUDIO_DIR:$THREE_MACHINE_TENSORFOLD_SOURCE"
  "$THREE_MACHINE_STUDIO_PYTHON" -m experiments.three_machine.mac_decode
  --model "$THREE_MACHINE_PORTABLE_MODEL_STUDIO_DIR"
  --prompt "$prompt"
  --max-new-tokens 400 --force-length
  --mailbox "$THREE_MACHINE_MAILBOX" --timeout 600
  --compare-local --allow-local-drift
  --no-copy-drafts
  --parallel 1)
if [[ "$handoff" == progressive ]]; then
  command+=(--progressive-handoff --chunk-mib "$progressive_frame_mib")
else
  command+=(--chunk-mib "$THREE_MACHINE_HANDOFF_CHUNK_MIB")
fi
if [[ "$decoder" == dflash ]]; then
  command+=(--mtp-drafts 2 --drafter "$THREE_MACHINE_DFLASH_STUDIO_DIR" --drafter-bits 4)
else
  command+=(--mtp-drafts 1)
fi
if [[ "$mode" == cold ]]; then
  command+=(--warmups 0 --repeats 5)
  note "Running five complete requests; every prompt is rebuilt on both Sparks and transferred to the Mac"
else
  command+=(--warmups 1 --warmup-new-tokens 32 --repeats 5 --reuse-warm-prefix)
  note "Running Mia's warmed C1 prose shape: 32-token warmup, then five fixed 400-token requests"
fi
printf -v remote 'cd %q && ' "$THREE_MACHINE_STUDIO_DIR"
printf -v command_text '%q ' "${command[@]}"
studio_ssh "$remote$command_text" \
  >"$evidence/result.json" 2>"$evidence/mac.stderr"

head_after=$(cache_handoff_container_state head)
worker_after=$(cache_handoff_container_state worker)
validate_cache_handoff_container_state "head Spark" "$head_after"
validate_cache_handoff_container_state "worker Spark" "$worker_after"
cache_handoff_container_running "$head_after" || die "head Spark stopped during the benchmark"
cache_handoff_container_running "$worker_after" || die "worker Spark stopped during the benchmark"
[[ "$(cache_handoff_container_id "$head_after")" == "$head_id" ]] \
  || die "head Spark identity changed during the benchmark"
[[ "$(cache_handoff_container_id "$worker_after")" == "$worker_id" ]] \
  || die "worker Spark identity changed during the benchmark"
[[ "$head_after" == *'|true|false|0|cache-handoff' ]] \
  || die "head Spark reported an unhealthy or OOM state"
[[ "$worker_after" == *'|true|false|0|cache-handoff' ]] \
  || die "worker Spark reported an unhealthy or OOM state"
link_after=$(adapter peek)
mcdma_link_healthy "$link_after" || die "MCDMA is not healthy after the benchmark"
mcdma_link_busy "$link_after" && die "MCDMA remained busy after the benchmark"
printf '%s\n' "$head_after" >"$evidence/head-after.txt"
printf '%s\n' "$worker_after" >"$evidence/worker-after.txt"
printf '%s\n' "$link_after" >"$evidence/mcdma-after.json"
head_ssh "docker logs '$head_id' 2>&1" >"$evidence/head.log"
worker_ssh "docker logs '$worker_id' 2>&1" >"$evidence/worker.log"
head_ssh "docker inspect '$head_id'" >"$evidence/head-inspect.json"
worker_ssh "docker inspect '$worker_id'" >"$evidence/worker-inspect.json"

set +e
python3 - "$evidence/result.json" "$evidence/mcdma-before.json" \
  "$evidence/mcdma-after.json" "$mode" "$decoder" "$handoff" \
  "$progressive_frame_mib" <<'PY' | tee "$evidence/summary.json"
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
before = json.loads(Path(sys.argv[2]).read_text())
after = json.loads(Path(sys.argv[3]).read_text())
mode = sys.argv[4]
decoder = sys.argv[5]
handoff = sys.argv[6]
progressive_frame_mib = int(sys.argv[7])
expected_protocol = 4 if handoff == "progressive" else 2

mia_decode_tps = 60.4
mia_ttft_seconds = 0.170
mia_complete_seconds = mia_ttft_seconds + 399 / mia_decode_tps
required_complete_seconds = mia_complete_seconds * 0.97

if result.get("event") != "glm_mcdma_benchmark":
    raise SystemExit("unexpected benchmark record")
expected = {
    "parallel": 1,
    "max_new_tokens": 400,
    "force_length": True,
    "allow_local_drift": True,
    "reuse_warm_prefix": mode == "warm",
    "progressive_handoff": handoff == "progressive",
    "copy_drafts": False,
    "mtp_drafts": 2 if decoder == "dflash" else 1,
    "warmup_count": 1 if mode == "warm" else 0,
    "warmup_new_tokens": 32 if mode == "warm" else 400,
    "repeat_count": 5,
}
if handoff == "progressive":
    expected["chunk_mib"] = progressive_frame_mib
for key, value in expected.items():
    if result.get(key) != value:
        raise SystemExit(f"benchmark contract differs for {key}: {result.get(key)!r}")
if decoder == "dflash":
    if not result.get("drafter") or result.get("drafter_bits") != 4:
        raise SystemExit("DFlash2 was requested but is absent from the benchmark record")
else:
    if result.get("drafter") is not None or result.get("drafter_bits") is not None:
        raise SystemExit("the MTP control unexpectedly loaded an external drafter")
runs = result.get("runs") or []
if len(runs) != 5:
    raise SystemExit("benchmark did not preserve five measured runs")
if not result.get("output_token_sha256"):
    raise SystemExit("benchmark has no stable output-token digest")
prefix_setup = result.get("prefix_setup") or {}
if mode == "warm":
    if prefix_setup.get("cache_wire_transfers") != 1 or prefix_setup.get("cache_bytes", 0) <= 0:
        raise SystemExit("the warm prefix did not cross MCDMA exactly once")
    if not prefix_setup.get("handoff_telemetry"):
        raise SystemExit("the warm-prefix handoff has no phase telemetry")
    if prefix_setup.get("cache_protocol") != expected_protocol:
        raise SystemExit(
            f"the warm-prefix handoff used cache protocol {prefix_setup.get('cache_protocol')!r}, "
            f"not {expected_protocol}"
        )
elif prefix_setup:
    raise SystemExit("a complete-request benchmark unexpectedly retained a warm prefix")
for run in runs:
    if run.get("cache_protocol") != expected_protocol:
        raise SystemExit(
            f"a measured run used cache protocol {run.get('cache_protocol')!r}, "
            f"not {expected_protocol}"
        )
    if run.get("completion_tokens") != 400:
        raise SystemExit("a measured run ended before 400 tokens")
    if run.get("cached_tokens") != run.get("prompt_tokens", 0):
        raise SystemExit("the Sparks did not prepare the complete prompt prefix")
    if run.get("mtp_cached_tokens") != run.get("prompt_tokens", 0) - 1:
        raise SystemExit("the Spark MTP prefix stopped at the wrong position")
    if run.get("spark_reference_first_token") is None:
        raise SystemExit("the Spark prefill did not bind its first reply token")
    if run.get("result", {}).get("first_token_source") != "mac_head_from_spark_final_hidden":
        raise SystemExit("the Mac did not generate the first reply token from the Spark prompt state")
    if run.get("cache_bytes_per_request", 0) <= 0:
        raise SystemExit("a measured run has no retained prompt state")
    drafter_telemetry = run.get("result", {}).get("drafter_telemetry")
    if decoder == "dflash":
        if not isinstance(drafter_telemetry, dict) or drafter_telemetry.get("dflash_proposals", 0) <= 0:
            raise SystemExit("the measured Mac decode did not run the reviewed DFlash2 helper")
    elif drafter_telemetry is not None:
        raise SystemExit("the MTP control unexpectedly reported DFlash2 telemetry")
    if mode == "warm":
        if run.get("cache_wire_transfers") != 0 or run.get("warm_prefix_reused") is not True:
            raise SystemExit("a measured run rebuilt the warmed prefix")
    else:
        if run.get("cache_wire_transfers") != 1 or run.get("warm_prefix_reused") is not False:
            raise SystemExit("a complete request did not rebuild and transfer the Spark prefix")
        telemetry = run.get("handoff_telemetry") or {}
        if handoff == "progressive":
            required_phases = (
                "prepare_call_seconds", "plan_decode_seconds", "frame_call_seconds",
                "frame_assemble_seconds", "seal_call_seconds", "verify_seconds",
                "release_call_seconds", "total_seconds", "job_id", "plan_id",
                "transfer_id",
            )
            if telemetry.get("protocol") != 4:
                raise SystemExit("a progressive transfer did not attest protocol 4")
            if telemetry.get("frame_strategy") != "execution_order_tensor_groups":
                raise SystemExit("a progressive transfer used the wrong frame order")
        else:
            required_phases = (
                "prepare_call_seconds", "frame_call_seconds", "frame_assemble_seconds",
                "verify_seconds", "release_call_seconds", "total_seconds",
            )
        if any(name not in telemetry for name in required_phases):
            raise SystemExit("a complete request is missing handoff phase telemetry")
if len({run["result"]["token_sha256"] for run in runs}) != 1:
    raise SystemExit("greedy measured outputs differed")
for side in ("mac", "spark"):
    if after.get(side, {}).get("calls", 0) <= before.get(side, {}).get("calls", 0):
        raise SystemExit(f"MCDMA {side} call counter did not advance")
    if after.get(side, {}).get("failures") != 0:
        raise SystemExit(f"MCDMA {side} reports a failure")

median_complete = float(result["complete_request_seconds"]["median"])
max_complete = float(result["complete_request_seconds"]["max"])
median_ttft = float(result["ttft_seconds"]["median"])
median_decode_tps = float(result["decode_tokens_per_second"]["median"])
exact_local_match = all(run.get("exact_token_match") is True for run in runs)
first_local_difference = min(
    (run.get("first_local_token_difference") for run in runs if run.get("first_local_token_difference") is not None),
    default=None,
)
passes = {
    "complete_time_3_percent_faster": median_complete <= required_complete_seconds,
    "ttft_no_regression": median_ttft <= mia_ttft_seconds,
    "tail_no_regression": max_complete <= mia_complete_seconds,
    "exact_same_checkpoint_output": exact_local_match,
    "mcdma_zero_failures": True,
}
summary = {
    "status": "PASS" if all(passes.values()) else "FAIL",
    "comparison": "MiaAI-Lab v1.5 published C1 prose claim",
    "request_mode": mode,
    "decoder": decoder,
    "handoff": handoff,
    "cache_protocol": expected_protocol,
    "progressive_frame_mib": progressive_frame_mib if handoff == "progressive" else None,
    "mia_claim": {
        "decode_tokens_per_second": mia_decode_tps,
        "ttft_seconds": mia_ttft_seconds,
        "implied_complete_seconds": mia_complete_seconds,
    },
    "required_three_machine_complete_seconds": required_complete_seconds,
    "three_machine": {
        "median_complete_seconds": median_complete,
        "max_complete_seconds": max_complete,
        "median_ttft_seconds": median_ttft,
        "median_decode_tokens_per_second": median_decode_tps,
        "output_token_sha256": result["output_token_sha256"],
        "first_local_token_difference": first_local_difference,
    },
    "passes": passes,
}
print(json.dumps(summary, sort_keys=True))
raise SystemExit(0 if summary["status"] == "PASS" else 3)
PY
analysis_rc=${PIPESTATUS[0]}
set -e

if (( analysis_rc != 0 )); then
  exit "$analysis_rc"
fi
benchmark_complete=true
if [[ "$mode" == cold ]]; then
  note "PASS: the complete three-machine request beat Mia's published C1 target; evidence is at $evidence"
else
  note "PASS: the warmed three-machine path beat Mia's published C1 target; evidence is at $evidence"
fi
