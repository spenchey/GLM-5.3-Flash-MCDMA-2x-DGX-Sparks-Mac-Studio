#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config

trap release_three_machine_lifecycle_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
acquire_three_machine_lifecycle_lock

"$PROJECT_ROOT/scripts/preflight-cache-handoff.sh"

for legacy in "$THREE_MACHINE_CONTAINER" dsv41-exl3-head dsv41-exl3-worker glm53-flash-tf; do
  [[ "$(head_ssh "docker inspect '$legacy' --format '{{.State.Running}}' 2>/dev/null || true")" != true ]] \
    || die "$legacy is still running on the head Spark"
  [[ "$(worker_ssh "docker inspect '$legacy' --format '{{.State.Running}}' 2>/dev/null || true")" != true ]] \
    || die "$legacy is still running on the worker Spark"
done
for role in head worker; do
  case "$role" in
    head)
      native_running=$(head_ssh "docker inspect '$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER' --format '{{.State.Running}}' 2>/dev/null || true")
      ;;
    worker)
      native_running=$(worker_ssh "docker inspect '$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER' --format '{{.State.Running}}' 2>/dev/null || true")
      ;;
  esac
  [[ "$native_running" != true ]] \
    || die "the temporary native reference is still using the $role Spark"
done

run_token="cache-handoff-$(date -u +%Y%m%dT%H%M%SZ)-$$"
head_id=""
worker_id=""
started=false
complete=false
cleanup_failed_start() {
  local rc=$? evidence
  if (( rc != 0 )) && [[ "$complete" != true ]]; then
    evidence="$PROJECT_ROOT/results/raw/failed-cache-handoff-$run_token"
    mkdir -p "$evidence"
    if [[ -n "$head_id" ]]; then
      head_ssh "docker logs '$head_id' 2>&1" >"$evidence/head.log" || true
      head_ssh "docker rm -f '$head_id' >/dev/null 2>&1" || true
    fi
    if [[ -n "$worker_id" ]]; then
      worker_ssh "docker logs '$worker_id' 2>&1" >"$evidence/worker.log" || true
      worker_ssh "docker rm -f '$worker_id' >/dev/null 2>&1" || true
    fi
    if [[ "$started" == true ]]; then
      adapter cleanup >"$evidence/mcdma-cleanup.json" 2>"$evidence/mcdma-cleanup.stderr" || true
    fi
    note "Startup failed; evidence is preserved at $evidence"
  fi
  release_three_machine_lifecycle_lock
  return "$rc"
}
trap cleanup_failed_start EXIT

started=true
link=$(adapter snapshot)
mcdma_transport_healthy "$link" || die "MCDMA transport did not become healthy"
head_ssh "test -S '/tmp/mcdma-rpcd.$THREE_MACHINE_MAILBOX.sock'" \
  || die "the head Spark MCDMA control socket is absent"

common=(
  --gpus all --init --ipc=host --network=host --shm-size=16g
  --device=/dev/infiniband --cap-add=IPC_LOCK
  --ulimit=memlock=-1 --ulimit=stack=67108864
  --label "glm.mcdma.mode=cache-handoff" --label "glm.mcdma.run=$run_token"
  -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/work
  -e TF_GLM_DENSE=q4 -e TF_GLM_KDA_CHUNKED=1 -e "TF_GLM_L2PF=$THREE_MACHINE_L2PF"
  # The Mac cache uses bfloat16 latent rows. Keep prompt export in that same
  # representation; TensorFold's NCCL wrapper does not carry raw uint8 rows.
  -e TF_GLM_HC_SPLIT=1 -e TF_GLM_PREFILL_OVERLAP=2 -e TF_GLM_KV=bf16
  -e TF_GLM_COMM=roce -e TF_ROCE_MAX_KB=512
  -e "NCCL_SOCKET_IFNAME=$THREE_MACHINE_NCCL_INTERFACE"
  -e "NCCL_IB_HCA=$THREE_MACHINE_NCCL_HCA" -e NCCL_IB_GID_INDEX=3
  -e NCCL_NET_PLUGIN=spcx -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4
  -v "$THREE_MACHINE_HF_CACHE_DIR:/root/.cache/huggingface:ro"
  -v "$THREE_MACHINE_TENSORFOLD_CACHE_DIR:/cache"
  -v "$THREE_MACHINE_PROJECT_DIR:/work:ro"
)

note "Starting the worker Spark prompt rank"
worker_command=(docker run -d --name "$THREE_MACHINE_HANDOFF_CONTAINER" "${common[@]}" \
  "$THREE_MACHINE_IMAGE_ID" python3 -m experiments.three_machine.cuda_prefill \
  --rank 1 --model "$THREE_MACHINE_PORTABLE_MODEL_CONTAINER_DIR" \
  --master "$THREE_MACHINE_NCCL_MASTER" --port "$THREE_MACHINE_HANDOFF_NCCL_PORT" \
  --prefill-rows "$THREE_MACHINE_HANDOFF_PREFILL_ROWS" --timeout 600)
printf -v worker_remote '%q ' "${worker_command[@]}"
worker_id=$(worker_ssh "$worker_remote")
[[ "$worker_id" =~ ^[0-9a-f]{64}$ ]] || die "worker launch did not return an exact container ID"

note "Starting the head Spark prompt rank and MCDMA cache service"
head_ssh "mkdir -p '$THREE_MACHINE_CACHE_BUNDLE_DIR' && test -d '$THREE_MACHINE_CACHE_BUNDLE_DIR' && test ! -L '$THREE_MACHINE_CACHE_BUNDLE_DIR' && chmod 700 '$THREE_MACHINE_CACHE_BUNDLE_DIR'"
head_command=(docker run -d --name "$THREE_MACHINE_HANDOFF_CONTAINER" "${common[@]}" \
  -e MCDMA_RPC_LIBRARY=/opt/mcdma/libmcdma-rpc.so \
  -v "$THREE_MACHINE_SPARK_MCDMA_LIBRARY:/opt/mcdma/libmcdma-rpc.so:ro" \
  -v "$THREE_MACHINE_CACHE_BUNDLE_DIR:/cache-bundles" \
  --mount "type=bind,src=/tmp/mcdma-rpcd.$THREE_MACHINE_MAILBOX.sock,dst=/run/mcdma/control.sock,readonly" \
  "$THREE_MACHINE_IMAGE_ID" python3 -m experiments.three_machine.cuda_prefill \
  --rank 0 --model "$THREE_MACHINE_PORTABLE_MODEL_CONTAINER_DIR" \
  --master "$THREE_MACHINE_NCCL_MASTER" --port "$THREE_MACHINE_HANDOFF_NCCL_PORT" \
  --mailbox "$THREE_MACHINE_MAILBOX" --socket /run/mcdma/control.sock \
  --archive-dir /cache-bundles \
  --prefill-rows "$THREE_MACHINE_HANDOFF_PREFILL_ROWS" --timeout 600)
printf -v head_remote '%q ' "${head_command[@]}"
head_id=$(head_ssh "$head_remote")
[[ "$head_id" =~ ^[0-9a-f]{64}$ ]] || die "head launch did not return an exact container ID"

deadline=$((SECONDS + 1200))
while (( SECONDS < deadline )); do
  head_running=$(head_ssh "docker inspect '$head_id' --format '{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}'")
  worker_running=$(worker_ssh "docker inspect '$worker_id' --format '{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}'")
  [[ "$head_running" == true\|false\|0 ]] || die "the head Spark rank stopped during model load: $head_running"
  [[ "$worker_running" == true\|false\|0 ]] || die "the worker Spark rank stopped during model load: $worker_running"
  link=$(adapter peek 2>/dev/null || true)
  if [[ -n "$link" ]] && python3 - "$link" <<'PY'
import json
import sys

state = json.loads(sys.argv[1])
raise SystemExit(0 if (
    state.get("link") == "up"
    and state.get("mac", {}).get("alive")
    and state.get("spark", {}).get("alive")
    and state.get("spark", {}).get("service") == "poll"
    and not state.get("mac", {}).get("failures")
    and not state.get("spark", {}).get("failures")
) else 1)
PY
  then
    complete=true
    note "PASS: both Sparks are loaded for prompt work and MCDMA is ready to hand the cache to the Mac"
    exit 0
  fi
  sleep 3
done
die "the cache-handoff service did not become ready within 20 minutes"
