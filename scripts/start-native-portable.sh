#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config

trap release_three_machine_lifecycle_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
acquire_three_machine_lifecycle_lock

"$PROJECT_ROOT/scripts/preflight-cache-handoff.sh"

container=$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER
for role in head worker; do
  case "$role" in
    head) inspect=$(head_ssh "docker inspect '$container' --format '{{.State.Running}}' 2>/dev/null || true") ;;
    worker) inspect=$(worker_ssh "docker inspect '$container' --format '{{.State.Running}}' 2>/dev/null || true") ;;
  esac
  [[ -z "$inspect" ]] || die "$role already has $container; stop the owned native reference first"
  handoff=$(cache_handoff_container_state "$role")
  [[ -z "$handoff" ]] || die "$role still has the cache-handoff service; stop it before the native reference"
done

run_token="native-portable-$(date -u +%Y%m%dT%H%M%SZ)-$$"
head_id=""
worker_id=""
complete=false
cleanup_failed_start() {
  local rc=$? evidence
  if (( rc != 0 )) && [[ "$complete" != true ]]; then
    evidence="$PROJECT_ROOT/results/raw/failed-$run_token"
    mkdir -p "$evidence"
    if [[ -n "$head_id" ]]; then
      head_ssh "docker logs '$head_id' 2>&1" >"$evidence/head.log" || true
      head_ssh "docker inspect '$head_id' --format '{{json .State}}'" >"$evidence/head-state.json" || true
      head_ssh "docker rm -f '$head_id' >/dev/null 2>&1" || true
    fi
    if [[ -n "$worker_id" ]]; then
      worker_ssh "docker logs '$worker_id' 2>&1" >"$evidence/worker.log" || true
      worker_ssh "docker inspect '$worker_id' --format '{{json .State}}'" >"$evidence/worker-state.json" || true
      worker_ssh "docker rm -f '$worker_id' >/dev/null 2>&1" || true
    fi
    note "Startup failed; evidence is preserved at $evidence"
  fi
  release_three_machine_lifecycle_lock
  return "$rc"
}
trap cleanup_failed_start EXIT

common=(
  --gpus all --init --ipc=host --network=host --shm-size=16g
  --device=/dev/infiniband --cap-add=IPC_LOCK
  --ulimit=memlock=-1 --ulimit=stack=67108864
  --label "glm.mcdma.mode=native-portable" --label "glm.mcdma.run=$run_token"
  --label "glm.mcdma.drafter=dflash2"
  -e HF_HUB_OFFLINE=1 -e TENSORFOLD_NO_UPDATE_CHECK=1
  -e TF_GLM_DENSE=q4 -e TF_GLM_KDA_CHUNKED=1 -e "TF_GLM_L2PF=$THREE_MACHINE_L2PF"
  -e TF_GLM_HC_SPLIT=1 -e TF_GLM_PREFILL_OVERLAP=2 -e TF_GLM_KV=fp8
  -e TF_GLM_COMM=roce -e TF_ROCE_MAX_KB=512
  -e "NCCL_SOCKET_IFNAME=$THREE_MACHINE_NCCL_INTERFACE"
  -e "NCCL_IB_HCA=$THREE_MACHINE_NCCL_HCA" -e NCCL_IB_GID_INDEX=3
  -e NCCL_NET_PLUGIN=spcx -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4
  -v "$THREE_MACHINE_HF_CACHE_DIR:/root/.cache/huggingface:ro"
  -v "$THREE_MACHINE_TENSORFOLD_CACHE_DIR:/cache"
)
serve=(python3 -m tensorfold serve "$THREE_MACHINE_PORTABLE_MODEL_CONTAINER_DIR"
  --backend cuda --tp 2 --master "$THREE_MACHINE_NCCL_MASTER"
  --master-port "$THREE_MACHINE_NATIVE_PORTABLE_NCCL_PORT"
  --name GLM-5.3-Flash-Portable --parallel 4 --context 2051
  --drafter incoai/GLM-5.3-Flash-DFlash2
  --mtp-drafts "$THREE_MACHINE_MTP_DRAFTS"
  --temperature 0 --no-thinking --no-update-check)

note "Starting the worker rank for the explicit DFlash2 two-Spark reference"
worker_command=(docker run -d --name "$container" "${common[@]}" "$THREE_MACHINE_IMAGE_ID"
  "${serve[@]}" --rank 1 --host 127.0.0.1 --port "$THREE_MACHINE_NATIVE_PORTABLE_PORT")
printf -v worker_remote '%q ' "${worker_command[@]}"
worker_id=$(worker_ssh "$worker_remote")
[[ "$worker_id" =~ ^[0-9a-f]{64}$ ]] || die "worker native-reference launch returned no container ID"

note "Starting the head rank for the explicit DFlash2 two-Spark reference"
head_command=(docker run -d --name "$container" "${common[@]}" "$THREE_MACHINE_IMAGE_ID"
  "${serve[@]}" --rank 0 --host 0.0.0.0 --port "$THREE_MACHINE_NATIVE_PORTABLE_PORT")
printf -v head_remote '%q ' "${head_command[@]}"
head_id=$(head_ssh "$head_remote")
[[ "$head_id" =~ ^[0-9a-f]{64}$ ]] || die "head native-reference launch returned no container ID"

deadline=$((SECONDS + 1200))
while (( SECONDS < deadline )); do
  head_state=$(head_ssh "docker inspect '$head_id' --format '{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}'")
  worker_state=$(worker_ssh "docker inspect '$worker_id' --format '{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}'")
  [[ "$head_state" == true\|false\|0 ]] || die "head native-reference rank stopped: $head_state"
  [[ "$worker_state" == true\|false\|0 ]] || die "worker native-reference rank stopped: $worker_state"
  if head_ssh "curl -fsS --max-time 5 'http://127.0.0.1:$THREE_MACHINE_NATIVE_PORTABLE_PORT/health' >/dev/null"; then
    complete=true
    note "PASS: explicit DFlash2 two-Spark reference is ready on port $THREE_MACHINE_NATIVE_PORTABLE_PORT"
    exit 0
  fi
  sleep 3
done
die "same-model two-Spark reference did not become ready within 20 minutes"
