#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config

trap release_three_machine_lifecycle_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
acquire_three_machine_lifecycle_lock

container=${THREE_MACHINE_SINGLE_STREAM_CONTAINER:-glm53-single-stream-native}
port=${THREE_MACHINE_SINGLE_STREAM_PORT:-8896}
nccl_port=${THREE_MACHINE_SINGLE_STREAM_NCCL_PORT:-29643}
context=${THREE_MACHINE_SINGLE_STREAM_CONTEXT:-2051}
source_dir=${THREE_MACHINE_SPARK_TENSORFOLD_065_SOURCE:-${THREE_MACHINE_SPARK_TENSORFOLD_064_SOURCE:-${THREE_MACHINE_PROJECT_DIR%/runtime}/tensorfold-v0.6.5}}
source_tree=${TENSORFOLD_LATEST_RELEASE_SOURCE_SHA256:?missing current TensorFold source pin}

[[ "$TENSORFOLD_LATEST_RELEASE" == 0.6.5 ]] \
  || die "the single-stream reference is pinned to TensorFold 0.6.5"
[[ "$container" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]] \
  || die "single-stream container name is unsafe"
[[ "$port" =~ ^[0-9]+$ && "$nccl_port" =~ ^[0-9]+$ && "$context" =~ ^[1-9][0-9]*$ ]] \
  || die "single-stream ports and context must be positive whole numbers"

manifest="python3 '$THREE_MACHINE_PROJECT_DIR/scripts/tree-manifest.py' '$source_dir'"
[[ "$(head_ssh "$manifest")" == "$source_tree" ]] \
  || die "head TensorFold 0.6.5 source differs from the pin"
[[ "$(worker_ssh "$manifest")" == "$source_tree" ]] \
  || die "worker TensorFold 0.6.5 source differs from the pin"

for role in head worker; do
  case "$role" in
    head) running=$(head_ssh "docker inspect '$container' --format '{{.State.Running}}' 2>/dev/null || true") ;;
    worker) running=$(worker_ssh "docker inspect '$container' --format '{{.State.Running}}' 2>/dev/null || true") ;;
  esac
  [[ -z "$running" ]] || die "$role already has $container"
done

run_token="native-single-v065-$(date -u +%Y%m%dT%H%M%SZ)-$$"
head_id=""
worker_id=""
complete=false
cleanup_failed_start() {
  local rc=$? evidence
  if ((rc != 0)) && [[ "$complete" != true ]]; then
    evidence="$PROJECT_ROOT/results/raw/failed-$run_token"
    mkdir -p "$evidence"
    if [[ -n "$head_id" ]]; then
      head_ssh "docker logs '$head_id' 2>&1" >"$evidence/head.log" || true
      head_ssh "docker inspect '$head_id'" >"$evidence/head-inspect.json" || true
      head_ssh "docker rm -f '$head_id' >/dev/null 2>&1" || true
    fi
    if [[ -n "$worker_id" ]]; then
      worker_ssh "docker logs '$worker_id' 2>&1" >"$evidence/worker.log" || true
      worker_ssh "docker inspect '$worker_id'" >"$evidence/worker-inspect.json" || true
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
  --label "glm.mcdma.mode=native-single-stream-v065"
  --label "glm.mcdma.run=$run_token"
  --label "glm.mcdma.tensorfold=$TENSORFOLD_LATEST_RELEASE_COMMIT"
  -e HF_HUB_OFFLINE=1 -e TENSORFOLD_NO_UPDATE_CHECK=1
  -e PYTHONPATH=/tf/src -e TF_GLM_MTP=1
  -e TF_GLM_DENSE=q4 -e TF_GLM_KDA_CHUNKED=1 -e "TF_GLM_L2PF=$THREE_MACHINE_L2PF"
  -e TF_GLM_HC_SPLIT=1 -e TF_GLM_PREFILL_OVERLAP=2 -e TF_GLM_KV=bf16
  -e TF_GLM_COMM=roce -e TF_ROCE_MAX_KB=512
  -e "NCCL_SOCKET_IFNAME=$THREE_MACHINE_NCCL_INTERFACE"
  -e "NCCL_IB_HCA=$THREE_MACHINE_NCCL_HCA" -e NCCL_IB_GID_INDEX=3
  -e NCCL_NET_PLUGIN=spcx -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4
  -v "$source_dir:/tf:ro"
  -v "$THREE_MACHINE_HF_CACHE_DIR:/root/.cache/huggingface:ro"
  -v "$THREE_MACHINE_TENSORFOLD_CACHE_DIR:/cache"
)
serve=(python3 -m tensorfold serve "$THREE_MACHINE_PORTABLE_MODEL_CONTAINER_DIR"
  --backend cuda --tp 2 --master "$THREE_MACHINE_NCCL_MASTER"
  --master-port "$nccl_port" --name GLM-5.3-Flash-Portable
  --parallel 1 --context "$context" --drafter none
  --mtp-drafts "$THREE_MACHINE_MTP_DRAFTS"
  --temperature 0 --no-thinking --no-update-check)

note "Starting TensorFold 0.6.5 worker rank for the one-request reference"
worker_command=(docker run -d --name "$container" "${common[@]}" "$THREE_MACHINE_IMAGE_ID"
  "${serve[@]}" --rank 1 --host 127.0.0.1 --port "$port")
printf -v worker_remote '%q ' "${worker_command[@]}"
worker_id=$(worker_ssh "$worker_remote")
[[ "$worker_id" =~ ^[0-9a-f]{64}$ ]] || die "worker launch returned no container ID"

note "Starting TensorFold 0.6.5 head rank for the one-request reference"
head_command=(docker run -d --name "$container" "${common[@]}" "$THREE_MACHINE_IMAGE_ID"
  "${serve[@]}" --rank 0 --host 0.0.0.0 --port "$port")
printf -v head_remote '%q ' "${head_command[@]}"
head_id=$(head_ssh "$head_remote")
[[ "$head_id" =~ ^[0-9a-f]{64}$ ]] || die "head launch returned no container ID"

deadline=$((SECONDS + 1200))
while ((SECONDS < deadline)); do
  head_state=$(head_ssh "docker inspect '$head_id' --format '{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}'")
  worker_state=$(worker_ssh "docker inspect '$worker_id' --format '{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}'")
  [[ "$head_state" == true\|false\|0 ]] || die "head rank stopped: $head_state"
  [[ "$worker_state" == true\|false\|0 ]] || die "worker rank stopped: $worker_state"
  if head_ssh "curl -fsS --max-time 5 'http://127.0.0.1:$port/health' >/dev/null"; then
    version=$(head_ssh "docker exec '$head_id' python3 -c 'import tensorfold; print(tensorfold.__version__)'")
    [[ "$version" == 0.6.5 ]] || die "running service imported TensorFold $version"
    complete=true
    note "PASS: current two-Spark one-request reference is ready on port $port"
    exit 0
  fi
  sleep 3
done
die "single-stream reference did not become ready within 20 minutes"
