#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_three_machine_config

trap release_three_machine_lifecycle_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
acquire_three_machine_lifecycle_lock

health_ready() {
  python3 - "$1" <<'PY'
import json, sys
s = json.loads(sys.argv[1])
raise SystemExit(0 if s.get("status") == "ready" and s.get("model") == "GLM-5.3-Flash-TensorFold-MCDMA-3Machine" else 1)
PY
}

link_ready() {
  python3 - "$1" <<'PY'
import json, sys
s = json.loads(sys.argv[1])
raise SystemExit(0 if (
    s.get("link") == "up"
    and s.get("mac", {}).get("alive")
    and s.get("spark", {}).get("alive")
    and s.get("spark", {}).get("service") == "poll"
    and s.get("mac", {}).get("failures") == 0
    and s.get("spark", {}).get("failures") == 0
) else 1)
PY
}

"$PROJECT_ROOT/scripts/preflight-three-machine.sh"

head_state=$(three_machine_container_state head)
worker_state=$(three_machine_container_state worker)
validate_three_machine_container_state "head Spark" "$head_state"
validate_three_machine_container_state "worker Spark" "$worker_state"
head_running=false
worker_running=false
three_machine_container_running "$head_state" && head_running=true
three_machine_container_running "$worker_state" && worker_running=true
if [[ "$head_running" == true && "$worker_running" == true ]]; then
  api_pid=$(verified_studio_api_listener_pid)
  health=$(studio_ssh "curl -fsS --max-time 3 'http://$THREE_MACHINE_API_HOST:$THREE_MACHINE_API_PORT/health'" 2>/dev/null || true)
  link=$(adapter peek 2>/dev/null || true)
  api_serving=false
  [[ -n "$health" ]] && health_ready "$health" && api_serving=true
  [[ -n "$link" ]] && mcdma_link_busy "$link" && api_serving=true
  if [[ -n "$api_pid" && "$api_serving" == true && -n "$link" ]] \
      && link_ready "$link"; then
    note "Three-machine TensorFold is already running and fully healthy"
    "$PROJECT_ROOT/scripts/status-three-machine.sh"
    exit 0
  fi
  die "both Spark ranks exist but the complete service is unhealthy; run recover-three-machine.sh"
fi
[[ "$head_running" != true && "$worker_running" != true ]] || die "only one Spark stage is running; inspect before restart"

api_pid=$(probe_studio_api_pid)
[[ -z "$api_pid" ]] || die "the verified Mac API is still running without both ranks; run recover-three-machine.sh"
if studio_api_port_open; then
  die "the configured API port is owned by another listener; refusing to start"
fi
stale_api_pid=$(studio_api_pid_file_value)
stale_api_start=$(studio_api_start_file_value)

for legacy in glm53-flash-tf dsv41-exl3-head dsv41-exl3-worker; do
  if ! legacy_head=$(head_ssh "docker inspect '$legacy' --format '{{.State.Running}}' 2>/dev/null || true"); then
    die "could not inspect $legacy on the head Spark"
  fi
  if ! legacy_worker=$(worker_ssh "docker inspect '$legacy' --format '{{.State.Running}}' 2>/dev/null || true"); then
    die "could not inspect $legacy on the worker Spark"
  fi
  [[ "$legacy_head" != true ]] || die "$legacy is still running on the head Spark"
  [[ "$legacy_worker" != true ]] || die "$legacy is still running on the worker Spark"
done

api_log_present=false
if ! api_log_state=$(studio_ssh "if test -s '$THREE_MACHINE_API_STATE_DIR/api.log'; then printf 'present\\n'; else printf 'absent\\n'; fi"); then
  die "could not inspect the prior Mac API log"
fi
[[ "$api_log_state" == present || "$api_log_state" == absent ]] \
  || die "Mac API log inspection returned an invalid state"
[[ "$api_log_state" != present ]] || api_log_present=true
if [[ -n "$head_state" || -n "$worker_state" || "$api_log_present" == true \
      || -n "$stale_api_pid" || -n "$stale_api_start" ]]; then
  replaced_id=$(date -u +%Y%m%dT%H%M%SZ)
  replaced_dir="$PROJECT_ROOT/results/raw/replaced-start-$replaced_id"
  umask 077
  mkdir -p "$replaced_dir"
  printf '%s\n' "$head_state" >"$replaced_dir/head-inspect.txt"
  printf '%s\n' "$worker_state" >"$replaced_dir/worker-inspect.txt"
  printf 'pid=%s\nstart=%s\n' "$stale_api_pid" "$stale_api_start" >"$replaced_dir/api-identity.txt"
  [[ -z "$head_state" ]] || head_ssh "docker inspect '$(three_machine_container_id "$head_state")'" >"$replaced_dir/head-full-inspect.json"
  [[ -z "$worker_state" ]] || worker_ssh "docker inspect '$(three_machine_container_id "$worker_state")'" >"$replaced_dir/worker-full-inspect.json"
  [[ -z "$head_state" ]] || head_ssh "docker logs '$(three_machine_container_id "$head_state")' 2>&1" >"$replaced_dir/head.log"
  [[ -z "$worker_state" ]] || worker_ssh "docker logs '$(three_machine_container_id "$worker_state")' 2>&1" >"$replaced_dir/worker.log"
  [[ "$api_log_present" != true ]] || studio_ssh "cat '$THREE_MACHINE_API_STATE_DIR/api.log'" >"$replaced_dir/api.log"
  note "Preserved prior stopped-container evidence at $replaced_dir"
fi
studio_ssh "rm -f '$THREE_MACHINE_API_STATE_DIR/api.identity'"
[[ -z "$head_state" ]] || head_ssh "docker rm '$(three_machine_container_id "$head_state")' >/dev/null"
[[ -z "$worker_state" ]] || worker_ssh "docker rm '$(three_machine_container_id "$worker_state")' >/dev/null"

transport_started=false
containers_started=false
api_started=false
api_launch_pid=""
api_launch_start=""
head_launch_id=""
worker_launch_id=""
run_token="$(date -u +%Y%m%dT%H%M%SZ)-$$"
start_complete=false
cleanup_failed_start() {
  local rc=$? failed_head_state failed_worker_state failed_name="" failed_id="" failed_image=""
  local failed_running=false failed_head_known=true failed_worker_known=true cleanup_api_pid="" cleanup_api_start=""
  local failure_id evidence_dir deadline final_state process_state identity_rc port_rc api_identity_unknown=false
  local api_owned=false force_owned=false cleanup_stderr=/dev/null failed_cleanup=""
  failed_start_state() {
    local role=${1:?role required} attempt state
    for attempt in 1 2 3; do
      if state=$(three_machine_container_state "$role" 2>/dev/null); then
        printf '%s\n' "$state"
        return 0
      fi
      note "Retrying $role container inspection during failed-start cleanup ($attempt/3)" >&2
      sleep 1
    done
    note "Could not verify the $role container after three attempts; it may still be running and recovery is required" >&2
    return 1
  }
  preserve_and_term_rank() {
    local role=$1 state=$2 expected_id=${3:-} remote_id
    [[ -n "$state" ]] || return 0
    IFS='|' read -r failed_name failed_id failed_image _ _ _ <<<"$state"
    if [[ "$failed_name" != "/$THREE_MACHINE_CONTAINER" \
        || ! "$failed_id" =~ ^[0-9a-f]{64}$ \
        || "$failed_image" != "$THREE_MACHINE_IMAGE_ID" ]]; then
      note "Refusing to inspect or signal an unverified $role container during failed-start cleanup" >&2
      return 0
    fi
    remote_id="$failed_id"
    if [[ -z "$evidence_dir" ]]; then
      note "No local evidence directory is available for the $role rank" >&2
    elif [[ "$role" == head ]]; then
      head_ssh "docker inspect '$remote_id'" >"$evidence_dir/head-full-inspect.json" 2>&1 \
        || note "Could not preserve the head container inspection" >&2
      head_ssh "docker logs '$remote_id' 2>&1" >"$evidence_dir/head.log" \
        || note "Could not preserve the head container log" >&2
    else
      worker_ssh "docker inspect '$remote_id'" >"$evidence_dir/worker-full-inspect.json" 2>&1 \
        || note "Could not preserve the worker container inspection" >&2
      worker_ssh "docker logs '$remote_id' 2>&1" >"$evidence_dir/worker.log" \
        || note "Could not preserve the worker container log" >&2
    fi
    if [[ -z "$expected_id" ]]; then
      note "No exact launch ID was returned for the $role rank; evidence was preserved but it will not be signaled" >&2
      return 0
    fi
    if [[ "$failed_id" != "$expected_id" ]]; then
      note "The $role container is not this start's exact launch ID; evidence was preserved but it will not be signaled" >&2
      return 0
    fi
    if [[ "$role" == head ]]; then
      head_ssh "docker kill --signal=TERM '$remote_id' >/dev/null 2>&1" \
        || note "Could not send TERM to the verified head container" >&2
    else
      worker_ssh "docker kill --signal=TERM '$remote_id' >/dev/null 2>&1" \
        || note "Could not send TERM to the verified worker container" >&2
    fi
  }
  force_rank_if_running() {
    local role=$1 state=$2 expected_id=${3:-}
    [[ -n "$state" ]] || return 0
    [[ -n "$expected_id" ]] || return 0
    IFS='|' read -r failed_name failed_id failed_image failed_running _ _ <<<"$state"
    if [[ "$failed_name" == "/$THREE_MACHINE_CONTAINER" \
        && "$failed_id" =~ ^[0-9a-f]{64}$ \
        && "$failed_image" == "$THREE_MACHINE_IMAGE_ID" \
        && "$failed_id" == "$expected_id" \
        && "$failed_running" == true ]]; then
      note "$role rank ignored TERM during failed startup; force-stopping only container $failed_id" >&2
      if [[ "$role" == head ]]; then
        head_ssh "docker kill '$failed_id' >/dev/null 2>&1" \
          || note "Could not KILL the verified head container" >&2
      else
        worker_ssh "docker kill '$failed_id' >/dev/null 2>&1" \
          || note "Could not KILL the verified worker container" >&2
      fi
    fi
  }
  if (( rc != 0 )) && [[ "$start_complete" != true ]]; then
    failure_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
    evidence_dir="$PROJECT_ROOT/results/raw/failed-start-$failure_id"
    umask 077
    if ! mkdir -p "$evidence_dir"; then
      note "Could not create the failed-start evidence directory; cleanup will continue without local evidence" >&2
      evidence_dir=""
    fi
    if [[ -n "$evidence_dir" ]]; then
      studio_ssh "test -r '$THREE_MACHINE_API_STATE_DIR/api.log' && cat '$THREE_MACHINE_API_STATE_DIR/api.log' || true" \
        >"$evidence_dir/api.log" 2>&1 \
        || note "Could not preserve the Mac API log" >&2
    fi
    if [[ "$api_started" == true ]]; then
      cleanup_api_pid="$api_launch_pid"
      cleanup_api_start="$api_launch_start"
      if [[ -z "$cleanup_api_pid" ]]; then
        if ! cleanup_api_pid=$(studio_api_pid_file_value 2>/dev/null); then
          api_identity_unknown=true
          cleanup_api_pid=""
          note "Could not read the Studio API identity during failed-start cleanup; recovery is required" >&2
        fi
      fi
      if [[ -z "$cleanup_api_start" ]]; then
        if ! cleanup_api_start=$(studio_api_start_file_value 2>/dev/null); then
          api_identity_unknown=true
          cleanup_api_start=""
          note "Could not read the Studio API start identity during failed-start cleanup; recovery is required" >&2
        fi
      fi
      if [[ -n "$cleanup_api_pid" && -n "$cleanup_api_start" ]]; then
        if ! process_state=$(studio_process_state "$cleanup_api_pid"); then
          process_state=unknown
        fi
        if [[ "$process_state" == dead ]]; then
          clear_studio_api_identity "$cleanup_api_pid" "$cleanup_api_start" \
            || note "Could not clear the crashed Mac API identity file" >&2
        else
          api_owned=false
          identity_rc=1
          if [[ "$cleanup_api_pid" == "$api_launch_pid" \
                && "$cleanup_api_start" == "$api_launch_start" ]]; then
            if studio_process_start_matches "$cleanup_api_pid" "$cleanup_api_start"; then
              api_owned=true
            else
              identity_rc=$?
            fi
          elif studio_api_process_matches "$cleanup_api_pid" "$cleanup_api_start"; then
            api_owned=true
          else
            identity_rc=$?
          fi
          if [[ "$api_owned" == true ]]; then
          studio_ssh "kill -TERM '$cleanup_api_pid'" \
            || note "Could not send TERM to the verified Mac API" >&2
          deadline=$((SECONDS + 5))
          process_state=alive
          while (( SECONDS < deadline )); do
            if ! process_state=$(studio_process_state "$cleanup_api_pid"); then
              process_state=unknown
              note "Could not inspect the Mac API after TERM; recovery is required" >&2
              break
            fi
            [[ "$process_state" != alive ]] && break
            sleep 1
          done
          if [[ "$process_state" == alive ]]; then
            force_owned=false
            if [[ "$cleanup_api_pid" == "$api_launch_pid" \
                  && "$cleanup_api_start" == "$api_launch_start" ]]; then
              studio_process_start_matches "$cleanup_api_pid" "$cleanup_api_start" \
                && force_owned=true
            else
              studio_api_process_matches "$cleanup_api_pid" "$cleanup_api_start" \
                && force_owned=true
            fi
            if [[ "$force_owned" == true ]]; then
              studio_ssh "kill -KILL '$cleanup_api_pid'" \
                || note "Could not KILL the verified Mac API" >&2
              deadline=$((SECONDS + 10))
              while (( SECONDS < deadline )); do
                if ! process_state=$(studio_process_state "$cleanup_api_pid"); then
                  process_state=unknown
                  break
                fi
                [[ "$process_state" != alive ]] && break
                sleep 1
              done
            else
              note "Mac API identity changed before forced cleanup; refusing to signal it" >&2
            fi
          fi
          if ! process_state=$(studio_process_state "$cleanup_api_pid"); then
            process_state=unknown
          fi
          if [[ "$process_state" != dead ]]; then
            note "The verified Mac API survived failed-start cleanup; recovery is required" >&2
          else
            clear_studio_api_identity "$cleanup_api_pid" "$cleanup_api_start" \
              || note "Could not clear the stopped Mac API identity file" >&2
          fi
          else
            note "Mac API ownership could not be re-proven (code $identity_rc); refusing to signal it and requiring recovery" >&2
          fi
        fi
      else
        if [[ "$api_identity_unknown" == true ]]; then
          note "Mac API ownership is unknown; refusing to infer absence from its listener" >&2
        elif studio_ssh "nc -z -w 1 '$THREE_MACHINE_API_HOST' '$THREE_MACHINE_API_PORT' >/dev/null 2>&1"; then
          note "The API port is active but no exact launched API process can be verified; recovery is required" >&2
        else
          port_rc=$?
          (( port_rc != 255 )) || note "The Studio was unreachable during API cleanup; recovery is required" >&2
        fi
      fi
    fi
    if [[ "$containers_started" == true ]]; then
      if ! failed_head_state=$(failed_start_state head); then
        failed_head_known=false
        failed_head_state=""
      fi
      if ! failed_worker_state=$(failed_start_state worker); then
        failed_worker_known=false
        failed_worker_state=""
      fi
      if [[ -n "$evidence_dir" ]]; then
        printf '%s\n' "$failed_head_state" >"$evidence_dir/head-state.txt"
        printf '%s\n' "$failed_worker_state" >"$evidence_dir/worker-state.txt"
      fi
      preserve_and_term_rank head "$failed_head_state" "$head_launch_id"
      preserve_and_term_rank worker "$failed_worker_state" "$worker_launch_id"
      sleep 5
      if [[ "$failed_head_known" == true ]] && ! failed_head_state=$(failed_start_state head); then
        failed_head_known=false
        failed_head_state=""
      fi
      [[ "$failed_head_known" != true ]] || force_rank_if_running head "$failed_head_state" "$head_launch_id"
      if [[ "$failed_worker_known" == true ]] && ! failed_worker_state=$(failed_start_state worker); then
        failed_worker_known=false
        failed_worker_state=""
      fi
      [[ "$failed_worker_known" != true ]] || force_rank_if_running worker "$failed_worker_state" "$worker_launch_id"
      sleep 2
      for role in head worker; do
        if final_state=$(failed_start_state "$role"); then
          if three_machine_container_running "$final_state"; then
            note "The verified $role rank survived failed-start cleanup; recovery is required" >&2
          fi
        else
          note "Failed-start cleanup cannot prove the $role rank stopped; recovery is required" >&2
        fi
      done
      if [[ "$failed_head_known" != true || "$failed_worker_known" != true ]]; then
        note "Failed-start cleanup could not prove both Spark ranks stopped; do not retry until recovery verifies them" >&2
      fi
    fi
    if [[ -n "$evidence_dir" ]]; then
      note "Preserved failed-start evidence at $evidence_dir" >&2
    else
      note "Failed-start evidence could not be preserved locally" >&2
    fi
    if [[ "$transport_started" == true ]]; then
      cleanup_stderr=/dev/null
      [[ -z "$evidence_dir" ]] || cleanup_stderr="$evidence_dir/mcdma-cleanup.stderr"
      if failed_cleanup=$(adapter cleanup 2>"$cleanup_stderr"); then
        [[ -z "$evidence_dir" ]] || printf '%s\n' "$failed_cleanup" >"$evidence_dir/mcdma-cleanup.json"
        validate_mcdma_cleanup "$failed_cleanup" || note "MCDMA failed-start cleanup reported a problem: $failed_cleanup"
      else
        note "MCDMA failed-start cleanup command failed"
      fi
    fi
  fi
  release_three_machine_lifecycle_lock
  return "$rc"
}
trap cleanup_failed_start EXIT

transport_started=true
link=$(adapter snapshot)
python3 - "$link" <<'PY'
import json, sys
s = json.loads(sys.argv[1])
if s["link"] != "up" or not s["mac"]["alive"] or not s["spark"]["alive"]:
    raise SystemExit("MCDMA did not become ready")
if s["mac"]["failures"] or s["spark"]["failures"]:
    raise SystemExit("MCDMA reports a failure before TensorFold start")
PY
head_ssh "test -S '/tmp/mcdma-rpcd.$THREE_MACHINE_MAILBOX.sock'" \
  || die "the head Spark MCDMA control path is not a socket"

common=(
  --gpus all --init --ipc=host --network=host --shm-size=16g
  --device=/dev/infiniband --cap-add=IPC_LOCK
  --ulimit=memlock=-1 --ulimit=stack=67108864
  --label "glm.mcdma.run=$run_token"
  -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/work
  -e TF_GLM_DENSE=q4 -e TF_GLM_EXL3_LOADS=nc -e TF_GLM_KDA_CHUNKED=1 -e "TF_GLM_L2PF=$THREE_MACHINE_L2PF"
  -e TF_GLM_HC_SPLIT=1 -e TF_GLM_PREFILL_OVERLAP=2 -e TF_GLM_KV=fp8
  -e TF_GLM_COMM=roce -e TF_ROCE_MAX_KB=512 -e TF_GLM_WIDE_GRAPHS=16 -e TF_GLM_MTP=0
  -e "THREE_MACHINE_PHASE_TIMING=$THREE_MACHINE_PHASE_TIMING"
  -e "THREE_MACHINE_MTP_DRAFTS=$THREE_MACHINE_MTP_DRAFTS"
  -e "THREE_MACHINE_MTP_CONFIDENCE=$THREE_MACHINE_MTP_CONFIDENCE"
  -e "NCCL_SOCKET_IFNAME=$THREE_MACHINE_NCCL_INTERFACE"
  -e "NCCL_IB_HCA=$THREE_MACHINE_NCCL_HCA" -e NCCL_IB_GID_INDEX=3
  -e NCCL_NET_PLUGIN=spcx -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4
  -v "$THREE_MACHINE_HF_CACHE_DIR:/root/.cache/huggingface:ro"
  -v "$THREE_MACHINE_TENSORFOLD_CACHE_DIR:/cache"
  -v "$THREE_MACHINE_PROJECT_DIR:/work:ro"
)

note "Starting TensorFold CUDA rank 1 on the worker Spark"
containers_started=true
worker_command=(docker run -d --name "$THREE_MACHINE_CONTAINER" \
  "${common[@]}" "$THREE_MACHINE_IMAGE_ID" \
  python3 /work/experiments/three_machine/cuda_main.py \
  --rank 1 --model "$THREE_MACHINE_MODEL_CONTAINER_DIR" --master "$THREE_MACHINE_NCCL_MASTER" \
  --port "$THREE_MACHINE_NCCL_PORT" --capacity 2051 --prefill-rows 64 --timeout 180)
printf -v worker_remote '%q ' "${worker_command[@]}"
worker_launch_id=$(worker_ssh "$worker_remote")
[[ "$worker_launch_id" =~ ^[0-9a-f]{64}$ ]] || die "worker launch did not return an exact container ID"
[[ "$(worker_ssh "docker inspect '$worker_launch_id' --format '{{index .Config.Labels \"glm.mcdma.run\"}}'")" == "$run_token" ]] \
  || die "worker launch label does not match this start"
note "Starting TensorFold CUDA rank 0 on the head Spark"
head_command=(docker run -d --name "$THREE_MACHINE_CONTAINER" \
  "${common[@]}" \
  -e MCDMA_RPC_LIBRARY=/opt/mcdma/libmcdma-rpc.so \
  -v "$THREE_MACHINE_SPARK_MCDMA_LIBRARY:/opt/mcdma/libmcdma-rpc.so:ro" \
  --mount "type=bind,src=/tmp/mcdma-rpcd.$THREE_MACHINE_MAILBOX.sock,dst=/run/mcdma/control.sock,readonly" \
  "$THREE_MACHINE_IMAGE_ID" \
  python3 /work/experiments/three_machine/cuda_main.py \
  --rank 0 --model "$THREE_MACHINE_MODEL_CONTAINER_DIR" --master "$THREE_MACHINE_NCCL_MASTER" \
  --port "$THREE_MACHINE_NCCL_PORT" --mailbox "$THREE_MACHINE_MAILBOX" \
  --socket /run/mcdma/control.sock \
  --capacity 2051 --prefill-rows 64 --timeout 180)
printf -v head_remote '%q ' "${head_command[@]}"
head_launch_id=$(head_ssh "$head_remote")
[[ "$head_launch_id" =~ ^[0-9a-f]{64}$ ]] || die "head launch did not return an exact container ID"
[[ "$(head_ssh "docker inspect '$head_launch_id' --format '{{index .Config.Labels \"glm.mcdma.run\"}}'")" == "$run_token" ]] \
  || die "head launch label does not match this start"

note "Waiting for both Sparks to load GLM and attach to MCDMA"
deadline=$((SECONDS + 360))
while (( SECONDS < deadline )); do
  head_state=$(three_machine_container_state head)
  worker_state=$(three_machine_container_state worker)
  validate_three_machine_container_state "head Spark" "$head_state"
  validate_three_machine_container_state "worker Spark" "$worker_state"
  [[ -n "$head_state" && -n "$worker_state" ]] || break
  [[ "$(three_machine_container_id "$head_state")" == "$head_launch_id" \
      && "$(three_machine_container_id "$worker_state")" == "$worker_launch_id" ]] \
    || die "a Spark rank identity changed during startup"
  three_machine_container_running "$head_state" && three_machine_container_running "$worker_state" || break
  link=$(adapter peek 2>/dev/null || true)
  if [[ -n "$link" ]] && link_ready "$link"
  then
    note "Starting the persistent OpenAI-compatible endpoint on the Mac Studio"
    api_started=true
    if ! api_launch_identity=$(studio_ssh "set -eu
umask 077
mkdir -p '$THREE_MACHINE_API_STATE_DIR'
pid=
launch_complete=false
cleanup_launch() {
  rc=\$?
  trap - EXIT INT TERM HUP
  if test \"\$launch_complete\" != true && test -n \"\$pid\"; then
    kill -TERM \"\$pid\" 2>/dev/null || true
    remaining=50
    while test \"\$remaining\" -gt 0 && kill -0 \"\$pid\" 2>/dev/null; do
      sleep 0.1
      remaining=\$((remaining - 1))
    done
    if kill -0 \"\$pid\" 2>/dev/null; then kill -KILL \"\$pid\" 2>/dev/null || true; fi
    wait \"\$pid\" 2>/dev/null || true
    if test -r '$THREE_MACHINE_API_STATE_DIR/api.identity' \
        && test \"\$(cut -f1 '$THREE_MACHINE_API_STATE_DIR/api.identity')\" = \"\$pid\"; then
      rm -f '$THREE_MACHINE_API_STATE_DIR/api.identity'
    fi
  fi
  exit \"\$rc\"
}
trap cleanup_launch EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
if nc -z -w 1 '$THREE_MACHINE_API_HOST' '$THREE_MACHINE_API_PORT' >/dev/null 2>&1; then exit 18; fi
rm -f '$THREE_MACHINE_API_STATE_DIR/api.identity'
cd '$THREE_MACHINE_STUDIO_DIR'
deployment_id=\$(cat .deployment-id)
project_manifest=\$(cat .deployment-sha256)
deployment_commit=\$(cat .deployment-commit)
config_sha=\$(shasum -a 256 config.local.env | awk '{print \$1}')
env MCDMA_RPC_LIBRARY='$THREE_MACHINE_MCDMA_LIBRARY' \
  PYTHONPATH='$THREE_MACHINE_STUDIO_DIR:$THREE_MACHINE_TENSORFOLD_SOURCE' \
  THREE_MACHINE_DEPLOYMENT_ID="\$deployment_id" \
  THREE_MACHINE_PROJECT_MANIFEST="\$project_manifest" \
  THREE_MACHINE_COMMIT="\$deployment_commit" \
  THREE_MACHINE_CONFIG_SHA256="\$config_sha" \
  THREE_MACHINE_RUNTIME_IMAGE_ID='$THREE_MACHINE_IMAGE_ID' \
  TF_GLM_EXL3_LOADS=nc TF_GLM_KDA_CHUNKED=1 TF_GLM_L2PF='$THREE_MACHINE_L2PF' \
  THREE_MACHINE_MTP_DRAFTS='$THREE_MACHINE_MTP_DRAFTS' \
  THREE_MACHINE_MTP_CONFIDENCE='$THREE_MACHINE_MTP_CONFIDENCE' \
  THREE_MACHINE_PHASE_TIMING='$THREE_MACHINE_PHASE_TIMING' \
  nohup scripts/studio-api-launcher.sh '$THREE_MACHINE_API_STATE_DIR' '$THREE_MACHINE_STUDIO_PYTHON' \
  experiments/three_machine/api_server.py --model '$THREE_MACHINE_STAGE_MODEL_DIR' \
  --mailbox '$THREE_MACHINE_MAILBOX' --host '$THREE_MACHINE_API_HOST' \
  --port '$THREE_MACHINE_API_PORT' --timeout 180 \
  >'$THREE_MACHINE_API_STATE_DIR/api.log' 2>&1 &
pid=\$!
deadline=100
while test \"\$deadline\" -gt 0 && test ! -s '$THREE_MACHINE_API_STATE_DIR/api.identity' \
    && kill -0 \"\$pid\" 2>/dev/null; do
  sleep 0.05
  deadline=\$((deadline - 1))
done
test -s '$THREE_MACHINE_API_STATE_DIR/api.identity'
test \"\$(cut -f1 '$THREE_MACHINE_API_STATE_DIR/api.identity')\" = \"\$pid\"
cat '$THREE_MACHINE_API_STATE_DIR/api.identity'
launch_complete=true
trap - EXIT INT TERM HUP"); then
      die "Mac API launch failed; the configured port may already be in use"
    fi
    IFS=$'\t' read -r api_launch_pid api_launch_start api_launch_extra <<<"$api_launch_identity"
    [[ "$api_launch_pid" =~ ^[0-9]+$ && -n "$api_launch_start" && -z "$api_launch_extra" ]] \
      || die "Mac API launch returned a malformed identity"
    match_deadline=$((SECONDS + 10))
    api_command_ready=false
    while (( SECONDS < match_deadline )); do
      if studio_api_process_matches "$api_launch_pid" "$api_launch_start"; then
        api_command_ready=true
        break
      else
        match_rc=$?
      fi
      [[ "$match_rc" != 1 ]] || break
      [[ "$match_rc" == 3 ]] || die "Mac API launch identity could not be inspected"
      sleep 0.1
    done
    [[ "$api_command_ready" == true ]] \
      || die "Mac API did not complete its verified launcher-to-Python exec chain"
    api_deadline=$((SECONDS + 360))
    while (( SECONDS < api_deadline )); do
      head_state=$(three_machine_container_state head)
      worker_state=$(three_machine_container_state worker)
      validate_three_machine_container_state "head Spark" "$head_state"
      validate_three_machine_container_state "worker Spark" "$worker_state"
      if ! three_machine_container_running "$head_state" || ! three_machine_container_running "$worker_state"; then
        die "a TensorFold rank exited while the Mac API was loading"
      fi
      [[ "$(three_machine_container_id "$head_state")" == "$head_launch_id" \
          && "$(three_machine_container_id "$worker_state")" == "$worker_launch_id" ]] \
        || die "a Spark rank identity changed while the Mac API was loading"
      link=$(adapter peek)
      link_ready "$link" || die "MCDMA became unhealthy while the Mac API was loading"
      health=$(studio_ssh "curl -fsS --max-time 2 'http://$THREE_MACHINE_API_HOST:$THREE_MACHINE_API_PORT/health'" 2>/dev/null || true)
      if [[ -n "$health" ]] && health_ready "$health"; then
        ready_api_pid=$(verified_studio_api_listener_pid)
        if [[ "$ready_api_pid" == "$api_launch_pid" ]] \
            && three_machine_container_running "$head_state" \
            && three_machine_container_running "$worker_state" \
            && link_ready "$link"; then
          # A rank can cross the initial health check and then fail while its
          # startup work is settling.  Require five consecutive healthy
          # seconds before publishing readiness.
          stable_until=$((SECONDS + 5))
          while (( SECONDS < stable_until )); do
            sleep 1
            head_state=$(three_machine_container_state head)
            worker_state=$(three_machine_container_state worker)
            validate_three_machine_container_state "head Spark" "$head_state"
            validate_three_machine_container_state "worker Spark" "$worker_state"
            three_machine_container_running "$head_state" \
              && three_machine_container_running "$worker_state" \
              || die "a TensorFold rank exited during the startup stability gate"
            [[ "$(three_machine_container_id "$head_state")" == "$head_launch_id" \
                && "$(three_machine_container_id "$worker_state")" == "$worker_launch_id" ]] \
              || die "a Spark rank identity changed during the startup stability gate"
            link=$(adapter peek)
            link_ready "$link" || die "MCDMA became unhealthy during the startup stability gate"
            [[ "$(verified_studio_api_listener_pid)" == "$api_launch_pid" ]] \
              || die "the Mac API changed identity during the startup stability gate"
          done
          start_complete=true
          note "PASS: Mac API/Metal stage, MCDMA, and both TensorFold CUDA ranks are ready"
          exit 0
        fi
      fi
      if [[ "$(probe_studio_api_pid)" != "$api_launch_pid" ]]; then
        studio_ssh "tail -n 80 '$THREE_MACHINE_API_STATE_DIR/api.log'" >&2 || true
        die "Mac API process exited during startup"
      fi
      sleep 1
    done
    studio_ssh "tail -n 80 '$THREE_MACHINE_API_STATE_DIR/api.log'" >&2 || true
    die "Mac API did not become ready"
  fi
  sleep 2
done

head_ssh "docker logs --tail 80 '$THREE_MACHINE_CONTAINER'" >&2 || true
worker_ssh "docker logs --tail 80 '$THREE_MACHINE_CONTAINER'" >&2 || true
die "three-machine TensorFold did not become ready"
