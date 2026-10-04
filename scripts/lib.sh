#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_FILE=${CONFIG_FILE:-$PROJECT_ROOT/config.local.env}
LOCK_FILE=$PROJECT_ROOT/UPSTREAM.lock

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
note() { printf '[glm-mcdma] %s\n' "$*"; }

[[ -f "$CONFIG_FILE" ]] || die "Missing $CONFIG_FILE. Copy config.example.env and fill in this deployment."
[[ -f "$LOCK_FILE" ]] || die "Missing $LOCK_FILE"

set -a
# shellcheck disable=SC1090
source "$LOCK_FILE"
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

THREE_MACHINE_PHASE_TIMING=${THREE_MACHINE_PHASE_TIMING:-0}
THREE_MACHINE_KDA_CHUNKED=${THREE_MACHINE_KDA_CHUNKED:-1}
THREE_MACHINE_MTP_DRAFTS=${THREE_MACHINE_MTP_DRAFTS:-3}
THREE_MACHINE_MTP_CONFIDENCE=${THREE_MACHINE_MTP_CONFIDENCE:-0.35}
THREE_MACHINE_CACHE_BUNDLE_DIR=${THREE_MACHINE_CACHE_BUNDLE_DIR:-${THREE_MACHINE_PROJECT_DIR%/runtime}/cache-bundles}
THREE_MACHINE_CACHE_REPLAY_CONTAINER=${THREE_MACHINE_CACHE_REPLAY_CONTAINER:-glm53-cache-replay}
export THREE_MACHINE_PHASE_TIMING THREE_MACHINE_KDA_CHUNKED THREE_MACHINE_MTP_DRAFTS THREE_MACHINE_MTP_CONFIDENCE
export THREE_MACHINE_CACHE_BUNDLE_DIR THREE_MACHINE_CACHE_REPLAY_CONTAINER

for required in HEAD_SSH WORKER_SSH STUDIO_SSH GLM_RECIPE_DIR MCDMA_STUDIO_DIR GLM_PORT \
  STUDIO_RDMA_DEVICE STUDIO_RDMA_INTERFACE SPARK_RDMA_DEVICE SPARK_RDMA_INTERFACE; do
  [[ -n "${!required:-}" ]] || die "$required is not set"
done

SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=2)

head_ssh() { ssh "${SSH_OPTIONS[@]}" "$HEAD_SSH" "$@"; }
worker_ssh() {
  [[ -n "${WORKER_MANAGEMENT_SSH:-}" ]] || die "WORKER_MANAGEMENT_SSH is not set"
  ssh "${SSH_OPTIONS[@]}" "$WORKER_MANAGEMENT_SSH" "$@"
}
studio_ssh() {
  ssh -S "$HOME/.ssh/cm-studio" -o ControlMaster=no "${SSH_OPTIONS[@]}" "$STUDIO_SSH" "$@"
}

require_three_machine_config() {
  local required
  for required in WORKER_MANAGEMENT_SSH THREE_MACHINE_IMAGE THREE_MACHINE_CONTAINER \
    THREE_MACHINE_PROJECT_DIR THREE_MACHINE_STUDIO_DIR THREE_MACHINE_MODEL_DIR THREE_MACHINE_MODEL_CONTAINER_DIR \
    THREE_MACHINE_HF_CACHE_DIR THREE_MACHINE_TENSORFOLD_CACHE_DIR \
    THREE_MACHINE_STAGE_MODEL_DIR THREE_MACHINE_TENSORFOLD_SOURCE \
    THREE_MACHINE_STUDIO_PYTHON THREE_MACHINE_MCDMA_LIBRARY THREE_MACHINE_SPARK_MCDMA_LIBRARY \
    THREE_MACHINE_STUDIO_MCDMA_DAEMON THREE_MACHINE_SPARK_MCDMA_DAEMON \
    THREE_MACHINE_STUDIO_TO_HEAD_SSH THREE_MACHINE_MCDMA_CONTROL_PORT \
    THREE_MACHINE_STUDIO_RDMA_INDEX THREE_MACHINE_SPARK_RDMA_INDEX \
    THREE_MACHINE_IMAGE_ID THREE_MACHINE_PATCH_LABEL THREE_MACHINE_MODEL_MANIFEST_SHA256 \
    THREE_MACHINE_MODEL_METADATA_SHA256 THREE_MACHINE_TENSORFOLD_SOURCE_MANIFEST_SHA256 \
    TENSORFOLD_MAC_PACKAGE_VERSION \
    THREE_MACHINE_STUDIO_PYTHON_SHA256 THREE_MACHINE_STUDIO_PYTHON_ENV \
    THREE_MACHINE_MCDMA_LIBRARY_SHA256 THREE_MACHINE_SPARK_MCDMA_LIBRARY_SHA256 \
    THREE_MACHINE_STUDIO_MCDMA_DAEMON_SHA256 THREE_MACHINE_SPARK_MCDMA_DAEMON_SHA256 \
    THREE_MACHINE_METAL_LINEAR_SHA256 THREE_MACHINE_METAL_WEIGHTS_SHA256 \
    THREE_MACHINE_STAGE_FILE THREE_MACHINE_STAGE_SHA256 THREE_MACHINE_STAGE_MANIFEST_SHA256 \
    THREE_MACHINE_MAILBOX THREE_MACHINE_NCCL_MASTER THREE_MACHINE_NCCL_PORT \
    THREE_MACHINE_NCCL_INTERFACE THREE_MACHINE_NCCL_HCA THREE_MACHINE_L2PF THREE_MACHINE_MCDMA_STATE \
    THREE_MACHINE_MTP_DRAFTS THREE_MACHINE_MTP_CONFIDENCE \
    THREE_MACHINE_API_HOST THREE_MACHINE_API_PORT THREE_MACHINE_API_STATE_DIR; do
    [[ -n "${!required:-}" ]] || die "$required is not set"
  done
  [[ "$THREE_MACHINE_PHASE_TIMING" == 0 || "$THREE_MACHINE_PHASE_TIMING" == 1 ]] \
    || die "THREE_MACHINE_PHASE_TIMING must be 0 or 1"
  [[ "$THREE_MACHINE_MTP_DRAFTS" =~ ^[1-8]$ ]] \
    || die "THREE_MACHINE_MTP_DRAFTS must be a whole number in 1..8"
  [[ "$THREE_MACHINE_MTP_CONFIDENCE" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] \
    || die "THREE_MACHINE_MTP_CONFIDENCE must be between 0 and 1"
  [[ -z "${THREE_MACHINE_RECOVERY_BUDGET_SECONDS:-}" \
      || "${THREE_MACHINE_RECOVERY_BUDGET_SECONDS:-}" =~ ^[0-9]+$ \
      && "${THREE_MACHINE_RECOVERY_BUDGET_SECONDS:-0}" -gt 0 ]] \
    || die "THREE_MACHINE_RECOVERY_BUDGET_SECONDS must be a positive whole number"
  [[ -z "${THREE_MACHINE_CLEAN_START_SECONDS:-}" \
      || "${THREE_MACHINE_CLEAN_START_SECONDS:-}" =~ ^[0-9]+$ \
      && "${THREE_MACHINE_CLEAN_START_SECONDS:-0}" -gt 0 ]] \
    || die "THREE_MACHINE_CLEAN_START_SECONDS must be a positive whole number"
  [[ "$THREE_MACHINE_IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || die "THREE_MACHINE_IMAGE_ID must be an immutable sha256 image ID"
  [[ "$TENSORFOLD_MAC_PACKAGE_VERSION" =~ ^[0-9]+[.][0-9]+[.][0-9]+$ ]] \
    || die "TENSORFOLD_MAC_PACKAGE_VERSION must be an exact semantic version"
  [[ "$THREE_MACHINE_CONTAINER" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]] \
    || die "THREE_MACHINE_CONTAINER is not a safe Docker container name"
  [[ "$THREE_MACHINE_API_PORT" =~ ^[0-9]+$ && "$THREE_MACHINE_NCCL_PORT" =~ ^[0-9]+$ \
      && "$THREE_MACHINE_MCDMA_CONTROL_PORT" =~ ^[0-9]+$ ]] \
    || die "three-machine ports must be numeric"
  for required in THREE_MACHINE_MODEL_MANIFEST_SHA256 THREE_MACHINE_MODEL_METADATA_SHA256 \
    THREE_MACHINE_TENSORFOLD_SOURCE_MANIFEST_SHA256 THREE_MACHINE_STUDIO_PYTHON_SHA256 \
    THREE_MACHINE_MCDMA_LIBRARY_SHA256 THREE_MACHINE_SPARK_MCDMA_LIBRARY_SHA256 \
    THREE_MACHINE_STUDIO_MCDMA_DAEMON_SHA256 THREE_MACHINE_SPARK_MCDMA_DAEMON_SHA256 \
    THREE_MACHINE_METAL_LINEAR_SHA256 THREE_MACHINE_METAL_WEIGHTS_SHA256 \
    THREE_MACHINE_STAGE_SHA256 THREE_MACHINE_STAGE_MANIFEST_SHA256; do
    [[ "${!required}" =~ ^[0-9a-f]{64}$ ]] || die "$required must be a SHA-256 digest"
  done
}

require_cache_handoff_config() {
  local required
  require_three_machine_config
  for required in THREE_MACHINE_HANDOFF_CONTAINER THREE_MACHINE_PORTABLE_MODEL_DIR \
    THREE_MACHINE_PORTABLE_MODEL_CONTAINER_DIR THREE_MACHINE_PORTABLE_MODEL_STUDIO_DIR \
    THREE_MACHINE_PORTABLE_MODEL_MANIFEST_SHA256 THREE_MACHINE_PORTABLE_MODEL_METADATA_SHA256 \
    THREE_MACHINE_PORTABLE_MODEL_CONTENT_SHA256 THREE_MACHINE_PORTABLE_DOWNLOAD_STAGE \
    THREE_MACHINE_HEAD_DOWNLOAD_STATE THREE_MACHINE_WORKER_DOWNLOAD_STATE \
    THREE_MACHINE_DFLASH_STUDIO_DIR THREE_MACHINE_DFLASH_MANIFEST_SHA256 \
    THREE_MACHINE_HANDOFF_NCCL_PORT \
    THREE_MACHINE_HANDOFF_PREFILL_ROWS THREE_MACHINE_HANDOFF_CHUNK_MIB \
    THREE_MACHINE_MCDMA_REQUEST_MIB THREE_MACHINE_MCDMA_REPLY_MIB THREE_MACHINE_MCDMA_PULL \
    THREE_MACHINE_NATIVE_PORTABLE_CONTAINER \
    THREE_MACHINE_NATIVE_PORTABLE_PORT THREE_MACHINE_NATIVE_PORTABLE_NCCL_PORT; do
    [[ -n "${!required:-}" ]] || die "$required is not set"
  done
  [[ "$THREE_MACHINE_HANDOFF_CONTAINER" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]] \
    || die "THREE_MACHINE_HANDOFF_CONTAINER is not a safe Docker container name"
  [[ "$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]] \
    || die "THREE_MACHINE_NATIVE_PORTABLE_CONTAINER is not a safe Docker container name"
  [[ "$THREE_MACHINE_HANDOFF_NCCL_PORT" =~ ^[0-9]+$ ]] \
    || die "THREE_MACHINE_HANDOFF_NCCL_PORT must be numeric"
  [[ "$THREE_MACHINE_NATIVE_PORTABLE_PORT" =~ ^[0-9]+$ \
      && "$THREE_MACHINE_NATIVE_PORTABLE_NCCL_PORT" =~ ^[0-9]+$ ]] \
    || die "portable native service ports must be numeric"
  [[ "$THREE_MACHINE_HANDOFF_PREFILL_ROWS" =~ ^[1-9][0-9]*$ ]] \
    || die "THREE_MACHINE_HANDOFF_PREFILL_ROWS must be positive"
  [[ "$THREE_MACHINE_KDA_CHUNKED" == 0 || "$THREE_MACHINE_KDA_CHUNKED" == 1 ]] \
    || die "THREE_MACHINE_KDA_CHUNKED must be 0 or 1"
  [[ "$THREE_MACHINE_HANDOFF_CHUNK_MIB" =~ ^[0-9]+$ \
      && "$THREE_MACHINE_HANDOFF_CHUNK_MIB" -ge 1 \
      && "$THREE_MACHINE_HANDOFF_CHUNK_MIB" -le 256 ]] \
    || die "THREE_MACHINE_HANDOFF_CHUNK_MIB must be a whole number in 1..256"
  [[ "$THREE_MACHINE_MCDMA_REQUEST_MIB" =~ ^[0-9]+$ \
      && "$THREE_MACHINE_MCDMA_REPLY_MIB" =~ ^[0-9]+$ \
      && "$THREE_MACHINE_MCDMA_REQUEST_MIB" -ge 4 \
      && "$THREE_MACHINE_MCDMA_REPLY_MIB" -ge 4 \
      && "$THREE_MACHINE_MCDMA_REQUEST_MIB" -le 256 \
      && "$THREE_MACHINE_MCDMA_REPLY_MIB" -le 256 \
      && $((THREE_MACHINE_MCDMA_REQUEST_MIB % 4)) -eq 0 \
      && $((THREE_MACHINE_MCDMA_REPLY_MIB % 4)) -eq 0 ]] \
    || die "MCDMA mailbox halves must be multiples of 4 MiB in 4..256"
  [[ "$THREE_MACHINE_MCDMA_PULL" == 0 || "$THREE_MACHINE_MCDMA_PULL" == 1 ]] \
    || die "THREE_MACHINE_MCDMA_PULL must be 0 or 1"
  for required in THREE_MACHINE_PORTABLE_MODEL_MANIFEST_SHA256 \
    THREE_MACHINE_PORTABLE_MODEL_METADATA_SHA256 THREE_MACHINE_PORTABLE_MODEL_CONTENT_SHA256 \
    THREE_MACHINE_DFLASH_MANIFEST_SHA256; do
    [[ "${!required}" =~ ^[0-9a-f]{64}$ ]] || die "$required must be a SHA-256 digest"
  done
}

cache_handoff_container_state() {
  local role=${1:?role required} quoted_container command
  local format='{{.Name}}|{{.Id}}|{{.Image}}|{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}|{{index .Config.Labels "glm.mcdma.mode"}}'
  printf -v quoted_container '%q' "$THREE_MACHINE_HANDOFF_CONTAINER"
  command="set -eu
docker info >/dev/null
name=\$(docker ps -a --filter name=^/$quoted_container\$ --format '{{.Names}}')
if test -z \"\$name\"; then exit 0; fi
test \"\$name\" = $quoted_container
docker inspect $quoted_container --format '$format'"
  case "$role" in
    head) head_ssh "$command" ;;
    worker) worker_ssh "$command" ;;
    *) die "unknown Spark role: $role" ;;
  esac
}

validate_cache_handoff_container_state() {
  local role=${1:?role required} state=${2:-}
  local name id image _running _oom _exit mode
  [[ -n "$state" ]] || return 0
  IFS='|' read -r name id image _running _oom _exit mode <<<"$state"
  [[ "$name" == "/$THREE_MACHINE_HANDOFF_CONTAINER" ]] \
    || die "$role container identity is not $THREE_MACHINE_HANDOFF_CONTAINER"
  [[ "$id" =~ ^[0-9a-f]{64}$ ]] || die "$role cache-handoff container ID is malformed"
  [[ "$image" == "$THREE_MACHINE_IMAGE_ID" ]] \
    || die "$role cache-handoff container image is not the pinned TensorFold image"
  [[ "$mode" == cache-handoff ]] \
    || die "$role container is not an owned cache-handoff process"
}

cache_handoff_container_running() {
  local state=${1:-} _name _id _image running _oom _exit _mode
  [[ -n "$state" ]] || return 1
  IFS='|' read -r _name _id _image running _oom _exit _mode <<<"$state"
  [[ "$running" == true ]]
}

cache_handoff_container_id() {
  local state=${1:?container state required} _name id _image _running _oom _exit _mode
  IFS='|' read -r _name id _image _running _oom _exit _mode <<<"$state"
  [[ "$id" =~ ^[0-9a-f]{64}$ ]] || die "cache-handoff container ID is malformed"
  printf '%s\n' "$id"
}

three_machine_recovery_budget_seconds() {
  local clean_start=${THREE_MACHINE_CLEAN_START_SECONDS:-}
  if [[ -n "${THREE_MACHINE_RECOVERY_BUDGET_SECONDS:-}" ]]; then
    printf '%s\n' "$THREE_MACHINE_RECOVERY_BUDGET_SECONDS"
  elif [[ -n "$clean_start" ]]; then
    # The measured clean-start allowance plus 50% and three minutes for
    # detection, owned shutdown, cleanup, and the exact post-restart canary.
    printf '%s\n' "$(( (clean_start * 3 + 1) / 2 + 180 ))"
  else
    # Safe first-run allowance until a clean-start measurement is supplied.
    printf '600\n'
  fi
}

adapter() {
  P06_LIVE_STATE="$PROJECT_ROOT/$THREE_MACHINE_MCDMA_STATE" \
    P06_LIVE_SCENARIO=fresh \
    P06_LIVE_STUDIO_DAEMON="$THREE_MACHINE_STUDIO_MCDMA_DAEMON" \
    P06_LIVE_SPARK_DAEMON="$THREE_MACHINE_SPARK_MCDMA_DAEMON" \
    P06_LIVE_STUDIO_HOST="$STUDIO_SSH" \
    P06_LIVE_SPARK_HOST="$HEAD_SSH" \
    P06_LIVE_TUNNEL_HOST="$THREE_MACHINE_STUDIO_TO_HEAD_SSH" \
    P06_LIVE_CONTROL_PORT="$THREE_MACHINE_MCDMA_CONTROL_PORT" \
    P06_LIVE_STUDIO_DEVICE="$STUDIO_RDMA_DEVICE" \
    P06_LIVE_SPARK_DEVICE="$SPARK_RDMA_DEVICE" \
    P06_LIVE_STUDIO_DEVICE_INDEX="$THREE_MACHINE_STUDIO_RDMA_INDEX" \
    P06_LIVE_SPARK_DEVICE_INDEX="$THREE_MACHINE_SPARK_RDMA_INDEX" \
    P06_LIVE_REQUEST_MIB="$THREE_MACHINE_MCDMA_REQUEST_MIB" \
    P06_LIVE_REPLY_MIB="$THREE_MACHINE_MCDMA_REPLY_MIB" \
    P06_LIVE_PULL="$THREE_MACHINE_MCDMA_PULL" \
    python3 "$PROJECT_ROOT/scripts/tests/p06-live-adapter.py" "$@"
}

three_machine_container_state() {
  local role=${1:?role required} quoted_container command
  local format='{{.Name}}|{{.Id}}|{{.Image}}|{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}'
  printf -v quoted_container '%q' "$THREE_MACHINE_CONTAINER"
  command="set -eu
docker info >/dev/null
name=\$(docker ps -a --filter name=^/$quoted_container\$ --format '{{.Names}}')
if test -z \"\$name\"; then exit 0; fi
test \"\$name\" = $quoted_container
docker inspect $quoted_container --format '$format'"
  case "$role" in
    head)
      head_ssh "$command"
      ;;
    worker)
      worker_ssh "$command"
      ;;
    *) die "unknown Spark role: $role" ;;
  esac
}

lifecycle_lock_owner_is_stale_local() {
  local owner=${1:?owner token required} owner_host owner_pid owner_start_hash owner_stamp extra
  local actual_start actual_hash
  IFS=: read -r owner_host owner_pid owner_start_hash owner_stamp extra <<<"$owner"
  [[ -z "$extra" && "$owner_host" == "$(hostname)" && "$owner_pid" =~ ^[0-9]+$ ]] || return 1
  if ! actual_start=$(LC_ALL=C TZ=UTC ps -p "$owner_pid" -o lstart= 2>/dev/null) || [[ -z "$actual_start" ]]; then
    return 0
  fi
  # Older lock tokens did not include a start hash.  A live PID from that
  # format cannot be broken safely; a dead one was handled above.
  [[ "$owner_start_hash" =~ ^[0-9a-f]{64}$ && -n "$owner_stamp" ]] || return 1
  actual_hash=$(printf '%s' "$actual_start" | shasum -a 256 | awk '{print $1}')
  [[ "$actual_hash" != "$owner_start_hash" ]]
}

acquire_three_machine_lifecycle_lock() {
  local lock_dir owner_file token quoted_lock quoted_owner quoted_token acquire_rc
  local existing_owner local_start local_start_hash quoted_existing
  lock_dir="$THREE_MACHINE_API_STATE_DIR/lifecycle.lock"
  owner_file="$lock_dir/owner"
  printf -v quoted_lock '%q' "$lock_dir"
  printf -v quoted_owner '%q' "$owner_file"
  if [[ -n "${THREE_MACHINE_LIFECYCLE_LOCK_TOKEN:-}" ]]; then
    printf -v quoted_token '%q' "$THREE_MACHINE_LIFECYCLE_LOCK_TOKEN"
    studio_ssh "test -r $quoted_owner && test \"\$(cat $quoted_owner)\" = $quoted_token" \
      || die "the shared three-machine lifecycle lock changed owner"
    THREE_MACHINE_LIFECYCLE_LOCK_OWNED=false
    return 0
  fi
  if ! existing_owner=$(studio_ssh "if test -r $quoted_owner; then cat $quoted_owner; elif test -d $quoted_lock; then printf '__OWNER_MISSING__\\n'; fi"); then
    die "could not inspect the shared three-machine lifecycle lock"
  fi
  if [[ "$existing_owner" == __OWNER_MISSING__ ]]; then
    # Allow an interrupted mkdir/write pair to finish before considering its
    # ownerless directory stale.  The one-minute age test is performed again
    # atomically with the removal conditions on the Studio.
    sleep 2
    if studio_ssh "test -d $quoted_lock && test ! -e $quoted_owner && test -n \"\$(find $quoted_lock -prune -mmin +1 -print)\" && rmdir $quoted_lock"; then
      note "Recovered an abandoned ownerless lifecycle lock"
    else
      die "another lifecycle operation is acquiring the shared lock"
    fi
  elif [[ -n "$existing_owner" ]]; then
    if lifecycle_lock_owner_is_stale_local "$existing_owner"; then
      printf -v quoted_existing '%q' "$existing_owner"
      studio_ssh "test -r $quoted_owner && test \"\$(cat $quoted_owner)\" = $quoted_existing && rm -f $quoted_owner && rmdir $quoted_lock" \
        || die "the stale lifecycle lock changed while it was being recovered"
      note "Recovered an abandoned lifecycle lock from a dead local controller process"
    else
      die "another three-machine lifecycle operation is already running"
    fi
  fi

  local_start=$(LC_ALL=C TZ=UTC ps -p $$ -o lstart=) \
    || die "could not identify the local lifecycle controller process"
  local_start_hash=$(printf '%s' "$local_start" | shasum -a 256 | awk '{print $1}')
  [[ "$local_start_hash" =~ ^[0-9a-f]{64}$ ]] \
    || die "could not hash the local lifecycle controller identity"
  token="$(hostname):$$:$local_start_hash:$(date -u +%Y%m%dT%H%M%SZ)"
  printf -v quoted_token '%q' "$token"
  export THREE_MACHINE_LIFECYCLE_LOCK_TOKEN="$token"
  # Mark acquisition as pending before the remote round trip. If this shell is
  # interrupted after the remote mkdir but before SSH reports success, the
  # already-installed EXIT trap can still remove only this exact token.
  THREE_MACHINE_LIFECYCLE_LOCK_OWNED=pending
  if studio_ssh "umask 077; mkdir -p '$(dirname "$lock_dir")'; if mkdir $quoted_lock; then if printf '%s\\n' $quoted_token >$quoted_owner; then exit 0; else rm -f $quoted_owner; rmdir $quoted_lock 2>/dev/null || true; exit 74; fi; else printf 'lifecycle lock is already held by '; cat $quoted_owner 2>/dev/null || printf 'unknown'; exit 73; fi"; then
    THREE_MACHINE_LIFECYCLE_LOCK_OWNED=true
    return 0
  else
    acquire_rc=$?
  fi
  if (( acquire_rc == 73 )); then
    THREE_MACHINE_LIFECYCLE_LOCK_OWNED=false
    unset THREE_MACHINE_LIFECYCLE_LOCK_TOKEN
    die "another three-machine lifecycle operation is already running"
  fi
  # SSH can lose the success response after the Studio created the lock. Probe
  # for our token before concluding that another lifecycle action owns it.
  if studio_ssh "test -r $quoted_owner && test \"\$(cat $quoted_owner)\" = $quoted_token"; then
    THREE_MACHINE_LIFECYCLE_LOCK_OWNED=true
    note "Recovered ownership after an interrupted lifecycle-lock response"
    return 0
  fi
  # Keep the pending token until the EXIT trap runs. The transport failed, so
  # absence, another owner, and a lost success response cannot be distinguished.
  die "lifecycle-lock ownership could not be confirmed; token-checked cleanup will run"
}

release_three_machine_lifecycle_lock() {
  local lock_dir owner_file quoted_lock quoted_owner quoted_token attempt released=false
  [[ "${THREE_MACHINE_LIFECYCLE_LOCK_OWNED:-false}" == true \
      || "${THREE_MACHINE_LIFECYCLE_LOCK_OWNED:-false}" == pending ]] || return 0
  lock_dir="$THREE_MACHINE_API_STATE_DIR/lifecycle.lock"
  owner_file="$lock_dir/owner"
  printf -v quoted_lock '%q' "$lock_dir"
  printf -v quoted_owner '%q' "$owner_file"
  printf -v quoted_token '%q' "$THREE_MACHINE_LIFECYCLE_LOCK_TOKEN"
  for attempt in 1 2 3; do
    if studio_ssh "test -r $quoted_owner && test \"\$(cat $quoted_owner)\" = $quoted_token && rm -f $quoted_owner && rmdir $quoted_lock"; then
      released=true
      break
    fi
    sleep 1
  done
  if [[ "$released" == true ]]; then
    THREE_MACHINE_LIFECYCLE_LOCK_OWNED=false
    unset THREE_MACHINE_LIFECYCLE_LOCK_TOKEN
  else
    # Preserve the token locally until process exit.  A later operation on the
    # same controller can prove this PID/start identity is dead and recover it.
    note "WARNING: could not release the shared three-machine lifecycle lock after three attempts"
  fi
}

validate_three_machine_container_state() {
  local role=${1:?role required} state=${2:-}
  local name id image _running _oom _exit
  [[ -n "$state" ]] || return 0
  IFS='|' read -r name id image _running _oom _exit <<<"$state"
  [[ "$name" == "/$THREE_MACHINE_CONTAINER" ]] || die "$role container identity is not $THREE_MACHINE_CONTAINER"
  [[ "$id" =~ ^[0-9a-f]{64}$ ]] || die "$role container ID is malformed"
  [[ "$image" == "$THREE_MACHINE_IMAGE_ID" ]] || die "$role container image is not the pinned three-machine image"
}

three_machine_container_running() {
  local state=${1:-} _name _id _image running _oom _exit
  [[ -n "$state" ]] || return 1
  IFS='|' read -r _name _id _image running _oom _exit <<<"$state"
  [[ "$running" == true ]]
}

three_machine_container_id() {
  local state=${1:?container state required} _name id _image _running _oom _exit
  IFS='|' read -r _name id _image _running _oom _exit <<<"$state"
  [[ "$id" =~ ^[0-9a-f]{64}$ ]] || die "container ID is malformed"
  printf '%s\n' "$id"
}

studio_api_identity_file_value() {
  local identity pid started extra
  if ! identity=$(studio_ssh "if test -r '$THREE_MACHINE_API_STATE_DIR/api.identity'; then cat '$THREE_MACHINE_API_STATE_DIR/api.identity'; fi"); then
    die "could not inspect the Mac API identity file"
  fi
  [[ -n "$identity" ]] || return 0
  [[ "$identity" != *$'\n'* ]] || die "Mac API identity file has multiple records"
  IFS=$'\t' read -r pid started extra <<<"$identity"
  [[ "$pid" =~ ^[0-9]+$ && -n "$started" && -z "$extra" ]] \
    || die "Mac API identity file is malformed"
  printf '%s\t%s\n' "$pid" "$started"
}

studio_api_pid_file_value() {
  local identity pid _started
  if ! identity=$(studio_api_identity_file_value); then
    return 2
  fi
  [[ -n "$identity" ]] || return 0
  IFS=$'\t' read -r pid _started <<<"$identity"
  printf '%s\n' "$pid"
}

studio_api_start_file_value() {
  local identity _pid started
  if ! identity=$(studio_api_identity_file_value); then
    return 2
  fi
  [[ -n "$identity" ]] || return 0
  IFS=$'\t' read -r _pid started <<<"$identity"
  printf '%s\n' "$started"
}

studio_api_command_matches() {
  local command=${1:-} process_python=${2:-$THREE_MACHINE_STUDIO_PYTHON}
  python3 "$PROJECT_ROOT/scripts/api-command-match.py" \
    "$command" "$process_python" "$THREE_MACHINE_STAGE_MODEL_DIR" \
    "$THREE_MACHINE_MAILBOX" "$THREE_MACHINE_API_HOST" "$THREE_MACHINE_API_PORT"
}

studio_api_process_matches() {
  local pid=${1:?pid required} expected_start=${2:?start identity required}
  local command actual_start process_python
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  if ! command=$(studio_ssh "ps -p '$pid' -o command= 2>/dev/null || true"); then
    return 2
  fi
  [[ -n "$command" ]] || return 1
  if ! actual_start=$(studio_ssh "LC_ALL=C TZ=UTC ps -p '$pid' -o lstart= 2>/dev/null || true"); then
    return 2
  fi
  [[ "$actual_start" == "$expected_start" ]] || return 1
  if ! process_python=$(studio_ssh "'$THREE_MACHINE_STUDIO_PYTHON' -c 'import os,sys; print(os.path.join(os.path.realpath(sys.base_prefix), \"Resources/Python.app/Contents/MacOS/Python\"))'"); then
    return 2
  fi
  if studio_api_command_matches "$command" "$process_python"; then
    return 0
  fi
  # PID and exact process start still identify the launcher-owned process.
  # A changed command/config is drift, not permission to orphan that process.
  return 3
}

studio_process_start_matches() {
  local pid=${1:?pid required} expected_start=${2:?start identity required} actual_start
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  if ! actual_start=$(studio_ssh "LC_ALL=C TZ=UTC ps -p '$pid' -o lstart= 2>/dev/null || true"); then
    return 2
  fi
  [[ -n "$actual_start" && "$actual_start" == "$expected_start" ]]
}

clear_studio_api_identity() {
  local pid=${1:?pid required} started=${2:?start identity required} expected quoted_expected
  expected=$(printf '%s\t%s' "$pid" "$started")
  printf -v quoted_expected '%q' "$expected"
  studio_ssh "if test -r '$THREE_MACHINE_API_STATE_DIR/api.identity' && test \"\$(cat '$THREE_MACHINE_API_STATE_DIR/api.identity')\" = $quoted_expected; then rm -f '$THREE_MACHINE_API_STATE_DIR/api.identity'; fi"
}

clear_owned_socket_after_dead_pid() {
  local remote=${1:?remote function required} pid=${2:?owner pid required}
  local path=${3:?socket path required} quoted_path output rc
  [[ "$remote" == studio_ssh || "$remote" == head_ssh ]] \
    || die "unknown remote for owned socket cleanup"
  [[ "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 ]] \
    || die "owned socket cleanup requires an exact positive owner PID"
  printf -v quoted_path '%q' "$path"
  set +e
  output=$("$remote" "set -eu
if kill -0 '$pid' 2>/dev/null; then exit 73; fi
# A failed kill probe can mean permission denied; independently inspect the
# process table before treating the recorded owner as dead.
command -v ps >/dev/null || exit 77
processes=\$(ps -e -o pid=) || exit 77
if printf '%s\\n' \"\$processes\" | awk -v owner='$pid' '\$1 == owner { found=1 } END { exit !found }'; then exit 73; fi
if test ! -e $quoted_path; then exit 0; fi
test ! -L $quoted_path || exit 74
test -S $quoted_path || exit 74
test -O $quoted_path || exit 75
command -v lsof >/dev/null || exit 77
owners=\$(lsof -nP -t -- $quoted_path 2>/dev/null || true)
test -z \"\$owners\" || exit 76
rm -f $quoted_path
printf 'cleared\\n'")
  rc=$?
  set -e
  case "$rc" in
    0) ;;
    73) die "refusing to clear $path while its exact owner PID $pid is alive" ;;
    74) die "refusing to clear $path because it is not a socket" ;;
    75) die "refusing to clear $path because the remote account does not own it" ;;
    76) die "refusing to clear $path because another process has it open" ;;
    77) die "could not prove the socket owner is dead with the available inspection tools" ;;
    *) die "could not safely inspect owned socket $path" ;;
  esac
  [[ -z "$output" || "$output" == cleared ]] \
    || die "owned socket cleanup returned unexpected output"
  [[ "$output" != cleared ]] || printf '%s\t%s\t%s\n' "$remote" "$pid" "$path"
}

clear_owned_mcdma_socket_paths() {
  local state=${1:?MCDMA state JSON required} pids mac_pid spark_pid
  if ! pids=$(python3 - "$state" <<'PY'
import json
import sys

state = json.loads(sys.argv[1])
values = []
for side in ("mac", "spark"):
    pid = state.get(side, {}).get("pid")
    if type(pid) is not int or pid <= 1:
        raise SystemExit(f"MCDMA {side} state lacks an exact owner PID")
    values.append(str(pid))
print("\t".join(values))
PY
  ); then
    die "could not prove the owned MCDMA socket PIDs"
  fi
  IFS=$'\t' read -r mac_pid spark_pid <<<"$pids"
  clear_owned_socket_after_dead_pid studio_ssh "$mac_pid" /tmp/mcdma-rpcd.sock
  clear_owned_socket_after_dead_pid head_ssh "$spark_pid" \
    "/tmp/mcdma-rpcd.$THREE_MACHINE_MAILBOX.sock"
}

studio_process_state() {
  local pid=${1:?pid required} state
  [[ "$pid" =~ ^[0-9]+$ ]] || return 2
  if ! state=$(studio_ssh "if kill -0 '$pid' 2>/dev/null; then printf 'alive\\n'; else printf 'dead\\n'; fi"); then
    return 2
  fi
  [[ "$state" == alive || "$state" == dead ]] || return 2
  printf '%s\n' "$state"
}

probe_studio_api_pid() {
  local pid expected_start match_rc
  if ! pid=$(studio_api_pid_file_value); then
    die "could not read the Mac API identity on the Studio"
  fi
  [[ -n "$pid" ]] || return 0
  if ! expected_start=$(studio_api_start_file_value); then
    die "could not read the Mac API start identity on the Studio"
  fi
  [[ -n "$expected_start" ]] || return 0
  if studio_api_process_matches "$pid" "$expected_start"; then
    printf '%s\n' "$pid"
    return 0
  else
    match_rc=$?
  fi
  case "$match_rc" in
    1) return 0 ;;
    3)
      printf 'WARNING: the recorded Mac API command/config drifted; exact PID/start ownership still holds\n' >&2
      printf '%s\n' "$pid"
      ;;
    *) die "could not verify the Mac API process on the Studio" ;;
  esac
}

verified_studio_api_pid() {
  local pid expected_start match_rc
  if ! pid=$(studio_api_pid_file_value); then
    die "could not read the Mac API identity on the Studio"
  fi
  [[ -n "$pid" ]] || return 0
  if ! expected_start=$(studio_api_start_file_value); then
    die "could not read the Mac API start identity on the Studio"
  fi
  [[ -n "$expected_start" ]] || die "Mac API identity file is missing its start time"
  if studio_api_process_matches "$pid" "$expected_start"; then
    :
  else
    match_rc=$?
    case "$match_rc" in
      1) die "Mac API pid identity changed; refusing to signal it" ;;
      3)
        printf 'WARNING: Mac API command/config drifted; using exact PID/start ownership\n' >&2
        ;;
      *) die "could not verify the Mac API process on the Studio" ;;
    esac
  fi
  printf '%s\n' "$pid"
}

studio_api_port_open() {
  local state
  if ! state=$(studio_ssh "if nc -z -w 1 '$THREE_MACHINE_API_HOST' '$THREE_MACHINE_API_PORT' >/dev/null 2>&1; then printf 'open\\n'; else printf 'closed\\n'; fi"); then
    die "could not inspect the Mac API listener"
  fi
  [[ "$state" == open ]]
}

verified_studio_api_listener_pid() {
  local pid listeners
  pid=$(verified_studio_api_pid)
  [[ -n "$pid" ]] || return 0
  if ! listeners=$(studio_ssh "lsof -nP -tiTCP@'$THREE_MACHINE_API_HOST':'$THREE_MACHINE_API_PORT' -sTCP:LISTEN 2>/dev/null || true"); then
    die "could not inspect the Mac API listener owner"
  fi
  [[ "$listeners" == "$pid" ]] || die "the verified Mac API pid does not exclusively own its configured listener"
  printf '%s\n' "$pid"
}

validate_mcdma_cleanup() {
  local cleanup=${1:?cleanup JSON required}
  python3 - "$cleanup" <<'PY'
import json, sys
s = json.loads(sys.argv[1])
if not s.get("clean") or s.get("children_alive") or s.get("problems"):
    raise SystemExit(f"MCDMA cleanup did not reach a clean state: {s}")
PY
}

mcdma_link_busy() {
  local state=${1:?MCDMA state JSON required}
  python3 - "$state" <<'PY'
import json
import sys

raise SystemExit(0 if json.loads(sys.argv[1]).get("inflight") is True else 1)
PY
}

mcdma_transport_healthy() {
  local state=${1:?MCDMA state JSON required}
  python3 - "$state" <<'PY'
import json
import sys

s = json.loads(sys.argv[1])
raise SystemExit(0 if (
    s.get("link") == "up"
    and s.get("mac", {}).get("alive")
    and s.get("spark", {}).get("alive")
    and s.get("mac", {}).get("failures") == 0
    and s.get("spark", {}).get("failures") == 0
) else 1)
PY
}

mcdma_link_healthy() {
  local state=${1:?MCDMA state JSON required}
  mcdma_transport_healthy "$state" || return 1
  python3 - "$state" <<'PY'
import json
import sys

raise SystemExit(
    0 if json.loads(sys.argv[1]).get("spark", {}).get("service") == "poll" else 1
)
PY
}
