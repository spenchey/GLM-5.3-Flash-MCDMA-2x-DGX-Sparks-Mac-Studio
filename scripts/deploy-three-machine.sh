#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_three_machine_config

trap release_three_machine_lifecycle_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
acquire_three_machine_lifecycle_lock

head_state=$(three_machine_container_state head)
worker_state=$(three_machine_container_state worker)
validate_three_machine_container_state "head Spark" "$head_state"
validate_three_machine_container_state "worker Spark" "$worker_state"
! three_machine_container_running "$head_state" \
  || die "stop the running three-machine service before deploying new code"
! three_machine_container_running "$worker_state" \
  || die "stop the running three-machine service before deploying new code"
[[ -z "$(probe_studio_api_pid)" ]] \
  || die "stop the running Mac API before deploying new code"
if studio_api_port_open; then
  die "the configured API port is still in use; refusing to replace live code"
fi

[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] \
  || die "commit and clean the reviewed project before deploying it"
deploy_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
deploy_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
archive_file=$(mktemp "${TMPDIR:-/tmp}/glm53-three-machine.XXXXXXXX.tar")
cleanup_deploy_temp() {
  local rc=$?
  trap - EXIT
  release_three_machine_lifecycle_lock
  rm -f "$archive_file"
  exit "$rc"
}
trap cleanup_deploy_temp EXIT
git -C "$PROJECT_ROOT" archive --format=tar HEAD -- \
  UPSTREAM.lock config.example.env experiments/three_machine patches/tensorfold scripts \
  >"$archive_file"
# The private deployment configuration is transported only inside the
# mode-0600 temporary archive and remains ignored by both Git and the runtime
# content manifest.
tar -rf "$archive_file" -C "$PROJECT_ROOT" config.local.env
chmod 600 "$archive_file"
archive_sha=$(shasum -a 256 "$archive_file" | awk '{print $1}')
project_manifest=$(python3 "$PROJECT_ROOT/scripts/project-manifest.py" "$PROJECT_ROOT")
attempted_head=false
attempted_worker=false
attempted_studio=false
deployment_complete=false

validate_artifact_separation() {
  local role=$1 destination=$2
  shift 2
  python3 - "$role" "$destination" "$@" <<'PY'
from pathlib import Path
import sys

role, destination_raw, *artifacts = sys.argv[1:]
destination = Path(destination_raw)
if not destination.is_absolute() or destination == Path('/'):
    raise SystemExit(f"{role} deployment path must be an absolute non-root directory")
destination = destination.resolve(strict=False)
for raw in artifacts:
    artifact = Path(raw).resolve(strict=False)
    if artifact == destination or destination in artifact.parents:
        raise SystemExit(
            f"{role} deployment path contains a prepared artifact or state path: {artifact}"
        )
PY
}

validate_artifact_separation head "$THREE_MACHINE_PROJECT_DIR" \
  "$THREE_MACHINE_MODEL_DIR" "$THREE_MACHINE_HF_CACHE_DIR" \
  "$THREE_MACHINE_TENSORFOLD_CACHE_DIR" "$THREE_MACHINE_SPARK_MCDMA_LIBRARY" \
  "$THREE_MACHINE_SPARK_MCDMA_DAEMON"
validate_artifact_separation worker "$THREE_MACHINE_PROJECT_DIR" \
  "$THREE_MACHINE_MODEL_DIR" "$THREE_MACHINE_HF_CACHE_DIR" \
  "$THREE_MACHINE_TENSORFOLD_CACHE_DIR" "$THREE_MACHINE_SPARK_MCDMA_LIBRARY" \
  "$THREE_MACHINE_SPARK_MCDMA_DAEMON"
validate_artifact_separation studio "$THREE_MACHINE_STUDIO_DIR" \
  "$THREE_MACHINE_TENSORFOLD_SOURCE" "$THREE_MACHINE_STAGE_MODEL_DIR" \
  "$THREE_MACHINE_STUDIO_PYTHON" "$THREE_MACHINE_MCDMA_LIBRARY" \
  "$THREE_MACHINE_STUDIO_MCDMA_DAEMON" "$THREE_MACHINE_API_STATE_DIR"

deploy_remote() {
  local role=$1 destination=$2 quoted_destination quoted_incoming quoted_manifest quoted_deploy quoted_commit remote
  [[ "$destination" == /* && "$destination" != / ]] \
    || die "$role deployment path is not a safe absolute directory"
  printf -v quoted_destination '%q' "$destination"
  printf -v quoted_incoming '%q' "${destination}.incoming.${deploy_id}"
  printf -v quoted_manifest '%q' "$project_manifest"
  printf -v quoted_deploy '%q' "$deploy_id"
  printf -v quoted_commit '%q' "$deploy_commit"
  remote="set -eu
destination=$quoted_destination
incoming=$quoted_incoming
test ! -e \"\$incoming\" && test ! -L \"\$incoming\"
mkdir -p \"\$incoming\"
cleanup() { code=\$?; if test \$code -ne 0 && test -d \"\$incoming\" && test ! -L \"\$incoming\"; then rm -rf -- \"\$incoming\"; fi; exit \$code; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
tar -xf - -C \"\$incoming\"
test -r \"\$incoming/experiments/three_machine/api_server.py\"
test -r \"\$incoming/scripts/start-three-machine.sh\"
test -x \"\$incoming/scripts/deployment-state-machine.sh\"
test -r \"\$incoming/config.local.env\"
chmod 600 \"\$incoming/config.local.env\"
actual_manifest=\$(python3 \"\$incoming/scripts/project-manifest.py\" \"\$incoming\")
test \"\$actual_manifest\" = $quoted_manifest
printf '%s\\n' $quoted_manifest >\"\$incoming/.deployment-sha256\"
printf '%s\\n' $quoted_deploy >\"\$incoming/.deployment-id\"
printf '%s\\n' $quoted_commit >\"\$incoming/.deployment-commit\"
\"\$incoming/scripts/deployment-state-machine.sh\" install \"\$destination\" \"\$incoming\" $quoted_manifest $quoted_deploy"
  case "$role" in
    head) cat "$archive_file" | head_ssh "$remote" ;;
    worker) cat "$archive_file" | worker_ssh "$remote" ;;
    studio) cat "$archive_file" | studio_ssh "$remote" ;;
    *) die "unknown deployment role: $role" ;;
  esac
}

deployment_state() {
  local role=$1 destination=$2
  case "$role" in
    head) head_ssh "if test -r '$destination/.deployment-sha256' && test -r '$destination/.deployment-id' && test -r '$destination/.deployment-commit'; then printf '%s\\t%s\\t%s\\n' \"\$(cat '$destination/.deployment-sha256')\" \"\$(cat '$destination/.deployment-id')\" \"\$(cat '$destination/.deployment-commit')\"; fi" ;;
    worker) worker_ssh "if test -r '$destination/.deployment-sha256' && test -r '$destination/.deployment-id' && test -r '$destination/.deployment-commit'; then printf '%s\\t%s\\t%s\\n' \"\$(cat '$destination/.deployment-sha256')\" \"\$(cat '$destination/.deployment-id')\" \"\$(cat '$destination/.deployment-commit')\"; fi" ;;
    studio) studio_ssh "if test -r '$destination/.deployment-sha256' && test -r '$destination/.deployment-id' && test -r '$destination/.deployment-commit'; then printf '%s\\t%s\\t%s\\n' \"\$(cat '$destination/.deployment-sha256')\" \"\$(cat '$destination/.deployment-id')\" \"\$(cat '$destination/.deployment-commit')\"; fi" ;;
    *) die "unknown deployment role: $role" ;;
  esac
}

rollback_remote() {
  local role=$1 destination=$2 quoted_destination quoted_manifest quoted_deploy command
  [[ "$destination" == /* && "$destination" != / ]] \
    || return 1
  printf -v quoted_destination '%q' "$destination"
  printf -v quoted_manifest '%q' "$project_manifest"
  printf -v quoted_deploy '%q' "$deploy_id"
  command="bash -s -- rollback $quoted_destination $quoted_manifest $quoted_deploy"
  case "$role" in
    head) cat "$PROJECT_ROOT/scripts/deployment-state-machine.sh" | head_ssh "$command" ;;
    worker) cat "$PROJECT_ROOT/scripts/deployment-state-machine.sh" | worker_ssh "$command" ;;
    studio) cat "$PROJECT_ROOT/scripts/deployment-state-machine.sh" | studio_ssh "$command" ;;
    *) return 1 ;;
  esac
}

finish_deploy() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if (( rc != 0 )) && [[ "$deployment_complete" != true ]]; then
    note "Deployment did not complete; restoring every host that was attempted"
    [[ "$attempted_studio" != true ]] \
      || rollback_remote studio "$THREE_MACHINE_STUDIO_DIR" \
      || note "WARNING: Studio rollback needs manual attention"
    [[ "$attempted_worker" != true ]] \
      || rollback_remote worker "$THREE_MACHINE_PROJECT_DIR" \
      || note "WARNING: worker rollback needs manual attention"
    [[ "$attempted_head" != true ]] \
      || rollback_remote head "$THREE_MACHINE_PROJECT_DIR" \
      || note "WARNING: head rollback needs manual attention"
  fi
  release_three_machine_lifecycle_lock
  rm -f "$archive_file"
  exit "$rc"
}
deploy_interrupted() {
  local exit_code=$1
  trap - INT TERM HUP
  exit "$exit_code"
}
trap finish_deploy EXIT
trap 'deploy_interrupted 130' INT
trap 'deploy_interrupted 143' TERM HUP

note "Deploying one clean controller-code version to the head Spark"
attempted_head=true
deploy_remote head "$THREE_MACHINE_PROJECT_DIR" || die "head deployment failed"

note "Deploying one clean controller-code version to the worker Spark"
attempted_worker=true
deploy_remote worker "$THREE_MACHINE_PROJECT_DIR" || die "worker deployment failed"

note "Deploying one clean controller-code version to the Mac Studio"
attempted_studio=true
deploy_remote studio "$THREE_MACHINE_STUDIO_DIR" || die "Studio deployment failed"

expected_state=$(printf '%s\t%s\t%s' "$project_manifest" "$deploy_id" "$deploy_commit")
[[ "$(deployment_state head "$THREE_MACHINE_PROJECT_DIR")" == "$expected_state" ]] \
  || die "head post-deployment manifest marker differs"
[[ "$(deployment_state worker "$THREE_MACHINE_PROJECT_DIR")" == "$expected_state" ]] \
  || die "worker post-deployment manifest marker differs"
[[ "$(deployment_state studio "$THREE_MACHINE_STUDIO_DIR")" == "$expected_state" ]] \
  || die "Studio post-deployment manifest marker differs"
deployment_complete=true

note "Deployment complete (archive SHA-256 $archive_sha; project manifest $project_manifest); prepared model and TensorFold artifacts were preserved"
