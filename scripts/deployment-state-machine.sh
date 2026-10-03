#!/usr/bin/env bash
set -euo pipefail

# Move an already-verified runtime payload into place, or roll back one exact
# deployment. This helper is independent of SSH so its safety behavior can be
# tested locally before it is streamed to a runtime host.

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

validate_path() {
  local path=${1:?path required}
  [[ "$path" == /* && "$path" != / && "$path" != */../* && "$path" != */.. ]] \
    || die "unsafe deployment path: $path"
}

validate_managed_tree() {
  local path=${1:?path required}
  [[ -d "$path" && ! -L "$path" ]] || die "managed tree is not a real directory: $path"
  [[ ! -e "$path/.git" && ! -L "$path/.git" ]] \
    || die "refusing to manage a Git checkout: $path"
  [[ -r "$path/.deployment-sha256" ]] \
    || die "managed tree has no deployment marker: $path"
}

install_payload() {
  local destination=${1:?destination required}
  local incoming=${2:?incoming required}
  local expected_manifest=${3:?manifest required}
  local deploy_id=${4:?deployment ID required}
  local previous="${destination}.previous"
  local rc

  validate_path "$destination"
  validate_path "$incoming"
  [[ "$incoming" == "${destination}.incoming.${deploy_id}" ]] \
    || die "incoming path does not belong to this deployment"
  [[ -d "$incoming" && ! -L "$incoming" ]] || die "incoming payload is not a real directory"
  [[ ! -e "$incoming/.git" && ! -L "$incoming/.git" ]] \
    || die "incoming payload contains a Git checkout"
  [[ "$(cat "$incoming/.deployment-sha256")" == "$expected_manifest" ]] \
    || die "incoming deployment manifest differs"
  [[ "$(cat "$incoming/.deployment-id")" == "$deploy_id" ]] \
    || die "incoming deployment ID differs"

  # An untrappable interruption can land after the current tree was renamed
  # but before the incoming tree was promoted. Restore the marked previous
  # tree first so a later deployment never discards the only known-good copy.
  if [[ ! -e "$destination" && ! -L "$destination" \
        && ( -e "$previous" || -L "$previous" ) ]]; then
    validate_managed_tree "$previous"
    mv -- "$previous" "$destination"
  fi
  if [[ -e "$destination" || -L "$destination" ]]; then
    validate_managed_tree "$destination"
  fi
  if [[ -e "$previous" || -L "$previous" ]]; then
    validate_managed_tree "$previous"
    rm -rf -- "$previous"
  fi

  cleanup_install() {
    rc=$?
    trap - EXIT INT TERM HUP
    if (( rc != 0 )); then
      if [[ ! -e "$destination" && ! -L "$destination" \
            && -d "$previous" && ! -L "$previous" ]]; then
        validate_managed_tree "$previous" && mv -- "$previous" "$destination" || true
      fi
      if [[ -d "$incoming" && ! -L "$incoming" ]]; then
        rm -rf -- "$incoming"
      fi
    fi
    exit "$rc"
  }
  trap cleanup_install EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM HUP

  if [[ -e "$destination" ]]; then
    mv -- "$destination" "$previous"
  fi
  mv -- "$incoming" "$destination"
  trap - EXIT INT TERM HUP
}

rollback_payload() {
  local destination=${1:?destination required}
  local expected_manifest=${2:?manifest required}
  local deploy_id=${3:?deployment ID required}
  local previous="${destination}.previous"
  local failed="${destination}.failed.${deploy_id}"
  local current_id='' rc

  validate_path "$destination"
  validate_path "$failed"
  if [[ -e "$failed" || -L "$failed" ]]; then
    validate_managed_tree "$failed"
    [[ "$(cat "$failed/.deployment-id")" == "$deploy_id" ]] \
      || die "rollback scratch deployment ID differs"
    [[ "$(cat "$failed/.deployment-sha256")" == "$expected_manifest" ]] \
      || die "rollback scratch manifest differs"
    if [[ ! -e "$destination" && ! -L "$destination" \
          && ( -e "$previous" || -L "$previous" ) ]]; then
      validate_managed_tree "$previous"
      mv -- "$previous" "$destination"
    elif [[ -e "$destination" || -L "$destination" ]]; then
      validate_managed_tree "$destination"
      [[ ! -e "$previous" && ! -L "$previous" ]] \
        || die "rollback retry found both destination and previous trees"
    fi
    rm -rf -- "$failed"
    return 0
  fi

  if [[ ! -e "$destination" && ! -L "$destination" ]]; then
    if [[ -e "$previous" || -L "$previous" ]]; then
      validate_managed_tree "$previous"
      mv -- "$previous" "$destination"
    fi
    return 0
  fi

  validate_managed_tree "$destination"
  if [[ -r "$destination/.deployment-id" ]]; then
    current_id=$(cat "$destination/.deployment-id")
  fi
  # A failed upload or pre-swap validation leaves the old deployment in place.
  # It must not be mistaken for the deployment currently being rolled back.
  [[ "$current_id" == "$deploy_id" ]] || return 0
  [[ "$(cat "$destination/.deployment-sha256")" == "$expected_manifest" ]] \
    || die "accepted deployment manifest differs during rollback"
  if [[ -e "$previous" || -L "$previous" ]]; then
    validate_managed_tree "$previous"
  fi

  cleanup_rollback() {
    rc=$?
    trap - EXIT INT TERM HUP
    if (( rc != 0 )) && [[ ! -e "$destination" && ! -L "$destination" \
                          && -d "$failed" && ! -L "$failed" ]]; then
      mv -- "$failed" "$destination" || true
    fi
    exit "$rc"
  }
  trap cleanup_rollback EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM HUP

  mv -- "$destination" "$failed"
  if [[ -d "$previous" && ! -L "$previous" ]]; then
    mv -- "$previous" "$destination"
  fi
  validate_managed_tree "$failed"
  [[ "$(cat "$failed/.deployment-id")" == "$deploy_id" ]] \
    || die "rollback scratch deployment ID differs"
  rm -rf -- "$failed"
  trap - EXIT INT TERM HUP
}

case "${1:-}" in
  install)
    [[ $# -eq 5 ]] || die "install requires DESTINATION INCOMING MANIFEST DEPLOY_ID"
    install_payload "$2" "$3" "$4" "$5"
    ;;
  rollback)
    [[ $# -eq 4 ]] || die "rollback requires DESTINATION MANIFEST DEPLOY_ID"
    rollback_payload "$2" "$3" "$4"
    ;;
  *) die "unknown deployment operation: ${1:-missing}" ;;
esac
