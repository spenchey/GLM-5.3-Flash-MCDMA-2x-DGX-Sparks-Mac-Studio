#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"

output=${1:?usage: attest-native-two-spark.sh OUTPUT.json [MODEL_ID]}
model=${2:-GLM-5.3-Flash-EXL3}
[[ "$output" = /* ]] || die "native attestation output must be an absolute path"
[[ "$model" =~ ^[A-Za-z0-9][A-Za-z0-9_.:/-]+$ ]] || die "native model ID is unsafe"

temporary=$(mktemp -d "${TMPDIR:-/tmp}/native-two-spark-attestation.XXXXXXXX")
cleanup_attestation() {
  local rc=$?
  rm -rf -- "$temporary"
  exit "$rc"
}
trap cleanup_attestation EXIT INT TERM HUP

metadata=$(head_ssh "set -eu
cd '$GLM_RECIPE_DIR'
test \"\$(git rev-parse HEAD)\" = '$GLM_RECIPE_COMMIT'
test -z \"\$(git status --porcelain --untracked-files=no)\"
grep -qx 'DRAFTER=mtp' scripts/local.sh
. scripts/config.sh
test \"\$DRAFTER\" = mtp
printf '%s\\t%s\\t%s\\t%s\\n' \"\$CONTAINER_NAME\" \"\$(git rev-parse HEAD)\" \"\$(git rev-parse 'HEAD^{tree}')\" \"\$(sha256sum scripts/local.sh | awk '{print \$1}')\"")
IFS=$'\t' read -r container_name recipe_commit recipe_tree recipe_config_sha256 extra <<<"$metadata"
[[ -z "$extra" && "$container_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]] \
  || die "native recipe returned malformed metadata"

format='{{.Name}}|{{.Id}}|{{.Image}}|{{.State.Running}}|{{.State.OOMKilled}}|{{.State.ExitCode}}'
head_state=$(head_ssh "docker inspect '$container_name' --format '$format'")
worker_state=$(worker_ssh "docker inspect '$container_name' --format '$format'")
printf '%s' "$head_state" >"$temporary/head.state"
printf '%s' "$worker_state" >"$temporary/worker.state"

python3 - "$temporary" "$output" "$model" "$THREE_MACHINE_IMAGE_ID" \
  "$recipe_commit" "$recipe_tree" "$recipe_config_sha256" <<'PY'
import json
import os
from pathlib import Path
import sys

root, output, model, expected_image, commit, tree, config_sha = sys.argv[1:]

def sha(value, label, *, prefixed=False):
    candidate = value.removeprefix("sha256:") if prefixed else value
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise SystemExit(f"{label} is not a full SHA-256")

def git_oid(value, label):
    if len(value) not in (40, 64) or any(char not in "0123456789abcdef" for char in value):
        raise SystemExit(f"{label} is not a full Git object ID")

def state(role):
    parts = (Path(root) / f"{role}.state").read_text().strip().split("|")
    if len(parts) != 6:
        raise SystemExit(f"native {role} container state is malformed")
    name, container_id, image_id, running, oom, exit_code = parts
    sha(container_id, f"native {role} container ID")
    sha(image_id, f"native {role} image ID", prefixed=True)
    if image_id != expected_image or running != "true" or oom != "false" or exit_code != "0":
        raise SystemExit(f"native {role} is not cleanly running the pinned image")
    return {
        "name": name.removeprefix("/"), "container_id": container_id,
        "image_id": image_id, "running": True, "oom_killed": False, "exit_code": 0,
    }

sha(expected_image, "expected image", prefixed=True)
git_oid(commit, "recipe commit")
git_oid(tree, "recipe tree")
sha(config_sha, "recipe config")
head, worker = state("head"), state("worker")
if head["container_id"] == worker["container_id"]:
    raise SystemExit("native head and worker container IDs are not distinct")
document = {
    "schema": 1, "kind": "native-two-spark", "model": model,
    "image_id": expected_image, "drafter": "mtp", "recipe_commit": commit,
    "recipe_tree": tree, "recipe_config_sha256": config_sha,
    "head": head, "worker": worker,
}
path = Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print(path)
PY
