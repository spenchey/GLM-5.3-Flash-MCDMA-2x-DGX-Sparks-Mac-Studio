#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config

note "Checking the exact portable GLM checkpoint on the Mac and both Sparks"

local_manifest=$(python3 "$PROJECT_ROOT/scripts/project-manifest.py" "$PROJECT_ROOT")
local_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
for role in head worker studio; do
  case "$role" in
    head)
      marker=$(head_ssh "cat '$THREE_MACHINE_PROJECT_DIR/.deployment-sha256'")
      commit=$(head_ssh "cat '$THREE_MACHINE_PROJECT_DIR/.deployment-commit'")
      ;;
    worker)
      marker=$(worker_ssh "cat '$THREE_MACHINE_PROJECT_DIR/.deployment-sha256'")
      commit=$(worker_ssh "cat '$THREE_MACHINE_PROJECT_DIR/.deployment-commit'")
      ;;
    studio)
      marker=$(studio_ssh "cat '$THREE_MACHINE_STUDIO_DIR/.deployment-sha256'")
      commit=$(studio_ssh "cat '$THREE_MACHINE_STUDIO_DIR/.deployment-commit'")
      ;;
  esac
  [[ "$marker" == "$local_manifest" && "$commit" == "$local_commit" ]] \
    || die "$role does not have this exact reviewed deployment"
done

expected_image="$THREE_MACHINE_IMAGE_ID $THREE_MACHINE_PATCH_LABEL"
head_image=$(head_ssh "docker image inspect '$THREE_MACHINE_IMAGE' --format '{{.Id}} {{index .Config.Labels \"tf.patches\"}}'")
worker_image=$(worker_ssh "docker image inspect '$THREE_MACHINE_IMAGE' --format '{{.Id}} {{index .Config.Labels \"tf.patches\"}}'")
[[ "$head_image" == "$expected_image" ]] || die "the head Spark image differs from the pin"
[[ "$worker_image" == "$expected_image" ]] || die "the worker Spark image differs from the pin"

fact_dir="$PROJECT_ROOT/.state/preflight-facts-$$"
mkdir -p "$fact_dir"
cleanup_facts() {
  rm -f "$fact_dir/head.json" "$fact_dir/head.stderr" \
    "$fact_dir/worker.json" "$fact_dir/worker.stderr" \
    "$fact_dir/studio.json" "$fact_dir/studio.stderr"
  rmdir "$fact_dir" 2>/dev/null || true
}
trap cleanup_facts EXIT

head_ssh "python3 '$THREE_MACHINE_PROJECT_DIR/scripts/model_content_manifest.py' '$THREE_MACHINE_PORTABLE_MODEL_DIR'" \
  >"$fact_dir/head.json" 2>"$fact_dir/head.stderr" &
head_fact_pid=$!
worker_ssh "python3 '$THREE_MACHINE_PROJECT_DIR/scripts/model_content_manifest.py' '$THREE_MACHINE_PORTABLE_MODEL_DIR'" \
  >"$fact_dir/worker.json" 2>"$fact_dir/worker.stderr" &
worker_fact_pid=$!
studio_ssh "'$THREE_MACHINE_STUDIO_PYTHON' '$THREE_MACHINE_STUDIO_DIR/scripts/model_content_manifest.py' '$THREE_MACHINE_PORTABLE_MODEL_STUDIO_DIR'" \
  >"$fact_dir/studio.json" 2>"$fact_dir/studio.stderr" &
studio_fact_pid=$!

fact_failed=false
for role_pid in "head:$head_fact_pid" "worker:$worker_fact_pid" "studio:$studio_fact_pid"; do
  role=${role_pid%%:*}
  pid=${role_pid#*:}
  if ! wait "$pid"; then
    printf 'ERROR: %s portable-model identity check failed\n' "$role" >&2
    cat "$fact_dir/$role.stderr" >&2
    fact_failed=true
  fi
done
[[ "$fact_failed" == false ]] || die "one or more portable-model identity checks failed"

head_fact=$(<"$fact_dir/head.json")
worker_fact=$(<"$fact_dir/worker.json")
studio_fact=$(<"$fact_dir/studio.json")
cleanup_facts
trap - EXIT
python3 - "$THREE_MACHINE_PORTABLE_MODEL_CONTENT_SHA256" \
  "$THREE_MACHINE_PORTABLE_MODEL_MANIFEST_SHA256" "$head_fact" "$worker_fact" "$studio_fact" <<'PY'
import json
import sys

expected = sys.argv[1]
expected_sizes = sys.argv[2]
facts = [json.loads(value) for value in sys.argv[3:]]
if any(fact != facts[0] for fact in facts[1:]):
    raise SystemExit(f"portable model content differs across the three machines: {facts}")
fact = facts[0]
if fact.get("sha256") != expected:
    raise SystemExit(f"portable model content differs from the pin: {fact}")
if fact.get("size_sha256") != expected_sizes:
    raise SystemExit(f"portable model filename/size inventory differs from the pin: {fact}")
if fact.get("files") != 54 or fact.get("bytes") != 181741759037:
    raise SystemExit(f"portable model inventory is incomplete: {fact}")
PY

studio_metadata=$(studio_ssh "'$THREE_MACHINE_STUDIO_PYTHON' '$THREE_MACHINE_STUDIO_DIR/scripts/tree-manifest.py' --metadata-only '$THREE_MACHINE_PORTABLE_MODEL_STUDIO_DIR'")
[[ "$studio_metadata" == "$THREE_MACHINE_PORTABLE_MODEL_METADATA_SHA256" ]] \
  || die "the Mac portable-model metadata differs from the pin"
studio_ssh "PYTHONPATH='$THREE_MACHINE_TENSORFOLD_SOURCE' '$THREE_MACHINE_STUDIO_PYTHON' -c 'import mlx, numpy, tensorfold'"
studio_ssh "PYTHONPATH='$THREE_MACHINE_TENSORFOLD_SOURCE' '$THREE_MACHINE_STUDIO_PYTHON' -c 'import inspect; from tensorfold.families.glm5_next import load; from tensorfold.families.glm5_next.runtime import GLMDFlash; assert \"drafter\" in inspect.signature(load).parameters; assert GLMDFlash.__name__ == \"GLMDFlash\"'" \
  || die "Mac TensorFold source does not expose the reviewed GLM DFlash2 runtime"
studio_ssh "test -r '$THREE_MACHINE_DFLASH_STUDIO_DIR/config.json' && test -r '$THREE_MACHINE_DFLASH_STUDIO_DIR/model.safetensors'" \
  || die "the reviewed Mac DFlash2 helper is absent"
dflash_manifest=$(studio_ssh "'$THREE_MACHINE_STUDIO_PYTHON' '$THREE_MACHINE_STUDIO_DIR/scripts/tree-manifest.py' '$THREE_MACHINE_DFLASH_STUDIO_DIR'")
[[ "$dflash_manifest" == "$THREE_MACHINE_DFLASH_MANIFEST_SHA256" ]] \
  || die "the Mac DFlash2 helper differs from the reviewed pin"
tensorfold_source_manifest=$(studio_ssh "'$THREE_MACHINE_STUDIO_PYTHON' '$THREE_MACHINE_STUDIO_DIR/scripts/tree-manifest.py' --exclude-relative .tensorfold-metal-stage.json '$THREE_MACHINE_TENSORFOLD_SOURCE'")
[[ "$tensorfold_source_manifest" == "$THREE_MACHINE_TENSORFOLD_SOURCE_MANIFEST_SHA256" ]] \
  || die "Mac TensorFold source tree differs from the reviewed pin"
studio_ssh "cmp -s '$THREE_MACHINE_TENSORFOLD_SOURCE/.tensorfold-metal-stage.json' '$THREE_MACHINE_STUDIO_DIR/patches/tensorfold/metal-stage-manifest.json'" \
  || die "Mac TensorFold source marker differs from the reviewed manifest"
studio_python_hash=$(studio_ssh "shasum -a 256 '$THREE_MACHINE_STUDIO_PYTHON' | awk '{print \$1}'")
[[ "$studio_python_hash" == "$THREE_MACHINE_STUDIO_PYTHON_SHA256" ]] \
  || die "Mac Python executable differs from the pin"
studio_python_env=$(studio_ssh "'$THREE_MACHINE_STUDIO_PYTHON' -c 'import importlib.metadata as m,platform; print(platform.python_version()+\",mlx=\"+m.version(\"mlx\")+\",mlx-lm=\"+m.version(\"mlx-lm\")+\",numpy=\"+m.version(\"numpy\"))'")
[[ "$studio_python_env" == "$THREE_MACHINE_STUDIO_PYTHON_ENV" ]] \
  || die "Mac Python/MLX environment differs from the pin"

for role in head worker; do
  state=$(cache_handoff_container_state "$role")
  [[ -z "$state" ]] || die "a prior cache-handoff container remains on the $role Spark; stop it first"
done

note "PASS: one exact portable GLM model, one TensorFold build, and one reviewed deployment are present on all three machines"
