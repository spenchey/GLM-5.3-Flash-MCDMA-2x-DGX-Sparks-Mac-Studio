#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_three_machine_config

inspect_image() {
  local target=$1
  if [[ "$target" == head ]]; then
    head_ssh "docker image inspect '$THREE_MACHINE_IMAGE' --format '{{.Id}} {{index .Config.Labels \"tf.patches\"}}'"
  else
    worker_ssh "docker image inspect '$THREE_MACHINE_IMAGE' --format '{{.Id}} {{index .Config.Labels \"tf.patches\"}}'"
  fi
}

note "Checking all three machines and pinned artifacts"
local_project_manifest=$(python3 "$PROJECT_ROOT/scripts/project-manifest.py" "$PROJECT_ROOT")
local_config_hash=$(shasum -a 256 "$CONFIG_FILE" | awk '{print $1}')
local_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
reviewed_tensorfold_tree=$(python3 - "$PROJECT_ROOT/patches/tensorfold/metal-stage-manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["patched_tree_sha256"])
PY
)
[[ "$THREE_MACHINE_TENSORFOLD_SOURCE_MANIFEST_SHA256" == "$reviewed_tensorfold_tree" ]] \
  || die "configured TensorFold tree pin differs from the committed reviewed manifest"
head_project_marker=$(head_ssh "test -r '$THREE_MACHINE_PROJECT_DIR/.deployment-sha256' && cat '$THREE_MACHINE_PROJECT_DIR/.deployment-sha256'") \
  || die "head Spark has no readable deployment manifest; deploy before preflight"
worker_project_marker=$(worker_ssh "test -r '$THREE_MACHINE_PROJECT_DIR/.deployment-sha256' && cat '$THREE_MACHINE_PROJECT_DIR/.deployment-sha256'") \
  || die "worker Spark has no readable deployment manifest; deploy before preflight"
studio_project_marker=$(studio_ssh "test -r '$THREE_MACHINE_STUDIO_DIR/.deployment-sha256' && cat '$THREE_MACHINE_STUDIO_DIR/.deployment-sha256'") \
  || die "Mac Studio has no readable deployment manifest; deploy before preflight"
head_deploy_id=$(head_ssh "test -r '$THREE_MACHINE_PROJECT_DIR/.deployment-id' && cat '$THREE_MACHINE_PROJECT_DIR/.deployment-id'") \
  || die "head Spark has no readable deployment ID; deploy before preflight"
worker_deploy_id=$(worker_ssh "test -r '$THREE_MACHINE_PROJECT_DIR/.deployment-id' && cat '$THREE_MACHINE_PROJECT_DIR/.deployment-id'") \
  || die "worker Spark has no readable deployment ID; deploy before preflight"
studio_deploy_id=$(studio_ssh "test -r '$THREE_MACHINE_STUDIO_DIR/.deployment-id' && cat '$THREE_MACHINE_STUDIO_DIR/.deployment-id'") \
  || die "Mac Studio has no readable deployment ID; deploy before preflight"
head_deploy_commit=$(head_ssh "test -r '$THREE_MACHINE_PROJECT_DIR/.deployment-commit' && cat '$THREE_MACHINE_PROJECT_DIR/.deployment-commit'") \
  || die "head Spark has no readable deployment commit; deploy before preflight"
worker_deploy_commit=$(worker_ssh "test -r '$THREE_MACHINE_PROJECT_DIR/.deployment-commit' && cat '$THREE_MACHINE_PROJECT_DIR/.deployment-commit'") \
  || die "worker Spark has no readable deployment commit; deploy before preflight"
studio_deploy_commit=$(studio_ssh "test -r '$THREE_MACHINE_STUDIO_DIR/.deployment-commit' && cat '$THREE_MACHINE_STUDIO_DIR/.deployment-commit'") \
  || die "Mac Studio has no readable deployment commit; deploy before preflight"
[[ -n "$head_deploy_id" && "$head_deploy_id" == "$worker_deploy_id" \
    && "$head_deploy_id" == "$studio_deploy_id" ]] \
  || die "the three runtime hosts do not share one deployment ID"
[[ "$head_deploy_commit" == "$local_commit" && "$worker_deploy_commit" == "$local_commit" \
    && "$studio_deploy_commit" == "$local_commit" ]] \
  || die "the three runtime hosts do not share the controller's exact deployed commit"
head_config_hash=$(head_ssh "sha256sum '$THREE_MACHINE_PROJECT_DIR/config.local.env' | cut -d' ' -f1")
worker_config_hash=$(worker_ssh "sha256sum '$THREE_MACHINE_PROJECT_DIR/config.local.env' | cut -d' ' -f1")
studio_config_hash=$(studio_ssh "shasum -a 256 '$THREE_MACHINE_STUDIO_DIR/config.local.env' | awk '{print \$1}'")
for config_hash in "$head_config_hash" "$worker_config_hash" "$studio_config_hash"; do
  [[ "$config_hash" == "$local_config_hash" ]] \
    || die "private deployment configuration differs across the controller and runtime hosts"
done
head_project_actual=$(head_ssh "python3 '$THREE_MACHINE_PROJECT_DIR/scripts/project-manifest.py' '$THREE_MACHINE_PROJECT_DIR'")
worker_project_actual=$(worker_ssh "python3 '$THREE_MACHINE_PROJECT_DIR/scripts/project-manifest.py' '$THREE_MACHINE_PROJECT_DIR'")
studio_project_actual=$(studio_ssh "python3 '$THREE_MACHINE_STUDIO_DIR/scripts/project-manifest.py' '$THREE_MACHINE_STUDIO_DIR'")
for project_hash in "$head_project_marker" "$worker_project_marker" "$studio_project_marker" \
  "$head_project_actual" "$worker_project_actual" "$studio_project_actual"; do
  [[ "$project_hash" == "$local_project_manifest" ]] \
    || die "deployed project code differs across the controller and three runtime hosts"
done
head_clock_ns=$(head_ssh "python3 -c 'import time; print(time.time_ns())'")
studio_clock_ns=$(studio_ssh "'$THREE_MACHINE_STUDIO_PYTHON' -c 'import time; print(time.time_ns())'")
python3 - "$head_clock_ns" "$studio_clock_ns" <<'PY'
import sys

if abs(int(sys.argv[1]) - int(sys.argv[2])) > 5_000_000_000:
    raise SystemExit("Mac Studio and head Spark wall clocks differ by more than five seconds")
PY
head_image=$(inspect_image head)
worker_image=$(inspect_image worker)
expected="$THREE_MACHINE_IMAGE_ID $THREE_MACHINE_PATCH_LABEL"
[[ "$head_image" == "$expected" ]] || die "head Spark image differs from the pinned image: $head_image"
[[ "$worker_image" == "$expected" ]] || die "worker Spark image differs from the pinned image: $worker_image"

head_ssh "test -r '$THREE_MACHINE_MODEL_DIR/config.json' && test -d '$THREE_MACHINE_PROJECT_DIR' && test -d '$THREE_MACHINE_HF_CACHE_DIR' && test -d '$THREE_MACHINE_TENSORFOLD_CACHE_DIR'"
worker_ssh "test -r '$THREE_MACHINE_MODEL_DIR/config.json' && test -d '$THREE_MACHINE_PROJECT_DIR' && test -d '$THREE_MACHINE_HF_CACHE_DIR' && test -d '$THREE_MACHINE_TENSORFOLD_CACHE_DIR'"
head_manifest=$(head_ssh "cd '$THREE_MACHINE_MODEL_DIR' && find -L . -type f -printf '%P\\t%s\\n' | LC_ALL=C sort | sha256sum | cut -d' ' -f1")
worker_manifest=$(worker_ssh "cd '$THREE_MACHINE_MODEL_DIR' && find -L . -type f -printf '%P\\t%s\\n' | LC_ALL=C sort | sha256sum | cut -d' ' -f1")
[[ "$head_manifest" == "$THREE_MACHINE_MODEL_MANIFEST_SHA256" ]] \
  || die "head Spark model filename/size manifest differs from the pin: $head_manifest"
[[ "$worker_manifest" == "$THREE_MACHINE_MODEL_MANIFEST_SHA256" ]] \
  || die "worker Spark model filename/size manifest differs from the pin: $worker_manifest"
head_model_metadata=$(head_ssh "python3 '$THREE_MACHINE_PROJECT_DIR/scripts/tree-manifest.py' --metadata-only '$THREE_MACHINE_MODEL_DIR'")
worker_model_metadata=$(worker_ssh "python3 '$THREE_MACHINE_PROJECT_DIR/scripts/tree-manifest.py' --metadata-only '$THREE_MACHINE_MODEL_DIR'")
[[ "$head_model_metadata" == "$THREE_MACHINE_MODEL_METADATA_SHA256" ]] \
  || die "head Spark model metadata/symlink manifest differs from the pin: $head_model_metadata"
[[ "$worker_model_metadata" == "$THREE_MACHINE_MODEL_METADATA_SHA256" ]] \
  || die "worker Spark model metadata/symlink manifest differs from the pin: $worker_model_metadata"
studio_ssh "test -r '$THREE_MACHINE_STAGE_MODEL_DIR/config.json' && test -r '$THREE_MACHINE_TENSORFOLD_SOURCE/tensorfold/__init__.py' && test -x '$THREE_MACHINE_STUDIO_PYTHON' && test -r '$THREE_MACHINE_MCDMA_LIBRARY' && test -d '$THREE_MACHINE_STUDIO_DIR'"
studio_ssh "PYTHONPATH='$THREE_MACHINE_TENSORFOLD_SOURCE' '$THREE_MACHINE_STUDIO_PYTHON' -c 'import mlx, mlx_lm, numpy, tensorfold'"

tensorfold_source_manifest=$(studio_ssh "'$THREE_MACHINE_STUDIO_PYTHON' '$THREE_MACHINE_STUDIO_DIR/scripts/tree-manifest.py' --exclude-relative .tensorfold-metal-stage.json '$THREE_MACHINE_TENSORFOLD_SOURCE'")
[[ "$tensorfold_source_manifest" == "$THREE_MACHINE_TENSORFOLD_SOURCE_MANIFEST_SHA256" ]] \
  || die "Mac TensorFold source tree differs from the reviewed pin: $tensorfold_source_manifest"
studio_ssh "cmp -s '$THREE_MACHINE_TENSORFOLD_SOURCE/.tensorfold-metal-stage.json' '$THREE_MACHINE_STUDIO_DIR/patches/tensorfold/metal-stage-manifest.json'" \
  || die "Mac TensorFold source review marker differs from the deployed manifest"
studio_python_hash=$(studio_ssh "shasum -a 256 '$THREE_MACHINE_STUDIO_PYTHON' | awk '{print \$1}'")
[[ "$studio_python_hash" == "$THREE_MACHINE_STUDIO_PYTHON_SHA256" ]] \
  || die "Mac Python executable differs from the pin: $studio_python_hash"
studio_python_env=$(studio_ssh "'$THREE_MACHINE_STUDIO_PYTHON' -c 'import importlib.metadata as m,platform; print(platform.python_version()+\",mlx=\"+m.version(\"mlx\")+\",mlx-lm=\"+m.version(\"mlx-lm\")+\",numpy=\"+m.version(\"numpy\"))'")
[[ "$studio_python_env" == "$THREE_MACHINE_STUDIO_PYTHON_ENV" ]] \
  || die "Mac Python/MLX environment differs from the pin: $studio_python_env"

studio_mcdma_library_hash=$(studio_ssh "shasum -a 256 '$THREE_MACHINE_MCDMA_LIBRARY' | awk '{print \$1}'")
studio_mcdma_daemon_hash=$(studio_ssh "shasum -a 256 '$THREE_MACHINE_STUDIO_MCDMA_DAEMON' | awk '{print \$1}'")
spark_mcdma_library_hash=$(head_ssh "sha256sum '$THREE_MACHINE_SPARK_MCDMA_LIBRARY' | cut -d' ' -f1")
spark_mcdma_daemon_hash=$(head_ssh "sha256sum '$THREE_MACHINE_SPARK_MCDMA_DAEMON' | cut -d' ' -f1")
[[ "$studio_mcdma_library_hash" == "$THREE_MACHINE_MCDMA_LIBRARY_SHA256" ]] \
  || die "Mac MCDMA library differs from the pin: $studio_mcdma_library_hash"
[[ "$studio_mcdma_daemon_hash" == "$THREE_MACHINE_STUDIO_MCDMA_DAEMON_SHA256" ]] \
  || die "Mac MCDMA daemon differs from the pin: $studio_mcdma_daemon_hash"
[[ "$spark_mcdma_library_hash" == "$THREE_MACHINE_SPARK_MCDMA_LIBRARY_SHA256" ]] \
  || die "Spark MCDMA library differs from the pin: $spark_mcdma_library_hash"
[[ "$spark_mcdma_daemon_hash" == "$THREE_MACHINE_SPARK_MCDMA_DAEMON_SHA256" ]] \
  || die "Spark MCDMA daemon differs from the pin: $spark_mcdma_daemon_hash"
studio_mcdma_commit=$(studio_ssh "git -C '$MCDMA_STUDIO_DIR' rev-parse HEAD")
[[ "$studio_mcdma_commit" == "$MCDMA_COMMIT" ]] \
  || die "Mac MCDMA source commit differs from the pin: $studio_mcdma_commit"
studio_mcdma_abi=$(studio_ssh "cd '$THREE_MACHINE_STUDIO_DIR'; env MCDMA_RPC_LIBRARY='$THREE_MACHINE_MCDMA_LIBRARY' PYTHONPATH='$THREE_MACHINE_STUDIO_DIR' '$THREE_MACHINE_STUDIO_PYTHON' -c 'from experiments.three_machine.mailbox import load_helper; print(load_helper().mcdma_rpc_abi())'")
[[ "$studio_mcdma_abi" == 1 ]] || die "Mac MCDMA helper ABI is not 1: $studio_mcdma_abi"

metal_linear_hash=$(studio_ssh "shasum -a 256 '$THREE_MACHINE_TENSORFOLD_SOURCE/tensorfold/families/glm5_next/linear.py' | awk '{print \$1}'")
metal_weights_hash=$(studio_ssh "shasum -a 256 '$THREE_MACHINE_TENSORFOLD_SOURCE/tensorfold/families/glm5_next/weights.py' | awk '{print \$1}'")
[[ "$metal_linear_hash" == "$THREE_MACHINE_METAL_LINEAR_SHA256" ]] \
  || die "Mac TensorFold linear.py differs from the reviewed stage patch: $metal_linear_hash"
[[ "$metal_weights_hash" == "$THREE_MACHINE_METAL_WEIGHTS_SHA256" ]] \
  || die "Mac TensorFold weights.py differs from the reviewed stage patch: $metal_weights_hash"

stage_hash=$(studio_ssh "shasum -a 256 '$THREE_MACHINE_STAGE_MODEL_DIR/$THREE_MACHINE_STAGE_FILE' | awk '{print \$1}'")
[[ "$stage_hash" == "$THREE_MACHINE_STAGE_SHA256" ]] || die "Mac stage checkpoint hash differs from the pin: $stage_hash"
stage_manifest=$(studio_ssh "'$THREE_MACHINE_STUDIO_PYTHON' '$THREE_MACHINE_STUDIO_DIR/scripts/tree-manifest.py' '$THREE_MACHINE_STAGE_MODEL_DIR'")
[[ "$stage_manifest" == "$THREE_MACHINE_STAGE_MANIFEST_SHA256" ]] \
  || die "Mac stage directory differs from the full reviewed pin: $stage_manifest"

for host in head worker; do
  if [[ "$host" == head ]]; then
    running=$(head_ssh "docker inspect '$THREE_MACHINE_CONTAINER' --format '{{.State.Running}}' 2>/dev/null || true")
  else
    running=$(worker_ssh "docker inspect '$THREE_MACHINE_CONTAINER' --format '{{.State.Running}}' 2>/dev/null || true")
  fi
  [[ -z "$running" || "$running" == true || "$running" == false ]] || die "$host container state is unreadable"
done

note "PASS: both Spark images/checkpoints, the exact Mac TensorFold source/stage, and MCDMA helper match the pinned deployment"
