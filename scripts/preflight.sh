#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"

note "Checking the pinned two-Spark GLM deployment"
head_ssh "set -eu
  test \"\$(git -C '$GLM_RECIPE_DIR' rev-parse HEAD)\" = '$GLM_RECIPE_COMMIT'
  grep -qx 'WORKER=$WORKER_SSH' '$GLM_RECIPE_DIR/scripts/local.sh'
  grep -qx 'DRAFTER=mtp' '$GLM_RECIPE_DIR/scripts/local.sh'
  cd '$GLM_RECIPE_DIR'
  source scripts/config.sh
  source scripts/nodes.sh
  test \"\$(prepared_state)\" = \"\$(cat \"\$PREPARED_MARKER\")\"
  test \"\$(docker image inspect '$TENSORFOLD_IMAGE_NAME' --format '{{.Id}}')\" = '$TENSORFOLD_IMAGE_ID'
  test \"\$(ssh -o BatchMode=yes '$WORKER_SSH' docker image inspect '$TENSORFOLD_IMAGE_NAME' --format={{.Id}})\" = '$TENSORFOLD_IMAGE_ID'
  test -f \"\$HOME/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/$GLM_MODEL_REVISION/config.json\"
  ssh -o BatchMode=yes '$WORKER_SSH' test -f \"\$HOME/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/$GLM_MODEL_REVISION/config.json\"
"

note "Checking the live Mac Studio MCDMA endpoint"
studio_ssh "set -eu
  test \"\$(git -C '$MCDMA_STUDIO_DIR' rev-parse HEAD)\" = '$MCDMA_COMMIT'
  '$MCDMA_STUDIO_DIR/build/cx5-native-check' --provider /usr/local/lib/rdma/libmcdma-rdmav34.so --require-gid >/dev/null
  ifconfig '$STUDIO_RDMA_INTERFACE' | grep -q 'RUNNING'
"

note "Checking the Spark endpoint wired to the Studio"
head_ssh "set -eu
  test -d /sys/class/infiniband/'$SPARK_RDMA_DEVICE'
  test \"\$(cat /sys/class/net/'$SPARK_RDMA_INTERFACE'/operstate)\" = up
"

note "PASS: GLM assets are pinned and MCDMA transport is live"
note "BOUNDARY: TensorFold is not yet using MCDMA"
