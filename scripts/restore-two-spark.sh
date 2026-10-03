#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"

usage() {
  printf 'Usage: %s [--dry-run]\n' "$0" >&2
  exit 2
}

dry_run=false
case "${1:-}" in
  '') ;;
  --dry-run) dry_run=true ;;
  *) usage ;;
esac
[[ $# -le 1 ]] || usage

container_name=glm53-flash-tf
image_name=$TENSORFOLD_IMAGE_NAME
api_model=GLM-5.3-Flash-EXL3
fixed_answer=GLM_READY
success_marker=TWO_SPARK_BASELINE_OK

verify_pins_command=$(cat <<EOF
set -eu
test "\$(git -C '$GLM_RECIPE_DIR' rev-parse HEAD)" = '$GLM_RECIPE_COMMIT'
git -C '$GLM_RECIPE_DIR' diff --quiet
git -C '$GLM_RECIPE_DIR' diff --cached --quiet
grep -qx 'WORKER=$WORKER_SSH' '$GLM_RECIPE_DIR/scripts/local.sh'
grep -qx 'DRAFTER=mtp' '$GLM_RECIPE_DIR/scripts/local.sh'
test "\$(docker image inspect '$image_name' --format '{{.Id}}')" = '$TENSORFOLD_IMAGE_ID'
test "\$(ssh -o BatchMode=yes '$WORKER_SSH' docker image inspect '$image_name' --format={{.Id}})" = '$TENSORFOLD_IMAGE_ID'
test -f "\$HOME/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/$GLM_MODEL_REVISION/config.json"
ssh -o BatchMode=yes '$WORKER_SSH' test -f "\$HOME/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/$GLM_MODEL_REVISION/config.json"
EOF
)
stop_command="cd '$GLM_RECIPE_DIR' && ./stop.sh"
start_command="cd '$GLM_RECIPE_DIR' && DRAFTER=mtp PREPARE=0 ./start.sh"
inspect_head_command="docker inspect '$container_name' --format '{{.State.Status}}|{{.State.Running}}|{{.State.OOMKilled}}|{{.RestartCount}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}|{{.Image}}'"
inspect_worker_command="ssh -o BatchMode=yes '$WORKER_SSH' \"docker inspect '$container_name' --format='{{.State.Status}}|{{.State.Running}}|{{.State.OOMKilled}}|{{.RestartCount}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}|{{.Image}}'\""
models_command="curl -fsS --max-time 30 http://127.0.0.1:'$GLM_PORT'/v1/models"
answer_command="curl -fsS --max-time 180 http://127.0.0.1:'$GLM_PORT'/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"$api_model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly $fixed_answer\"}],\"max_tokens\":256,\"temperature\":0}'"

plan() {
  local number=$1 description=$2 command=$3
  printf 'PLAN %s: %s\n' "$number" "$description"
  printf '  ssh -o BatchMode=yes %q %q\n' "$HEAD_SSH" "$command"
}

if [[ "$dry_run" == true ]]; then
  printf 'DRY-RUN: no SSH command will be executed and no host will be changed.\n'
  plan 1 'verify the pinned recipe, clean tracked recipe files, image ID, and model revision on both Sparks' "$verify_pins_command"
  plan 2 'stop the current experiment through the pinned two-Spark recipe' "$stop_command"
  plan 3 'start the pinned live two-Spark service with MTP and no preparation changes' "$start_command"
  plan 4 'check the head container is running, healthy when a healthcheck exists, not OOM-killed, never restarted, and uses the pinned image' "$inspect_head_command"
  plan 5 'check the worker container with the same health, OOM, restart, and image gates' "$inspect_worker_command"
  plan 6 "require the API model ID to be exactly $api_model" "$models_command"
  plan 7 "require a stopped-normal response whose visible content is exactly $fixed_answer" "$answer_command"
  printf 'DRY-RUN COMPLETE: the success marker is intentionally withheld.\n'
  exit 0
fi

note 'Verifying the exact pinned rollback inputs before stopping the experiment'
head_ssh "$verify_pins_command"

note 'Stopping the current experiment'
head_ssh "$stop_command"

note 'Restoring the pinned live two-Spark service'
head_ssh "$start_command"

validate_container() {
  local role=$1 state=$2
  local status running oom restarts health image extra
  IFS='|' read -r status running oom restarts health image extra <<<"$state"
  [[ -z "${extra:-}" && -n "${image:-}" ]] || die "$role returned malformed container state"
  [[ "$status" == running && "$running" == true ]] || die "$role container is not running"
  [[ "$oom" == false ]] || die "$role container was OOM-killed"
  [[ "$restarts" == 0 ]] || die "$role container restart count is $restarts, expected 0"
  [[ "$health" == healthy || "$health" == none ]] || die "$role container health is $health"
  [[ "$image" == "$TENSORFOLD_IMAGE_ID" ]] || die "$role container image is not the pinned image"
}

note 'Checking both restored containers'
head_state=$(head_ssh "$inspect_head_command")
worker_state=$(head_ssh "$inspect_worker_command")
validate_container head "$head_state"
validate_container worker "$worker_state"

note 'Checking the restored API model identity'
models_response=$(head_ssh "$models_command")
python3 - "$api_model" "$models_response" <<'PY'
import json
import sys

expected, raw = sys.argv[1:]
payload = json.loads(raw)
models = [entry.get("id") for entry in payload.get("data", [])]
if models != [expected]:
    raise SystemExit(f"API models were {models!r}, expected exactly {[expected]!r}")
PY

note 'Running the fixed-answer baseline check'
answer_response=$(head_ssh "$answer_command")
python3 - "$fixed_answer" "$answer_response" <<'PY'
import json
import sys

expected, raw = sys.argv[1:]
payload = json.loads(raw)
choices = payload.get("choices", [])
if len(choices) != 1:
    raise SystemExit(f"fixed-answer check returned {len(choices)} choices, expected 1")
choice = choices[0]
if choice.get("finish_reason") != "stop":
    raise SystemExit(f"fixed-answer check finish_reason was {choice.get('finish_reason')!r}")
answer = choice.get("message", {}).get("content")
if not isinstance(answer, str) or answer.strip() != expected:
    raise SystemExit(f"fixed-answer check returned {answer!r}")
PY

printf '%s\n' "$success_marker"
