#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_three_machine_config

temporary=$(mktemp -d "${TMPDIR:-/tmp}/glm53-three-machine-idle.XXXXXXXX")
cleanup_idle() {
  local rc=$?
  rm -rf -- "$temporary"
  exit "$rc"
}
trap cleanup_idle EXIT INT TERM HUP

head_before=$(three_machine_container_state head)
worker_before=$(three_machine_container_state worker)
validate_three_machine_container_state "head Spark" "$head_before"
validate_three_machine_container_state "worker Spark" "$worker_before"
three_machine_container_running "$head_before" || die "head Spark is not running before idle measurement"
three_machine_container_running "$worker_before" || die "worker Spark is not running before idle measurement"
head_id=$(three_machine_container_id "$head_before")
worker_id=$(three_machine_container_id "$worker_before")
head_log_since=$(head_ssh "date -u +%Y-%m-%dT%H:%M:%S.%NZ")
worker_log_since=$(worker_ssh "date -u +%Y-%m-%dT%H:%M:%S.%NZ")
link_before=$(adapter peek)
verified_studio_api_listener_pid >/dev/null || die "managed Mac API is not running"

quoted_args=
for argument in "$@"; do
  printf -v quoted_argument '%q' "$argument"
  quoted_args+=" $quoted_argument"
done
[[ "${THREE_MACHINE_NATIVE_REFERENCE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] \
  || die "THREE_MACHINE_NATIVE_REFERENCE_SHA256 is not a pinned SHA-256"
printf -v reference_sha256 '%q' "$THREE_MACHINE_NATIVE_REFERENCE_SHA256"
printf -v native_image_id '%q' "$THREE_MACHINE_IMAGE_ID"
idle=$(studio_ssh "cd '$THREE_MACHINE_STUDIO_DIR' && '$THREE_MACHINE_STUDIO_PYTHON' scripts/idle-three-machine.py --url 'http://$THREE_MACHINE_API_HOST:$THREE_MACHINE_API_PORT/v1/chat/completions' --reference-file experiments/three_machine/native-token-references.json --reference-sha256 $reference_sha256 --expected-native-image-id $native_image_id$quoted_args")
printf '%s' "$idle" >"$temporary/idle.json"
head_ssh "docker logs --since '$head_log_since' '$head_id' 2>&1" >"$temporary/head-runtime.log"
worker_ssh "docker logs --since '$worker_log_since' '$worker_id' 2>&1" >"$temporary/worker-runtime.log"

head_after=$(three_machine_container_state head)
worker_after=$(three_machine_container_state worker)
[[ "$head_after" == "$head_before" && "$worker_after" == "$worker_before" ]] \
  || die "Spark container identity changed during idle measurement"
link_after=$(adapter peek)

printf '%s' "$head_before" >"$temporary/head-before.txt"
printf '%s' "$head_after" >"$temporary/head-after.txt"
printf '%s' "$worker_before" >"$temporary/worker-before.txt"
printf '%s' "$worker_after" >"$temporary/worker-after.txt"
printf '%s' "$link_before" >"$temporary/link-before.json"
printf '%s' "$link_after" >"$temporary/link-after.json"

python3 - "$temporary" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
idle = json.loads((root / "idle.json").read_text())
before = json.loads((root / "link-before.json").read_text())
after = json.loads((root / "link-after.json").read_text())

def events(path, name):
    result = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("event") == name:
            result.append(value)
    return result

expected_keys = {
    (int(request["request_id"]), int(step["step"]))
    for request in idle["requests"] for step in request["steps"]
}
request_ids = {request_id for request_id, _ in expected_keys}
def matching(path, name):
    values = [
        value for value in events(path, name)
        if int(value.get("request_id", -1)) in request_ids
    ]
    keys = {(int(value.get("request_id", -1)), int(value.get("step", -1))) for value in values}
    if len(values) != len(keys) or keys != expected_keys:
        raise SystemExit(f"{name} does not match every idle request step")
    return values

phase = matching(root / "head-runtime.log", "three_machine_step")
rank0 = matching(root / "head-runtime.log", "three_machine_rank0_commit")
rank1 = matching(root / "worker-runtime.log", "three_machine_rank1_commit")
expected_calls = int(idle["expected_mcdma_calls"])
for label, state in (("before", before), ("after", after)):
    if state.get("link") != "up" or state.get("inflight"):
        raise SystemExit(f"MCDMA was not idle and up {label} idle measurement")
    for side in ("mac", "spark"):
        if not state[side].get("alive") or int(state[side].get("failures", -1)) != 0:
            raise SystemExit(f"MCDMA {side} was unhealthy {label} idle measurement")
call_delta = {side: int(after[side]["calls"]) - int(before[side]["calls"]) for side in ("mac", "spark")}
if any(value != expected_calls for value in call_delta.values()):
    raise SystemExit("idle MCDMA counters do not match measured work")

def container_state(name):
    parts = (root / f"{name}.txt").read_text().strip().split("|")
    if len(parts) != 6:
        raise SystemExit(f"{name} container state is malformed")
    container_name, container_id, image_id, running, oom, exit_code = parts
    return {
        "name": container_name, "id": container_id, "image_id": image_id,
        "running": running == "true", "oom_killed": oom == "true",
        "exit_code": int(exit_code),
    }

result = {key: value for key, value in idle.items() if key != "requests"}
result.update({
    "containers": {name: container_state(name) for name in (
        "head-before", "head-after", "worker-before", "worker-after"
    )},
    "mcdma_before": before, "mcdma_after": after, "mcdma_call_delta": call_delta,
    "spark_phase_events": phase, "rank0_commit_events": rank0,
    "rank1_commit_events": rank1, "expected_steps": len(expected_keys),
})
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
PY
