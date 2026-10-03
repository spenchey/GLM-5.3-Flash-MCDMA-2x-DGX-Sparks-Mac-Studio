#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_three_machine_config

temporary=$(mktemp -d "${TMPDIR:-/tmp}/glm53-three-machine-quality.XXXXXXXX")
cleanup_quality() {
  local rc=$?
  rm -rf -- "$temporary"
  exit "$rc"
}
trap cleanup_quality EXIT INT TERM HUP

head_before=$(three_machine_container_state head)
worker_before=$(three_machine_container_state worker)
validate_three_machine_container_state "head Spark" "$head_before"
validate_three_machine_container_state "worker Spark" "$worker_before"
three_machine_container_running "$head_before" || die "head Spark is not running before quality probe"
three_machine_container_running "$worker_before" || die "worker Spark is not running before quality probe"
head_id=$(three_machine_container_id "$head_before")
worker_id=$(three_machine_container_id "$worker_before")
head_log_since=$(head_ssh "date -u +%Y-%m-%dT%H:%M:%S.%NZ")
worker_log_since=$(worker_ssh "date -u +%Y-%m-%dT%H:%M:%S.%NZ")
link_before=$(adapter peek)
verified_studio_api_listener_pid >/dev/null || die "managed Mac API is not running"
printf '%s' "$head_before" >"$temporary/head-before.txt"
printf '%s' "$worker_before" >"$temporary/worker-before.txt"

quoted_args=
for argument in "$@"; do
  printf -v quoted_argument '%q' "$argument"
  quoted_args+=" $quoted_argument"
done
[[ "${THREE_MACHINE_NATIVE_REFERENCE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] \
  || die "THREE_MACHINE_NATIVE_REFERENCE_SHA256 is not a pinned SHA-256"
printf -v reference_sha256 '%q' "$THREE_MACHINE_NATIVE_REFERENCE_SHA256"
printf -v native_image_id '%q' "$THREE_MACHINE_IMAGE_ID"
quality=$(studio_ssh "cd '$THREE_MACHINE_STUDIO_DIR' && '$THREE_MACHINE_STUDIO_PYTHON' scripts/quality-three-machine.py --url 'http://$THREE_MACHINE_API_HOST:$THREE_MACHINE_API_PORT/v1/chat/completions' --tasks experiments/three_machine/equivalence-prompts.json --reference-sha256 $reference_sha256 --expected-native-image-id $native_image_id$quoted_args")
printf '%s' "$quality" >"$temporary/quality.json"
head_ssh "docker logs --since '$head_log_since' '$head_id' 2>&1" >"$temporary/head-runtime.log"
worker_ssh "docker logs --since '$worker_log_since' '$worker_id' 2>&1" >"$temporary/worker-runtime.log"

head_after=$(three_machine_container_state head)
worker_after=$(three_machine_container_state worker)
validate_three_machine_container_state "head Spark" "$head_after"
validate_three_machine_container_state "worker Spark" "$worker_after"
three_machine_container_running "$head_after" || die "head Spark stopped during quality probe"
three_machine_container_running "$worker_after" || die "worker Spark stopped during quality probe"
[[ "$(three_machine_container_id "$head_after")" == "$(three_machine_container_id "$head_before")" ]] \
  || die "head Spark identity changed during quality probe"
[[ "$(three_machine_container_id "$worker_after")" == "$(three_machine_container_id "$worker_before")" ]] \
  || die "worker Spark identity changed during quality probe"
link_after=$(adapter peek)
printf '%s' "$head_after" >"$temporary/head-after.txt"
printf '%s' "$worker_after" >"$temporary/worker-after.txt"
printf '%s' "$link_before" >"$temporary/link-before.json"
printf '%s' "$link_after" >"$temporary/link-after.json"

python3 - "$temporary" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
quality = json.loads((root / "quality.json").read_text())
before = json.loads((root / "link-before.json").read_text())
after = json.loads((root / "link-after.json").read_text())
request_ids = {
    int(completion["request_id"])
    for task in quality["results"]
    for completion in task["completions"]
}
expected_steps = (int(quality["expected_mcdma_calls"]) - 2 * len(request_ids)) // 2

def events(raw, event):
    found = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("event") == event:
            found.append(value)
    return found

phase_events = [
    value for value in events((root / "head-runtime.log").read_text(errors="replace"), "three_machine_step")
    if int(value.get("request_id", -1)) in request_ids
]
rank0_events = [
    value for value in events(
        (root / "head-runtime.log").read_text(errors="replace"),
        "three_machine_rank0_commit",
    )
    if int(value.get("request_id", -1)) in request_ids
]
rank1_events = [
    value for value in events(
        (root / "worker-runtime.log").read_text(errors="replace"),
        "three_machine_rank1_commit",
    )
    if int(value.get("request_id", -1)) in request_ids
]
expected_keys = {
    (int(completion["request_id"]), int(step["step"]))
    for task in quality["results"]
    for completion in task["completions"]
    for step in completion["steps"]
}
def event_keys(values, label):
    keys = {
        (int(value.get("request_id", -1)), int(value.get("step", -1)))
        for value in values
    }
    if len(values) != expected_steps or len(keys) != len(values) or keys != expected_keys:
        raise SystemExit(f"{label} does not match every quality request and step")
    return keys

event_keys(phase_events, "head Spark steps")
event_keys(rank0_events, "rank 0 commits")
event_keys(rank1_events, "rank 1 commits")
expected = int(quality["expected_mcdma_calls"])
for label, state in (("before", before), ("after", after)):
    if state.get("link") != "up" or state.get("inflight"):
        raise SystemExit(f"MCDMA was not idle and up {label} quality probe")
    for side in ("mac", "spark"):
        endpoint = state.get(side, {})
        if not endpoint.get("alive") or int(endpoint.get("failures", -1)) != 0:
            raise SystemExit(f"MCDMA {side} was unhealthy {label} quality probe")
for side in ("mac", "spark"):
    delta = int(after[side]["calls"]) - int(before[side]["calls"])
    if delta != expected:
        raise SystemExit(f"MCDMA {side} handled {delta} calls, expected {expected}")

def container_state(name):
    parts = (root / f"{name}.txt").read_text().strip().split("|")
    if len(parts) != 6:
        raise SystemExit(f"{name} container state is malformed")
    container_name, container_id, image_id, running, oom, exit_code = parts
    if not container_name.startswith("/") or len(container_id) != 64 \
            or running != "true" or oom != "false" or exit_code != "0":
        raise SystemExit(f"{name} container is not cleanly running")
    return {
        "name": container_name,
        "id": container_id,
        "image_id": image_id,
        "running": True,
        "oom_killed": False,
        "exit_code": 0,
    }

containers = {
    name: container_state(name)
    for name in ("head-before", "head-after", "worker-before", "worker-after")
}
for role in ("head", "worker"):
    if containers[f"{role}-before"] != containers[f"{role}-after"]:
        raise SystemExit(f"{role} container identity changed during quality probe")
print(json.dumps({
    "schema": 2,
    "quality": quality,
    "mcdma_before": before,
    "mcdma_after": after,
    "mcdma_call_delta": {
        side: int(after[side]["calls"]) - int(before[side]["calls"])
        for side in ("mac", "spark")
    },
    "mcdma_failure_delta": {
        side: int(after[side]["failures"]) - int(before[side]["failures"])
        for side in ("mac", "spark")
    },
    "expected_steps": expected_steps,
    "spark_phase_events": phase_events,
    "rank0_commit_events": rank0_events,
    "rank1_commit_events": rank1_events,
    "containers": containers,
}, sort_keys=True, separators=(",", ":")))
PY
