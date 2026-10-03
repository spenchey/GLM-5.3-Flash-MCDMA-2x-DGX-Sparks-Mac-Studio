#!/usr/bin/env python3
"""Validate and join one three-machine benchmark evidence bundle."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


MEMORY_FIELDS = (
    "cuda_allocated_bytes",
    "cuda_reserved_bytes",
    "process_rss_bytes",
)
PHASE_FIELDS = ("buffer_copy_s", "layers_s", "head_s", "agreement_s")


def event_keys(values: list[dict[str, Any]], label: str) -> set[tuple[int, int]]:
    keys = {
        (int(value.get("request_id", -1)), int(value.get("step", -1)))
        for value in values
    }
    if len(keys) != len(values) or any(request_id < 0 or step < 0 for request_id, step in keys):
        raise ValueError(f"{label} has duplicate or malformed request/step identities")
    return keys


def parse_container_state(raw: str, label: str) -> dict[str, Any]:
    parts = raw.strip().split("|")
    if len(parts) != 6:
        raise ValueError(f"{label} container state is malformed")
    name, container_id, image_id, running, oom_killed, exit_code = parts
    if not name.startswith("/") or len(container_id) != 64 \
            or any(char not in "0123456789abcdef" for char in container_id):
        raise ValueError(f"{label} container identity is malformed")
    if not image_id.startswith("sha256:") or len(image_id) != 71:
        raise ValueError(f"{label} container image identity is malformed")
    if running != "true" or oom_killed != "false" or exit_code != "0":
        raise ValueError(f"{label} container is not cleanly running")
    return {
        "name": name,
        "id": container_id,
        "image_id": image_id,
        "running": True,
        "oom_killed": False,
        "exit_code": 0,
    }


def parse_environment(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in raw.splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in result:
            raise ValueError(f"duplicate container environment key {key}")
        result[key] = value
    return result


def events(path: Path, event: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for raw in path.read_text(errors="replace").splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if value.get("event") == event:
            found.append(value)
    return found


def kib_value(path: Path, label: str) -> int:
    for raw in path.read_text(errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith(label):
            return int(stripped[len(label):].strip().split()[0])
    raise ValueError(f"{path.name} is missing {label}")


def growth_limit(baseline: int) -> int:
    return max(64 << 20, int(baseline * 0.01))


def require_growth_within(label: str, baseline: int, final: int) -> int:
    growth = final - baseline
    if growth > growth_limit(baseline):
        raise ValueError(f"{label} grew beyond 64 MiB or 1%")
    return growth


def require_available_memory(path_before: Path, path_after: Path, label: str) -> int:
    before = kib_value(path_before, "MemAvailable:") * 1024
    after = kib_value(path_after, "MemAvailable:") * 1024
    drop = before - after
    if drop > growth_limit(before):
        raise ValueError(f"{label} host MemAvailable dropped beyond 64 MiB or 1%")
    return drop


def _memory_snapshot(event: dict[str, Any], label: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for field in MEMORY_FIELDS:
        value = event.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} is missing nonnegative {field}")
        result[field] = value
    return result


def validate_rank_memory(
    rank_events: list[dict[str, Any]],
    warmup_request_id: int,
    final_request_id: int,
    label: str,
) -> dict[str, Any]:
    warm = [event for event in rank_events if int(event.get("request_id", -1)) == warmup_request_id]
    final = [event for event in rank_events if int(event.get("request_id", -1)) == final_request_id]
    if not warm or not final:
        raise ValueError(f"{label} has no post-warmup or final memory event")
    baseline = _memory_snapshot(warm[-1], f"{label} post-warmup")
    after = _memory_snapshot(final[-1], f"{label} final")
    growth = {
        field: require_growth_within(f"{label} {field}", baseline[field], after[field])
        for field in MEMORY_FIELDS
    }
    return {"post_warmup": baseline, "final": after, "growth_bytes": growth}


def join_step_evidence(
    benchmark: dict[str, Any],
    phase_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for event in phase_events:
        key = (int(event.get("request_id", -1)), int(event.get("step", -1)))
        if key in by_key:
            raise ValueError(f"duplicate head Spark step event {key}")
        by_key[key] = event
    joined: list[dict[str, Any]] = []
    for run in benchmark["warmups"] + benchmark["runs"]:
        request_id = int(run["request_id"])
        for step in run["steps"]:
            key = (request_id, int(step["step"]))
            spark = by_key.get(key)
            if spark is None:
                raise ValueError(f"missing head Spark step event {key}")
            roundtrip = float(step["spark_mcdma_roundtrip_s"])
            service = float(spark["spark_service_s"])
            residual = roundtrip - service
            if residual < -0.001:
                raise ValueError(f"Spark service time exceeds Mac round trip at {key}")
            joined.append({
                "request_id": request_id,
                "step": key[1],
                "prompt": bool(step["prompt"]),
                "mac_spark_roundtrip_s": roundtrip,
                "spark_service_s": service,
                "transport_wake_residual_s": max(0.0, residual),
                "gpu_total_s": float(spark["gpu_total_s"]),
            })
    if len(joined) != len(by_key):
        raise ValueError("head Spark emitted unmatched benchmark step events")
    return joined


def validate_bundle(root: Path) -> dict[str, Any]:
    benchmark = json.loads((root / "benchmark.json").read_text())
    before = json.loads((root / "link-before.json").read_text())
    after = json.loads((root / "link-after.json").read_text())
    all_runs = benchmark["warmups"] + benchmark["runs"]
    if not benchmark["warmups"] or not benchmark["runs"]:
        raise ValueError("benchmark evidence requires warmup and measured runs")
    request_ids = {int(run["request_id"]) for run in all_runs}
    expected_steps = sum(len(run["steps"]) for run in all_runs)

    phase_events = [
        value for value in events(root / "head-runtime.log", "three_machine_step")
        if int(value.get("request_id", -1)) in request_ids
    ]
    rank0_events = [
        value for value in events(root / "head-runtime.log", "three_machine_rank0_commit")
        if int(value.get("request_id", -1)) in request_ids
    ]
    rank1_events = [
        value for value in events(root / "worker-runtime.log", "three_machine_rank1_commit")
        if int(value.get("request_id", -1)) in request_ids
    ]
    for label, values in (
        ("head Spark steps", phase_events),
        ("rank 0 commits", rank0_events),
        ("rank 1 commits", rank1_events),
    ):
        if len(values) != expected_steps:
            raise ValueError(f"{label} emitted {len(values)} events, expected {expected_steps}")
    expected_keys = {
        (int(run["request_id"]), int(step["step"]))
        for run in all_runs for step in run["steps"]
    }
    for label, values in (
        ("head Spark steps", phase_events),
        ("rank 0 commits", rank0_events),
        ("rank 1 commits", rank1_events),
    ):
        if event_keys(values, label) != expected_keys:
            raise ValueError(f"{label} do not match every measured request and step")

    timing_mode = benchmark["health_before"]["performance_knobs"]["THREE_MACHINE_PHASE_TIMING"]
    expected_mode = "synchronized" if timing_mode == "1" else "host-enqueue"
    for event in phase_events:
        if event.get("timing_mode") != expected_mode:
            raise ValueError("Spark phase evidence does not match the deployed timing mode")
        if float(event.get("gpu_total_s", 0)) <= 0:
            raise ValueError("Spark event has no positive gpu_total_s")
        present = [field in event for field in PHASE_FIELDS]
        if expected_mode == "synchronized" and not all(present):
            raise ValueError("synchronized Spark event is missing phase timings")
        if expected_mode == "host-enqueue" and any(present):
            raise ValueError("host-enqueue event falsely labels unsynchronized phase timings")

    joined_steps = join_step_evidence(benchmark, phase_events)
    warmup_request_id = int(benchmark["warmups"][-1]["request_id"])
    final_request_id = int(benchmark["runs"][-1]["request_id"])
    rank_memory = {
        "rank0": validate_rank_memory(rank0_events, warmup_request_id, final_request_id, "rank 0"),
        "rank1": validate_rank_memory(rank1_events, warmup_request_id, final_request_id, "rank 1"),
    }

    host_memory = {
        "head_memavailable_drop_bytes": require_available_memory(
            root / "head-host-before.txt", root / "head-host-after.txt", "head Spark"
        ),
        "worker_memavailable_drop_bytes": require_available_memory(
            root / "worker-host-before.txt", root / "worker-host-after.txt", "worker Spark"
        ),
    }
    for name, label in (("head", "VmRSS:"), ("worker", "VmRSS:"), ("studio", "process_rss_kib=")):
        before_rss = kib_value(root / f"{name}-host-before.txt", label) * 1024
        after_rss = kib_value(root / f"{name}-host-after.txt", label) * 1024
        host_memory[f"{name}_rss_growth_bytes"] = require_growth_within(
            f"{name} RSS", before_rss, after_rss
        )

    for label, state in (("before", before), ("after", after)):
        if state.get("link") != "up" or state.get("inflight"):
            raise ValueError(f"MCDMA was not idle and up {label} benchmark")
        for side in ("mac", "spark"):
            endpoint = state.get(side, {})
            if not endpoint.get("alive") or int(endpoint.get("failures", -1)) != 0:
                raise ValueError(f"MCDMA {side} was unhealthy {label} benchmark")
        if state.get("spark", {}).get("service") != "poll":
            raise ValueError(f"Spark mailbox service was not attached {label} benchmark")
    expected_calls = int(benchmark["expected_mcdma_calls"])
    call_delta = {
        side: int(after[side]["calls"]) - int(before[side]["calls"])
        for side in ("mac", "spark")
    }
    if any(delta != expected_calls for delta in call_delta.values()):
        raise ValueError(f"MCDMA call delta does not equal {expected_calls}: {call_delta}")

    containers = {
        name: parse_container_state((root / f"{name}.txt").read_text(), name)
        for name in ("head-before", "head-after", "worker-before", "worker-after")
    }
    for role in ("head", "worker"):
        if containers[f"{role}-before"] != containers[f"{role}-after"]:
            raise ValueError(f"{role} container identity changed during benchmark")
    if containers["head-before"]["id"] == containers["worker-before"]["id"]:
        raise ValueError("head and worker container identities are not distinct")

    container_environment = {
        role: parse_environment((root / f"{role}-container-env.txt").read_text())
        for role in ("head", "worker")
    }
    deployed_knobs = benchmark["health_before"]["performance_knobs"]
    for role, environment in container_environment.items():
        for key, value in deployed_knobs.items():
            if environment.get(key) != str(value):
                raise ValueError(f"{role} container does not attest deployed knob {key}")

    failure_delta = {
        side: int(after[side]["failures"]) - int(before[side]["failures"])
        for side in ("mac", "spark")
    }

    return {
        "schema": 2,
        "benchmark": benchmark,
        "mcdma_before": before,
        "mcdma_after": after,
        "mcdma_call_delta": call_delta,
        "mcdma_failure_delta": failure_delta,
        "expected_steps": expected_steps,
        "spark_phase_events": phase_events,
        "rank0_commit_events": rank0_events,
        "rank1_commit_events": rank1_events,
        "joined_step_evidence": joined_steps,
        "rank_memory": rank_memory,
        "host_memory": host_memory,
        "containers": containers,
        "host_state": {
            name: (root / f"{name}.txt").read_text()
            for name in (
                "head-host-before", "head-host-after", "worker-host-before",
                "worker-host-after", "studio-host-before", "studio-host-after",
            )
        },
        "container_environment": container_environment,
        "measurement_code_sha256": (root / "measurement-code-sha256.txt").read_text(),
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: benchmark-evidence.py BUNDLE_DIRECTORY")
    try:
        result = validate_bundle(Path(sys.argv[1]))
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
