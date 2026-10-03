"""Fail-closed evaluators for restart-separated three-machine experiments."""

from __future__ import annotations

import hashlib
import math
import statistics
import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .evaluator import ManifestError, _canonical, _changed_settings


SCHEMA_VERSION = 2
SCREEN_SEQUENCE = tuple("ABBABAAB")
EXTENDED_SEQUENCE = tuple("ABBABAABABBA")
CONFIRMATION_SEQUENCE = tuple("BAAB")
BENCHMARK_SHAPES = ("B1", "B2", "B3-64", "B3-512", "B3-2040", "B4")
TTFT_SHAPES = ("B3-64", "B3-512", "B3-2040")
EQUIVALENCE_SHAPES = tuple(f"E{index}" for index in range(1, 9))
CONFIRMATION_SHAPES = (*BENCHMARK_SHAPES, *EQUIVALENCE_SHAPES, "idle")
HARD_GATES = (
    "reference_tokens", "identity", "mcdma_calls", "rank0_steps",
    "rank1_commits", "zero_failures", "memory", "tail", "containers",
    "container_environment",
)
IDENTITY_FIELDS = (
    "deployment_id", "commit", "project_manifest", "config_sha256", "image_id",
    "link_generation",
)
REQUIRED_PRIMARY_METRIC = "decode_step_median_s"
REQUIRED_DIRECTION = "lower"
REQUIRED_MINIMUM_GAIN_FRACTION = 0.03
REQUIRED_MAX_SECONDARY_REGRESSION_FRACTION = 0.02
REQUIRED_SECONDARY_METRICS = {"ttft_s": "lower"}


def _keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ManifestError(
            f"{label} keys must be exactly {sorted(expected)}; got {sorted(actual)}"
        )


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ManifestError(f"{label} must be finite{' and positive' if positive else ''}")
    return number


def _sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ManifestError(f"{label} must be a lowercase SHA-256")


def _median(value: Any, label: str) -> float:
    if isinstance(value, Mapping):
        if "median" not in value:
            raise ManifestError(f"{label} summary must contain median")
        value = value["median"]
    return _number(value, label, positive=True)


def _health(evidence: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = evidence.get(name)
    if not isinstance(value, Mapping):
        raise ManifestError(f"benchmark evidence must contain {name}")
    return value


def _container_ids(
    evidence: Mapping[str, Any], health: Mapping[str, Any], when: str
) -> tuple[str, ...]:
    raw = evidence.get(f"container_ids_{when}", health.get("container_ids"))
    if raw is None:
        raw = health.get("containers")
    states: list[Any] = []
    if isinstance(raw, Mapping):
        values = list(raw.values())
        if values and all(isinstance(value, Mapping) for value in values):
            states = [value.get("state") for value in values]
            values = [value.get("id") for value in values]
        raw = values
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ManifestError("benchmark evidence must identify exactly two containers")
    ids = tuple(sorted(raw)) if all(isinstance(value, str) for value in raw) else ()
    if len(ids) != 2 or any(not value for value in ids) or len(set(ids)) != 2:
        raise ManifestError("container IDs must be two distinct non-empty strings")
    if states and any(state not in {"running", "ready"} for state in states):
        raise ManifestError("both benchmark containers must be running")
    return ids


def _restart_identity(evidence: Mapping[str, Any]) -> dict[str, Any]:
    before = _health(evidence, "health_before")
    after = _health(evidence, "health_after")
    generation = before.get("link_generation")
    if type(generation) is not int or generation <= 0:
        raise ManifestError("benchmark link_generation must be a positive integer")
    if after.get("link_generation") != generation:
        raise ManifestError("benchmark link_generation changed within a round")
    before_ids = _container_ids(evidence, before, "before")
    after_ids = _container_ids(evidence, after, "after")
    if after_ids != before_ids:
        raise ManifestError("benchmark container IDs changed within a round")
    return {"link_generation": generation, "container_ids": list(before_ids)}


def _restart_id(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(dict(identity))).hexdigest()


def _timing_and_knobs(evidence: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    before = _health(evidence, "health_before")
    after = _health(evidence, "health_after")
    before_knobs = before.get("performance_knobs")
    after_knobs = after.get("performance_knobs")
    if not isinstance(before_knobs, dict) or not before_knobs:
        raise ManifestError("benchmark health must contain performance_knobs")
    if after_knobs != before_knobs:
        raise ManifestError("performance knobs changed within a benchmark round")
    timing = evidence.get("timing_mode", before.get("timing_mode"))
    if timing is None:
        phase_timing = before_knobs.get("THREE_MACHINE_PHASE_TIMING")
        timing = "synchronized" if str(phase_timing) == "1" else "host-enqueue"
    if not isinstance(timing, str) or not timing:
        raise ManifestError("benchmark timing_mode must be a non-empty string")
    return timing, dict(before_knobs)


def _counter(evidence: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in evidence:
            return evidence[name]
    return None


def _bundle_container_ids(evidence: Mapping[str, Any], when: str) -> list[str]:
    containers = evidence.get("containers")
    if not isinstance(containers, Mapping):
        raise ManifestError("benchmark bundle must contain container evidence")
    ids: list[str] = []
    for role in ("head", "worker"):
        state = containers.get(f"{role}-{when}")
        if not isinstance(state, Mapping):
            raise ManifestError(f"benchmark bundle is missing {role}-{when} container state")
        container_id = state.get("id")
        if (
            not isinstance(container_id, str) or len(container_id) != 64
            or any(char not in "0123456789abcdef" for char in container_id)
            or state.get("running") is not True or state.get("oom_killed") is not False
            or state.get("exit_code") != 0
        ):
            raise ManifestError(f"benchmark bundle {role}-{when} container is not cleanly running")
        ids.append(container_id)
    if len(set(ids)) != 2:
        raise ManifestError("benchmark bundle must identify two distinct containers")
    return sorted(ids)


def _event_keys(values: Any, label: str) -> set[tuple[int, int]]:
    if not isinstance(values, list):
        raise ManifestError(f"{label} must be a list")
    keys: set[tuple[int, int]] = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise ManifestError(f"{label} entries must be objects")
        request_id = value.get("request_id")
        step = value.get("step")
        if type(request_id) is not int or request_id < 0 or type(step) is not int or step < 0:
            raise ManifestError(f"{label} has malformed request/step identity")
        key = (request_id, step)
        if key in keys:
            raise ManifestError(f"{label} has duplicate request/step identity")
        keys.add(key)
    return keys


def _measured_step_keys(requests: Any, label: str) -> set[tuple[int, int]]:
    if not isinstance(requests, list) or not requests:
        raise ManifestError(f"{label} must be a non-empty list")
    keys: set[tuple[int, int]] = set()
    for request in requests:
        if not isinstance(request, Mapping):
            raise ManifestError(f"{label} entries must be objects")
        request_id = request.get("request_id")
        steps = request.get("steps")
        if type(request_id) is not int or request_id < 0 \
                or not isinstance(steps, list) or not steps:
            raise ManifestError(f"{label} has malformed request evidence")
        for step in steps:
            if not isinstance(step, Mapping):
                raise ManifestError(f"{label} step entries must be objects")
            step_id = step.get("step")
            if type(step_id) is not int or step_id < 0:
                raise ManifestError(f"{label} has malformed step identity")
            key = (request_id, step_id)
            if key in keys:
                raise ManifestError(f"{label} has duplicate request/step identity")
            keys.add(key)
    return keys


def _counter_pair(value: Any, label: str) -> int:
    if not isinstance(value, Mapping) or set(value) != {"mac", "spark"}:
        raise ManifestError(f"{label} must contain exactly mac and spark")
    mac = value.get("mac")
    spark = value.get("spark")
    if (
        type(mac) is not int or type(spark) is not int
        or mac < 0 or spark < 0 or mac != spark
    ):
        raise ManifestError(f"{label} must contain equal non-negative integer counters")
    return mac


def _validated_mcdma_measurement(
    evidence: Mapping[str, Any], expected_calls: int, label: str, *,
    require_poll_service: bool,
) -> tuple[int, int]:
    if type(expected_calls) is not int or expected_calls < 0:
        raise ManifestError(f"{label} expected MCDMA calls must be a non-negative integer")
    states: dict[str, Mapping[str, Any]] = {}
    for when in ("before", "after"):
        state = evidence.get(f"mcdma_{when}")
        if (
            not isinstance(state, Mapping)
            or state.get("link") != "up"
            or bool(state.get("inflight"))
        ):
            raise ManifestError(f"{label} MCDMA link was not up and idle {when}")
        for side in ("mac", "spark"):
            peer = state.get(side)
            if (
                not isinstance(peer, Mapping)
                or peer.get("alive") is not True
                or type(peer.get("calls")) is not int
                or peer["calls"] < 0
                or type(peer.get("failures")) is not int
                or peer["failures"] < 0
            ):
                raise ManifestError(f"{label} MCDMA {side} peer was unhealthy {when}")
        if require_poll_service and state["spark"].get("service") != "poll":
            raise ManifestError(f"{label} Spark MCDMA mailbox service was not attached {when}")
        states[when] = state
    calls = {
        side: states["after"][side]["calls"] - states["before"][side]["calls"]
        for side in ("mac", "spark")
    }
    failures = {
        side: states["after"][side]["failures"] - states["before"][side]["failures"]
        for side in ("mac", "spark")
    }
    if any(value != expected_calls for value in calls.values()):
        raise ManifestError(f"{label} MCDMA call counters do not match measured work")
    if any(value != 0 for value in failures.values()):
        raise ManifestError(f"{label} MCDMA failure counters changed")
    return expected_calls, 0


def _memory_bundle_pass(evidence: Mapping[str, Any]) -> bool:
    rank_memory = evidence.get("rank_memory")
    host_memory = evidence.get("host_memory")
    if not isinstance(rank_memory, Mapping) or set(rank_memory) != {"rank0", "rank1"}:
        return False
    if not isinstance(host_memory, Mapping) or not host_memory:
        return False
    for rank in ("rank0", "rank1"):
        proof = rank_memory.get(rank)
        if not isinstance(proof, Mapping):
            return False
        baseline = proof.get("post_warmup")
        final = proof.get("final")
        growth = proof.get("growth_bytes")
        if not all(isinstance(value, Mapping) for value in (baseline, final, growth)):
            return False
        for name, base in baseline.items():
            end = final.get(name)
            delta = growth.get(name)
            if type(base) is not int or type(end) is not int or type(delta) is not int:
                return False
            if delta != end - base or delta > max(64 << 20, int(base * 0.01)):
                return False
    return all(
        type(value) is int and value <= 64 << 20
        for value in host_memory.values()
    )


def _container_environment_pass(
    evidence: Mapping[str, Any], knobs: Mapping[str, Any]
) -> bool:
    environments = evidence.get("container_environment")
    if not isinstance(environments, Mapping) or set(environments) != {"head", "worker"}:
        return False
    for role in ("head", "worker"):
        environment = environments.get(role)
        if not isinstance(environment, Mapping):
            return False
        if any(environment.get(key) != str(value) for key, value in knobs.items()):
            return False
    return True


def normalize_benchmark_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one validated wrapper bundle into evaluator input."""
    if not isinstance(evidence, Mapping) or evidence.get("schema") != 2:
        raise ManifestError("benchmark evidence must be a schema-2 wrapper bundle")
    benchmark = evidence.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise ManifestError("benchmark bundle must contain benchmark output")
    result = copy.deepcopy(dict(benchmark))
    result["container_ids_before"] = _bundle_container_ids(evidence, "before")
    result["container_ids_after"] = _bundle_container_ids(evidence, "after")
    reported_calls = _counter_pair(
        evidence.get("mcdma_call_delta"), "mcdma_call_delta"
    )
    reported_failures = _counter_pair(
        evidence.get("mcdma_failure_delta"), "mcdma_failure_delta"
    )
    all_runs = result.get("warmups", []) + result.get("runs", [])
    expected_keys = _measured_step_keys(all_runs, "benchmark runs")
    derived_calls = 0
    for run in all_runs:
        observed = run.get("expected_mcdma_calls")
        required = 2 + 2 * len(run["steps"])
        if type(observed) is not int or observed != required:
            raise ManifestError("benchmark request MCDMA calls are inconsistent with its steps")
        derived_calls += required
    if result.get("expected_mcdma_calls") != derived_calls:
        raise ManifestError("benchmark expected MCDMA calls do not match request evidence")
    measured_calls, measured_failures = _validated_mcdma_measurement(
        evidence, derived_calls, "benchmark", require_poll_service=True
    )
    if reported_calls != measured_calls or reported_failures != measured_failures:
        raise ManifestError("benchmark reported MCDMA deltas differ from raw counters")
    result["mcdma_call_delta"] = measured_calls
    result["mcdma_failure_delta"] = measured_failures
    expected_steps = evidence.get("expected_steps")
    if type(expected_steps) is not int or expected_steps != len(expected_keys):
        raise ManifestError("benchmark expected_steps does not match request evidence")
    phase_keys = _event_keys(evidence.get("spark_phase_events"), "Spark phase events")
    rank0_keys = _event_keys(evidence.get("rank0_commit_events"), "rank 0 commits")
    rank1_keys = _event_keys(evidence.get("rank1_commit_events"), "rank 1 commits")
    joined = evidence.get("joined_step_evidence")
    joined_keys = _event_keys(joined, "joined step evidence")
    if any(keys != expected_keys for keys in (phase_keys, rank0_keys, rank1_keys, joined_keys)):
        raise ManifestError("benchmark event identities do not match every request step")
    result["expected_steps"] = expected_steps
    result["rank0_steps"] = len(rank0_keys)
    result["rank1_commits"] = len(rank1_keys)
    result["bundle_memory_pass"] = _memory_bundle_pass(evidence)
    knobs = (_health(result, "health_before").get("performance_knobs") or {})
    result["container_environment_pass"] = (
        isinstance(knobs, Mapping) and _container_environment_pass(evidence, knobs)
    )
    return result


def _reference_tokens_pass(evidence: Mapping[str, Any]) -> bool:
    token_hash = evidence.get("token_ids_sha256")
    reference_hash = evidence.get("reference_token_ids_sha256")
    try:
        _sha256(token_hash, "benchmark token_ids_sha256")
        _sha256(reference_hash, "benchmark reference_token_ids_sha256")
    except ManifestError:
        return False
    return token_hash == reference_hash


def _derive_hard_gates(evidence: Mapping[str, Any]) -> dict[str, bool]:
    before = _health(evidence, "health_before")
    after = _health(evidence, "health_after")
    try:
        before_ids = _container_ids(evidence, before, "before")
        containers = before_ids == _container_ids(evidence, after, "after")
    except ManifestError:
        containers = False

    identity = all(
        before.get(field) not in (None, "", "unknown")
        and before.get(field) == after.get(field)
        for field in IDENTITY_FIELDS
    )
    expected_calls = _counter(evidence, "expected_mcdma_calls")
    actual_calls = _counter(evidence, "mcdma_call_delta", "mcdma_calls_delta")
    expected_steps = _counter(evidence, "expected_steps", "rank0_expected_steps")
    rank0_steps = _counter(evidence, "rank0_steps", "rank0_step_delta")
    rank1_commits = _counter(evidence, "rank1_commits", "rank1_commit_delta")

    zero_failures = (
        before.get("failed_requests") == 0
        and after.get("failed_requests") == 0
        and _counter(evidence, "mcdma_failure_delta", "mcdma_failures") == 0
    )
    growth = evidence.get("mlx_post_warmup_growth_bytes")
    warm_memory = (_health(evidence, "health_after_warmup").get("memory") or {})
    warm_held = 0
    if isinstance(warm_memory, Mapping):
        for key in ("mlx_active_bytes", "mlx_cache_bytes"):
            value = warm_memory.get(key)
            if type(value) is int:
                warm_held += value
    memory_limit = max(64 << 20, int(warm_held * 0.01))
    memory = (
        type(growth) is int and growth <= memory_limit
        and evidence.get("bundle_memory_pass") is True
    )

    runs = evidence.get("runs")
    tail = isinstance(runs, list) and bool(runs)
    if tail:
        for run in runs:
            summary = run.get("decode_step_s") if isinstance(run, Mapping) else None
            if not isinstance(summary, Mapping):
                tail = False
                break
            median = summary.get("median")
            p99 = summary.get("p99")
            if (
                not isinstance(median, (int, float)) or isinstance(median, bool)
                or not isinstance(p99, (int, float)) or isinstance(p99, bool)
                or float(median) <= 0
                or float(p99) > 1.3 * float(median)
                or run.get("stall_count") != 0
                or not isinstance(run.get("hiccup_count"), int)
                or run["hiccup_count"] > 2
            ):
                tail = False
                break

    return {
        "reference_tokens": _reference_tokens_pass(evidence),
        "identity": identity,
        "mcdma_calls": type(expected_calls) is int and type(actual_calls) is int and actual_calls == expected_calls,
        "rank0_steps": type(expected_steps) is int and type(rank0_steps) is int and rank0_steps == expected_steps,
        "rank1_commits": type(expected_steps) is int and type(rank1_commits) is int and rank1_commits == expected_steps,
        "zero_failures": zero_failures,
        "memory": memory,
        "tail": tail,
        "containers": containers,
        "container_environment": evidence.get("container_environment_pass") is True,
    }


def build_round_from_evidence(
    arm: str,
    evidence: Mapping[str, Any],
    *,
    primary_metric: str = "decode_step_median_s",
    secondary_metrics: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a screen round from raw benchmark JSON rather than typed summaries."""
    if arm not in {"A", "B"}:
        raise ManifestError("round arm must be A or B")
    if not isinstance(evidence, Mapping):
        raise ManifestError("benchmark evidence must be an object")
    evidence = normalize_benchmark_evidence(evidence)
    warmups = evidence.get("warmups")
    runs = evidence.get("runs")
    if not isinstance(warmups, list) or not isinstance(runs, list):
        raise ManifestError("benchmark evidence must contain warmups and runs")
    identity = _restart_identity(evidence)
    timing_mode, knobs = _timing_and_knobs(evidence)
    if secondary_metrics is None:
        secondary_metrics = {"ttft_s": "first_token_s"}
    secondary: dict[str, float] = {}
    for name, evidence_name in secondary_metrics.items():
        if not isinstance(name, str) or not name or not isinstance(evidence_name, str):
            raise ManifestError("secondary metric map must contain non-empty strings")
        secondary[name] = _median(evidence.get(evidence_name), evidence_name)
    return {
        "arm": arm,
        "restart_id": _restart_id(identity),
        "restart_identity": identity,
        "timing_mode": timing_mode,
        "performance_knobs": knobs,
        "warmup_count": len(warmups),
        "measured_count": len(runs),
        "primary": _median(evidence.get(primary_metric), primary_metric),
        "secondary": secondary,
        "hard_gates": _derive_hard_gates(evidence),
        "receipt_sha256": hashlib.sha256(_canonical(dict(evidence))).hexdigest(),
    }


def _validate_restart_proof(
    round_: Mapping[str, Any], index: int, expected_knobs: Mapping[str, Any]
) -> None:
    identity = round_["restart_identity"]
    if not isinstance(identity, dict):
        raise ManifestError(f"round {index} restart_identity must be an object")
    _keys(identity, {"link_generation", "container_ids"}, f"round {index} restart_identity")
    generation = identity["link_generation"]
    container_ids = identity["container_ids"]
    if type(generation) is not int or generation <= 0:
        raise ManifestError("restart link_generation must be a positive integer")
    if (
        not isinstance(container_ids, list) or len(container_ids) != 2
        or any(not isinstance(value, str) or not value for value in container_ids)
        or len(set(container_ids)) != 2 or container_ids != sorted(container_ids)
    ):
        raise ManifestError("restart_identity must contain two sorted distinct container IDs")
    if round_["restart_id"] != _restart_id(identity):
        raise ManifestError("restart_id must be derived from generation and container IDs")
    if not isinstance(round_["timing_mode"], str) or not round_["timing_mode"]:
        raise ManifestError("timing_mode must be a non-empty string")
    if round_["performance_knobs"] != expected_knobs:
        raise ManifestError("round performance knobs do not match its selected arm")


def _validate_common_rounds(
    rounds: Any,
    sequence: tuple[str, ...],
    baseline_settings: Mapping[str, Any],
    candidate_settings: Mapping[str, Any],
    round_keys: set[str],
) -> None:
    if not isinstance(rounds, list) or len(rounds) != len(sequence):
        raise ManifestError(f"rounds must contain exactly {len(sequence)} rounds")
    actual_sequence = tuple(item.get("arm") for item in rounds if isinstance(item, dict))
    if actual_sequence != sequence:
        raise ManifestError(f"round arms must be {''.join(sequence)}")
    restart_ids: set[str] = set()
    restart_container_ids: set[tuple[str, ...]] = set()
    receipt_hashes: set[str] = set()
    timing_modes: set[str] = set()
    for index, round_ in enumerate(rounds):
        if not isinstance(round_, dict):
            raise ManifestError(f"round {index} must be an object")
        _keys(round_, round_keys, f"round {index}")
        expected_knobs = baseline_settings if round_["arm"] == "A" else candidate_settings
        _validate_restart_proof(round_, index, expected_knobs)
        restart_id = round_["restart_id"]
        if restart_id in restart_ids:
            raise ManifestError("restart identity must change between every round")
        restart_ids.add(restart_id)
        container_identity = tuple(round_["restart_identity"]["container_ids"])
        if container_identity in restart_container_ids:
            raise ManifestError("both Spark containers must be recreated between every round")
        restart_container_ids.add(container_identity)
        _sha256(round_["receipt_sha256"], f"round {index} receipt_sha256")
        if round_["receipt_sha256"] in receipt_hashes:
            raise ManifestError("round receipt hashes must be unique")
        receipt_hashes.add(round_["receipt_sha256"])
        timing_modes.add(round_["timing_mode"])
        gates = round_["hard_gates"]
        if not isinstance(gates, dict) or set(gates) != set(HARD_GATES):
            raise ManifestError(f"hard_gates must contain exactly {HARD_GATES}")
        if any(type(value) is not bool for value in gates.values()):
            raise ManifestError("hard gates must be booleans")
    if len(timing_modes) != 1:
        raise ManifestError("every round must use the same timing mode")


def materialize_campaign(campaign: dict[str, Any]) -> dict[str, Any]:
    """Derive every screen value and hard gate from raw wrapper evidence."""
    _keys(campaign, {
        "schema_version", "campaign_id", "baseline_settings", "candidate_settings",
        "design", "baseline_precondition_evidence", "identity", "source_rounds",
    }, "campaign")
    if campaign.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"campaign schema_version must be {SCHEMA_VERSION}")
    baseline_sources = campaign.get("baseline_precondition_evidence")
    if not isinstance(baseline_sources, list) or len(baseline_sources) != 3:
        raise ManifestError("baseline_precondition_evidence must contain three bundles")
    baseline_rounds = [
        build_round_from_evidence("A", evidence) for evidence in baseline_sources
    ]
    _validate_common_rounds(
        baseline_rounds,
        ("A", "A", "A"),
        campaign["baseline_settings"],
        campaign["candidate_settings"],
        {
            "arm", "restart_id", "restart_identity", "timing_mode", "performance_knobs",
            "warmup_count", "measured_count", "primary", "secondary", "hard_gates",
            "receipt_sha256",
        },
    )
    if any(not all(round_["hard_gates"].values()) for round_ in baseline_rounds):
        raise ManifestError("baseline precondition evidence failed a hard gate")
    if any(round_["warmup_count"] != 1 or round_["measured_count"] != 5
           for round_ in baseline_rounds):
        raise ManifestError("baseline precondition requires one warmup and five measured requests")

    sources = campaign.get("source_rounds")
    if not isinstance(sources, list) or len(sources) not in (8, 12):
        raise ManifestError("source_rounds must contain the 8-round screen or 12-round extension")
    rounds: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ManifestError(f"source round {index} must be an object")
        _keys(source, {"arm", "evidence"}, f"source round {index}")
        rounds.append(build_round_from_evidence(source["arm"], source["evidence"]))

    materialized = {
        "schema_version": campaign["schema_version"],
        "campaign_id": campaign["campaign_id"],
        "baseline_settings": copy.deepcopy(campaign["baseline_settings"]),
        "candidate_settings": copy.deepcopy(campaign["candidate_settings"]),
        "design": copy.deepcopy(campaign["design"]),
        "baseline_precondition": [round_["primary"] for round_ in baseline_rounds],
        "identity": copy.deepcopy(campaign["identity"]),
        "rounds": rounds,
    }
    _validate_materialized_campaign(materialized)
    return materialized


def validate_campaign(campaign: dict[str, Any]) -> None:
    materialize_campaign(campaign)


def _validate_materialized_campaign(campaign: dict[str, Any]) -> None:
    _keys(campaign, {
        "schema_version", "campaign_id", "baseline_settings", "candidate_settings",
        "design", "baseline_precondition", "identity", "rounds",
    }, "campaign")
    if type(campaign["schema_version"]) is not int or campaign["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(f"campaign schema_version must be {SCHEMA_VERSION}")
    if not isinstance(campaign["campaign_id"], str) or not campaign["campaign_id"]:
        raise ManifestError("campaign_id must be a non-empty string")
    baseline_settings = campaign["baseline_settings"]
    candidate_settings = campaign["candidate_settings"]
    if not isinstance(baseline_settings, dict) or not baseline_settings:
        raise ManifestError("baseline_settings must be a non-empty object")
    if not isinstance(candidate_settings, dict):
        raise ManifestError("candidate_settings must be an object")
    if len(_changed_settings(baseline_settings, candidate_settings)) != 1:
        raise ManifestError("campaign must change exactly one setting")

    design = campaign["design"]
    if not isinstance(design, dict):
        raise ManifestError("design must be an object")
    _keys(design, {
        "primary_metric", "direction", "minimum_gain_fraction", "confidence_multiplier",
        "max_secondary_regression_fraction", "secondary_metrics",
    }, "design")
    if design["primary_metric"] != REQUIRED_PRIMARY_METRIC:
        raise ManifestError(f"primary_metric must be {REQUIRED_PRIMARY_METRIC}")
    if design["direction"] != REQUIRED_DIRECTION:
        raise ManifestError(f"direction must be {REQUIRED_DIRECTION}")
    minimum = _number(design["minimum_gain_fraction"], "minimum_gain_fraction")
    regression = _number(design["max_secondary_regression_fraction"], "max_secondary_regression_fraction")
    if minimum != REQUIRED_MINIMUM_GAIN_FRACTION:
        raise ManifestError(
            f"minimum_gain_fraction must be {REQUIRED_MINIMUM_GAIN_FRACTION}"
        )
    if regression != REQUIRED_MAX_SECONDARY_REGRESSION_FRACTION:
        raise ManifestError(
            "max_secondary_regression_fraction must be "
            f"{REQUIRED_MAX_SECONDARY_REGRESSION_FRACTION}"
        )
    rounds = campaign["rounds"]
    if not isinstance(rounds, list) or len(rounds) not in (8, 12):
        raise ManifestError("rounds must contain the 8-round screen or one 12-round extension")
    expected_confidence = 2.353 if len(rounds) == 8 else 2.015
    confidence = _number(design["confidence_multiplier"], "confidence_multiplier", positive=True)
    if confidence != expected_confidence:
        raise ManifestError(f"confidence_multiplier must be {expected_confidence}")
    secondary = design["secondary_metrics"]
    if secondary != REQUIRED_SECONDARY_METRICS:
        raise ManifestError(f"secondary_metrics must be exactly {REQUIRED_SECONDARY_METRICS}")

    baseline = campaign["baseline_precondition"]
    if not isinstance(baseline, list) or len(baseline) != 3:
        raise ManifestError("baseline_precondition must contain three restart-separated medians")
    baseline_values = [_number(value, f"baseline_precondition[{index}]", positive=True) for index, value in enumerate(baseline)]
    if max(baseline_values) / min(baseline_values) - 1 > 0.02:
        raise ManifestError("baseline precondition differs by more than 2%")

    identity = campaign["identity"]
    if not isinstance(identity, dict):
        raise ManifestError("identity must be an object")
    _keys(identity, {"measurement_code_sha256", "timer_code_sha256", "reference_file_sha256"}, "identity")
    for key, value in identity.items():
        _sha256(value, f"identity.{key}")

    expected_sequence = SCREEN_SEQUENCE if len(rounds) == 8 else EXTENDED_SEQUENCE
    _validate_common_rounds(rounds, expected_sequence, baseline_settings, candidate_settings, {
        "arm", "restart_id", "restart_identity", "timing_mode", "performance_knobs",
        "warmup_count", "measured_count", "primary", "secondary", "hard_gates",
        "receipt_sha256",
    })
    for index, round_ in enumerate(rounds):
        if round_["warmup_count"] != 1 or round_["measured_count"] != 5:
            raise ManifestError("every round must have one warmup and five measured requests")
        _number(round_["primary"], f"round {index} primary", positive=True)
        observed_secondary = round_["secondary"]
        if not isinstance(observed_secondary, dict) or set(observed_secondary) != set(secondary):
            raise ManifestError("each round must contain every secondary metric exactly once")
        for name, value in observed_secondary.items():
            _number(value, f"round {index} secondary {name}", positive=True)


def _gain(a: float, b: float, direction: str) -> float:
    return math.log(a / b) if direction == "lower" else math.log(b / a)


def _fraction(log_gain: float) -> float:
    return 1.0 - math.exp(-log_gain)


def _pairs(rounds: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    paired = []
    for index in range(0, len(rounds), 2):
        first, second = rounds[index:index + 2]
        a = first if first["arm"] == "A" else second
        b = second if first["arm"] == "A" else first
        paired.append((a, b))
    return paired


def evaluate_campaign(campaign: dict[str, Any]) -> dict[str, Any]:
    source_campaign = campaign
    campaign = materialize_campaign(source_campaign)
    design = campaign["design"]
    rounds = campaign["rounds"]
    reasons: list[str] = []
    failed_gates = sorted({name for round_ in rounds for name, passed in round_["hard_gates"].items() if not passed})
    reasons.extend(f"hard gate failed: {name}" for name in failed_gates)

    paired_log_gains = [_gain(float(a["primary"]), float(b["primary"]), design["direction"]) for a, b in _pairs(rounds)]
    pair_gains = [_fraction(value) for value in paired_log_gains]
    if any(value <= 0 for value in paired_log_gains):
        reasons.append("not every pair favors the candidate")
    mean_log_gain = statistics.mean(paired_log_gains)
    standard_error = statistics.stdev(paired_log_gains) / math.sqrt(len(paired_log_gains))
    lower_log_gain = mean_log_gain - float(design["confidence_multiplier"]) * standard_error
    mean_gain = _fraction(mean_log_gain)
    lower_gain = _fraction(lower_log_gain)

    secondary_regressions: dict[str, float] = {}
    for metric, direction in design["secondary_metrics"].items():
        paired_regressions = []
        for a, b in _pairs(rounds):
            a_value = float(a["secondary"][metric])
            b_value = float(b["secondary"][metric])
            paired_regressions.append(b_value / a_value - 1 if direction == "lower" else a_value / b_value - 1)
        regression = statistics.mean(paired_regressions)
        secondary_regressions[metric] = regression
        if regression > float(design["max_secondary_regression_fraction"]):
            reasons.append(f"secondary regression: {metric}")

    threshold = float(design["minimum_gain_fraction"])
    verdict = "screen_pass"
    if reasons:
        verdict = "reject"
    elif lower_gain < threshold:
        if len(rounds) == 8 and mean_gain >= threshold:
            verdict = "extend_once"
            reasons.append("confidence bound not met; run the one allowed extension")
        else:
            verdict = "reject"
            reasons.append("paired performance threshold not met")

    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": hashlib.sha256(_canonical(source_campaign)).hexdigest(),
        "materialized_campaign_sha256": hashlib.sha256(_canonical(campaign)).hexdigest(),
        "baseline_settings_sha256": hashlib.sha256(
            _canonical(campaign["baseline_settings"])
        ).hexdigest(),
        "candidate_settings_sha256": hashlib.sha256(
            _canonical(campaign["candidate_settings"])
        ).hexdigest(),
        "changed_settings": _changed_settings(campaign["baseline_settings"], campaign["candidate_settings"]),
        "pair_gains_fraction": pair_gains,
        "mean_gain_fraction": mean_gain,
        "lower_confidence_gain_fraction": lower_gain,
        "secondary_regressions_fraction": secondary_regressions,
        "hard_gates_checked_before_performance": True,
        "failed_gates": failed_gates,
        "verdict": verdict,
        "final_acceptance": False,
        "stop": bool(failed_gates),
        "reasons": reasons,
    }


def _shape_value(value: Any, label: str) -> float:
    if isinstance(value, Mapping):
        for name in ("value", "median", "primary"):
            if name in value:
                return _number(value[name], label, positive=True)
    return _number(value, label, positive=True)


def _b2_final_128(value: Any, label: str) -> float:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{label} must contain final_128")
    for name in ("final_128", "final_128_median", "final_128_decode_step_median_s"):
        if name in value:
            return _median(value[name], f"{label}.{name}")
    raise ManifestError(f"{label} must contain final_128")


def normalize_quality_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or evidence.get("schema") != 2:
        raise ManifestError("quality evidence must be a schema-2 wrapper bundle")
    quality = evidence.get("quality")
    if not isinstance(quality, Mapping):
        raise ManifestError("quality bundle must contain quality output")
    result = copy.deepcopy(dict(quality))
    result["container_ids_before"] = _bundle_container_ids(evidence, "before")
    result["container_ids_after"] = _bundle_container_ids(evidence, "after")
    reported_calls = _counter_pair(
        evidence.get("mcdma_call_delta"), "quality mcdma_call_delta"
    )
    reported_failures = _counter_pair(
        evidence.get("mcdma_failure_delta"), "quality mcdma_failure_delta"
    )
    results = quality.get("results")
    if not isinstance(results, list) or not results:
        raise ManifestError("quality results must be a non-empty list")
    completions: list[Any] = []
    for task in results:
        if not isinstance(task, Mapping) or not isinstance(task.get("completions"), list):
            raise ManifestError("quality tasks must contain completion evidence")
        completions.extend(task["completions"])
    expected_keys = _measured_step_keys(completions, "quality completions")
    derived_calls = sum(2 + 2 * len(completion["steps"]) for completion in completions)
    if quality.get("expected_mcdma_calls") != derived_calls:
        raise ManifestError("quality expected MCDMA calls do not match completion evidence")
    measured_calls, measured_failures = _validated_mcdma_measurement(
        evidence, derived_calls, "quality", require_poll_service=False
    )
    if reported_calls != measured_calls or reported_failures != measured_failures:
        raise ManifestError("quality reported MCDMA deltas differ from raw counters")
    expected_steps = evidence.get("expected_steps")
    if type(expected_steps) is not int or expected_steps != len(expected_keys):
        raise ManifestError("quality expected_steps does not match request evidence")
    for name, label in (
        ("spark_phase_events", "quality Spark phase events"),
        ("rank0_commit_events", "quality rank 0 commits"),
        ("rank1_commit_events", "quality rank 1 commits"),
    ):
        if _event_keys(evidence.get(name), label) != expected_keys:
            raise ManifestError(f"{label} do not match every request step")
    before = _health(result, "health_before")
    after = _health(result, "health_after")
    if before.get("failed_requests") != 0 or after.get("failed_requests") != 0:
        raise ManifestError("quality evidence has failed requests")
    return result


def _quality_equivalence(quality: Mapping[str, Any]) -> dict[str, bool]:
    results = quality.get("results")
    if not isinstance(results, list):
        raise ManifestError("quality evidence results must be a list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in results:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise ManifestError("quality result is malformed")
        if item["id"] in by_id:
            raise ManifestError("quality result IDs must be unique")
        by_id[item["id"]] = item
    if set(by_id) != set(EQUIVALENCE_SHAPES):
        raise ManifestError(f"quality results must contain exactly {EQUIVALENCE_SHAPES}")
    outcome: dict[str, bool] = {}
    for shape, item in by_id.items():
        completions = item.get("completions")
        if not isinstance(completions, list) or len(completions) < 2:
            raise ManifestError(f"quality {shape} requires at least two completions")
        outcome[shape] = all(
            isinstance(completion, Mapping)
            and completion.get("token_ids_sha256")
            == completion.get("reference_token_ids_sha256")
            for completion in completions
        )
    return outcome


def _idle_values(
    evidence: Mapping[str, Any], first: Mapping[str, Any]
) -> dict[str, float]:
    if not isinstance(evidence, Mapping) or evidence.get("schema") != 1:
        raise ManifestError("idle evidence must be a schema-1 measured bundle")
    _keys(dict(evidence), {
        "schema", "health_before", "health_after", "containers", "steady_runs_s",
        "idle_seconds", "first_after_idle_s", "second_after_idle_s",
        "mcdma_before", "mcdma_after", "mcdma_call_delta", "expected_mcdma_calls",
        "spark_phase_events", "rank0_commit_events", "rank1_commit_events",
        "expected_steps",
    }, "idle evidence")
    normalized = dict(evidence)
    normalized["container_ids_before"] = _bundle_container_ids(evidence, "before")
    normalized["container_ids_after"] = _bundle_container_ids(evidence, "after")
    identity = _restart_identity(normalized)
    timing_mode, knobs = _timing_and_knobs(normalized)
    if (
        _restart_id(identity) != first["restart_id"]
        or timing_mode != first["timing_mode"]
        or knobs != first["performance_knobs"]
    ):
        raise ManifestError("idle evidence was not measured in the same restart and mode")
    before = _health(normalized, "health_before")
    after = _health(normalized, "health_after")
    if before.get("failed_requests") != 0 or after.get("failed_requests") != 0:
        raise ManifestError("idle evidence has failed requests")
    reported_calls = _counter_pair(
        evidence.get("mcdma_call_delta"), "idle mcdma_call_delta"
    )
    expected_calls = evidence.get("expected_mcdma_calls")
    if type(expected_calls) is not int or expected_calls < 0:
        raise ManifestError("idle MCDMA counters do not match the measured work")
    expected_steps = evidence.get("expected_steps")
    if type(expected_steps) is not int or expected_steps <= 0:
        raise ManifestError("idle expected_steps must be a positive integer")
    if expected_calls != 10 + 2 * expected_steps:
        raise ManifestError("idle MCDMA counters are inconsistent with five measured requests")
    measured_calls, _ = _validated_mcdma_measurement(
        evidence, expected_calls, "idle", require_poll_service=False
    )
    if reported_calls != measured_calls:
        raise ManifestError("idle reported MCDMA deltas differ from raw counters")
    phase_keys = _event_keys(evidence.get("spark_phase_events"), "idle Spark phase events")
    rank0_keys = _event_keys(evidence.get("rank0_commit_events"), "idle rank 0 commits")
    rank1_keys = _event_keys(evidence.get("rank1_commit_events"), "idle rank 1 commits")
    if (
        len(phase_keys) != expected_steps
        or phase_keys != rank0_keys
        or phase_keys != rank1_keys
    ):
        raise ManifestError("idle event identities do not match every measured step")
    steady = evidence.get("steady_runs_s")
    if not isinstance(steady, list) or len(steady) < 3:
        raise ManifestError("idle evidence requires at least three steady requests")
    steady_value = statistics.median(
        _number(value, "idle steady request", positive=True) for value in steady
    )
    _number(evidence.get("idle_seconds"), "idle_seconds", positive=True)
    return {
        "steady": steady_value,
        "first": _number(
            evidence.get("first_after_idle_s"), "first_after_idle_s", positive=True
        ),
        "second": _number(
            evidence.get("second_after_idle_s"), "second_after_idle_s", positive=True
        ),
    }


def materialize_confirmation(confirmation: dict[str, Any]) -> dict[str, Any]:
    """Derive a BAAB confirmation suite from raw benchmark, quality, and idle evidence."""
    _keys(confirmation, {
        "schema_version", "campaign_id", "baseline_settings", "candidate_settings",
        "screen_receipt_sha256", "source_rounds",
    }, "confirmation")
    if confirmation.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"confirmation schema_version must be {SCHEMA_VERSION}")
    sources = confirmation.get("source_rounds")
    if not isinstance(sources, list) or len(sources) != len(CONFIRMATION_SEQUENCE):
        raise ManifestError("confirmation source_rounds must contain exactly four rounds")
    rounds: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ManifestError(f"confirmation source round {index} must be an object")
        _keys(source, {"arm", "benchmarks", "quality", "idle"}, f"confirmation source round {index}")
        benchmarks = source["benchmarks"]
        quality = source["quality"]
        idle = source["idle"]
        if not isinstance(benchmarks, Mapping) or not isinstance(quality, Mapping) \
                or not isinstance(idle, Mapping):
            raise ManifestError("confirmation source evidence must be objects")
        rounds.append(build_suite_round_from_evidence(
            source["arm"], benchmarks, quality, idle
        ))
    materialized = {
        "schema_version": confirmation["schema_version"],
        "campaign_id": confirmation["campaign_id"],
        "baseline_settings": copy.deepcopy(confirmation["baseline_settings"]),
        "candidate_settings": copy.deepcopy(confirmation["candidate_settings"]),
        "screen_receipt_sha256": confirmation["screen_receipt_sha256"],
        "rounds": rounds,
    }
    _validate_materialized_confirmation(materialized)
    return materialized


def _validate_screen_receipt(
    confirmation: Mapping[str, Any], screen_receipt: Mapping[str, Any]
) -> None:
    if not isinstance(screen_receipt, Mapping):
        raise ManifestError("screen receipt must be an object")
    digest = hashlib.sha256(_canonical(dict(screen_receipt))).hexdigest()
    if confirmation.get("screen_receipt_sha256") != digest:
        raise ManifestError("confirmation is not bound to the supplied screen receipt")
    if screen_receipt.get("verdict") != "screen_pass" \
            or screen_receipt.get("final_acceptance") is not False:
        raise ManifestError("confirmation requires a passed non-final screen receipt")
    if screen_receipt.get("campaign_id") != confirmation.get("campaign_id"):
        raise ManifestError("screen receipt belongs to a different campaign")
    baseline_hash = hashlib.sha256(_canonical(confirmation["baseline_settings"])).hexdigest()
    candidate_hash = hashlib.sha256(_canonical(confirmation["candidate_settings"])).hexdigest()
    if screen_receipt.get("baseline_settings_sha256") != baseline_hash \
            or screen_receipt.get("candidate_settings_sha256") != candidate_hash:
        raise ManifestError("screen receipt settings do not match confirmation settings")
    if screen_receipt.get("changed_settings") != _changed_settings(
        confirmation["baseline_settings"], confirmation["candidate_settings"]
    ):
        raise ManifestError("screen receipt changed setting does not match confirmation")


def validate_confirmation(
    confirmation: dict[str, Any], screen_receipt: Mapping[str, Any]
) -> None:
    materialized = materialize_confirmation(confirmation)
    _validate_screen_receipt(materialized, screen_receipt)


def _validate_materialized_confirmation(confirmation: dict[str, Any]) -> None:
    _keys(confirmation, {
        "schema_version", "campaign_id", "baseline_settings", "candidate_settings",
        "screen_receipt_sha256", "rounds",
    }, "confirmation")
    if type(confirmation["schema_version"]) is not int or confirmation["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(f"confirmation schema_version must be {SCHEMA_VERSION}")
    if not isinstance(confirmation["campaign_id"], str) or not confirmation["campaign_id"]:
        raise ManifestError("campaign_id must be a non-empty string")
    _sha256(confirmation["screen_receipt_sha256"], "screen_receipt_sha256")
    baseline = confirmation["baseline_settings"]
    candidate = confirmation["candidate_settings"]
    if not isinstance(baseline, dict) or not baseline or not isinstance(candidate, dict):
        raise ManifestError("confirmation settings must be objects")
    if len(_changed_settings(baseline, candidate)) != 1:
        raise ManifestError("confirmation must change exactly one setting")
    rounds = confirmation["rounds"]
    _validate_common_rounds(rounds, CONFIRMATION_SEQUENCE, baseline, candidate, {
        "arm", "restart_id", "restart_identity", "timing_mode", "performance_knobs",
        "receipt_sha256", "hard_gates", "shapes",
    })
    for index, round_ in enumerate(rounds):
        shapes = round_["shapes"]
        if not isinstance(shapes, dict) or set(shapes) != set(CONFIRMATION_SHAPES):
            raise ManifestError(f"round {index} shapes must contain exactly {CONFIRMATION_SHAPES}")
        for shape in BENCHMARK_SHAPES:
            _shape_value(shapes[shape], f"round {index} {shape}")
        _b2_final_128(shapes["B2"], f"round {index} B2")
        for shape in EQUIVALENCE_SHAPES:
            if type(shapes[shape]) is not bool:
                raise ManifestError(f"round {index} {shape} must be a boolean")
        idle = shapes["idle"]
        if not isinstance(idle, dict):
            raise ManifestError(f"round {index} idle must be an object")
        _keys(idle, {"steady", "first", "second"}, f"round {index} idle")
        for name, value in idle.items():
            _number(value, f"round {index} idle {name}", positive=True)


def evaluate_suite(
    confirmation: dict[str, Any], screen_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the full BAAB suite; this is the only final optimization gate."""
    source_confirmation = confirmation
    confirmation = materialize_confirmation(source_confirmation)
    _validate_screen_receipt(confirmation, screen_receipt)
    rounds = confirmation["rounds"]
    reasons: list[str] = []
    failed_gates = sorted({name for round_ in rounds for name, passed in round_["hard_gates"].items() if not passed})
    reasons.extend(f"hard gate failed: {name}" for name in failed_gates)
    failed_equivalence = sorted({shape for round_ in rounds for shape in EQUIVALENCE_SHAPES if not round_["shapes"][shape]})
    reasons.extend(f"equivalence failed: {shape}" for shape in failed_equivalence)

    b2_ratios = []
    for index, round_ in enumerate(rounds):
        b1 = _shape_value(round_["shapes"]["B1"], f"round {index} B1")
        b2_final = _b2_final_128(round_["shapes"]["B2"], f"round {index} B2")
        ratio = b2_final / b1
        b2_ratios.append(ratio)
        if ratio > 1.2:
            reasons.append(f"B2 final128 exceeds 1.2x B1 in round {index}")

    paired_regressions: dict[str, list[float]] = {}
    for shape in BENCHMARK_SHAPES:
        values = []
        for a, b in _pairs(rounds):
            a_value = _shape_value(a["shapes"][shape], f"{shape} A")
            b_value = _shape_value(b["shapes"][shape], f"{shape} B")
            values.append(b_value / a_value - 1)
        paired_regressions[shape] = values
        if any(value > 0.02 for value in values):
            reasons.append(f"paired regression exceeds 2%: {shape}")

    ttft_repeat_ranges: dict[str, dict[str, float]] = {}
    for shape in TTFT_SHAPES:
        arm_ranges = {}
        for arm in ("A", "B"):
            values = [_shape_value(round_["shapes"][shape], f"{shape} {arm}") for round_ in rounds if round_["arm"] == arm]
            repeat_range = max(values) / min(values) - 1
            arm_ranges[arm] = repeat_range
            if repeat_range > 0.02:
                reasons.append(f"TTFT repeat range exceeds 2%: {shape} arm {arm}")
        ttft_repeat_ranges[shape] = arm_ranges

    idle_ratios = []
    for index, round_ in enumerate(rounds):
        idle = round_["shapes"]["idle"]
        steady = float(idle["steady"])
        first_ratio = float(idle["first"]) / steady
        second_delta = abs(float(idle["second"]) / steady - 1)
        idle_ratios.append({"first_to_steady": first_ratio, "second_delta": second_delta})
        if first_ratio > 2:
            reasons.append(f"idle first request exceeds 2x steady in round {index}")
        if second_delta > 0.02:
            reasons.append(f"idle second request differs from steady by more than 2% in round {index}")

    verdict = "keep" if not reasons else "reject"
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": confirmation["campaign_id"],
        "confirmation_sha256": hashlib.sha256(_canonical(source_confirmation)).hexdigest(),
        "materialized_confirmation_sha256": hashlib.sha256(
            _canonical(confirmation)
        ).hexdigest(),
        "screen_receipt_sha256": confirmation["screen_receipt_sha256"],
        "changed_settings": _changed_settings(confirmation["baseline_settings"], confirmation["candidate_settings"]),
        "paired_regressions_fraction": paired_regressions,
        "b2_final128_to_b1_ratios": b2_ratios,
        "ttft_repeat_ranges_fraction": ttft_repeat_ranges,
        "idle_ratios": idle_ratios,
        "failed_gates": failed_gates,
        "failed_equivalence": failed_equivalence,
        "verdict": verdict,
        "final_acceptance": verdict == "keep",
        "stop": bool(failed_gates or failed_equivalence),
        "reasons": reasons,
    }


evaluate_confirmation = evaluate_suite


def build_suite_round_from_evidence(
    arm: str,
    benchmark_evidence_by_shape: Mapping[str, Mapping[str, Any]],
    quality_evidence: Mapping[str, Any],
    idle: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one full confirmation round from its benchmark evidence files."""
    if set(benchmark_evidence_by_shape) != set(BENCHMARK_SHAPES):
        raise ManifestError(f"suite benchmarks must contain exactly {BENCHMARK_SHAPES}")
    built = {
        shape: build_round_from_evidence(
            arm,
            benchmark_evidence_by_shape[shape],
            primary_metric=("first_token_s" if shape in TTFT_SHAPES else "wall_s" if shape == "B4" else "decode_step_median_s"),
            secondary_metrics={},
        )
        for shape in BENCHMARK_SHAPES
    }
    first = built["B1"]
    for shape, proof in built.items():
        for key in ("restart_id", "restart_identity", "timing_mode", "performance_knobs"):
            if proof[key] != first[key]:
                raise ManifestError(f"{shape} was not measured in the same restart and mode")
    shapes: dict[str, Any] = {shape: built[shape]["primary"] for shape in BENCHMARK_SHAPES}
    b2_evidence = normalize_benchmark_evidence(benchmark_evidence_by_shape["B2"])
    shapes["B2"] = {
        "value": built["B2"]["primary"],
        "final_128": _median(b2_evidence.get("final_128_decode_step_median_s"), "B2 final_128_decode_step_median_s"),
    }
    all_gates = [proof["hard_gates"] for proof in built.values()]
    quality = normalize_quality_evidence(quality_evidence)
    quality_identity = _restart_identity(quality)
    quality_timing, quality_knobs = _timing_and_knobs(quality)
    if (
        _restart_id(quality_identity) != first["restart_id"]
        or quality_timing != first["timing_mode"]
        or quality_knobs != first["performance_knobs"]
    ):
        raise ManifestError("quality evidence was not measured in the same restart and mode")
    shapes.update(_quality_equivalence(quality))
    shapes["idle"] = _idle_values(idle, first)
    hard_gates = {gate: all(gates[gate] for gates in all_gates) for gate in HARD_GATES}
    source_evidence = {
        "benchmarks": dict(benchmark_evidence_by_shape),
        "quality": dict(quality_evidence),
        "idle": dict(idle),
    }
    return {
        "arm": arm,
        "restart_id": first["restart_id"],
        "restart_identity": first["restart_identity"],
        "timing_mode": first["timing_mode"],
        "performance_knobs": first["performance_knobs"],
        "receipt_sha256": hashlib.sha256(_canonical(source_evidence)).hexdigest(),
        "hard_gates": hard_gates,
        "shapes": shapes,
    }


def write_campaign_receipt(receipt_dir: Path, result: dict[str, Any]) -> Path:
    payload = _canonical(result)
    digest = hashlib.sha256(payload).hexdigest()
    receipt_dir.mkdir(parents=True, exist_ok=True)
    path = receipt_dir / f"paired-{digest}.json"
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise RuntimeError(f"content-address collision at {path}")
    return path
