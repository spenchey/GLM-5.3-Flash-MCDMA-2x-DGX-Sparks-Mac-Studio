"""Fail-closed evaluator for same-width paired capacity evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


SEQUENCE = "ABBABAAB"
MINIMUM_GAIN_FRACTION = 0.03
MAX_LATENCY_REGRESSION_FRACTION = 0.02
CONFIDENCE_MULTIPLIER = 2.353
LATENCY_METRICS = (
    "max_ttft_seconds",
    "p95_completion_seconds",
    "max_completion_seconds",
)


class CapacityEvidenceError(ValueError):
    """Raised when benchmark evidence cannot support a claim."""


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CapacityEvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise CapacityEvidenceError(f"{label} must be finite{' and positive' if positive else ''}")
    return result


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return the same linearly interpolated percentile used by the producer."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise CapacityEvidenceError("cannot derive a percentile from no lanes")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _require_derived_metric(
    benchmark: Mapping[str, Any], field: str, expected: float,
) -> None:
    actual = _finite(benchmark.get(field), field, positive=True)
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise CapacityEvidenceError(f"round {field} differs from raw lane evidence")


def _counter_delta(envelope: Mapping[str, Any]) -> dict[str, int]:
    states: dict[str, Mapping[str, Any]] = {}
    for when in ("before", "after"):
        state = envelope.get(f"mcdma_{when}")
        if not isinstance(state, Mapping) or state.get("link") != "up" or bool(state.get("inflight")):
            raise CapacityEvidenceError(f"MCDMA link was not up and idle {when}")
        if type(state.get("generation")) is not int or int(state["generation"]) < 1:
            raise CapacityEvidenceError(f"MCDMA generation was malformed {when}")
        if state.get("reply_present") is not False:
            raise CapacityEvidenceError(f"MCDMA retained an unconsumed reply {when}")
        for side in ("mac", "spark"):
            peer = state.get(side)
            if (
                not isinstance(peer, Mapping)
                or peer.get("alive") is not True
                or type(peer.get("pid")) is not int
                or int(peer["pid"]) <= 1
                or type(peer.get("calls")) is not int
                or type(peer.get("failures")) is not int
                or peer.get("failures") != 0
            ):
                raise CapacityEvidenceError(f"MCDMA {side} counter was malformed {when}")
        if state["spark"].get("service") != "poll":
            raise CapacityEvidenceError(f"Spark MCDMA service was not attached {when}")
        states[when] = state
    if states["before"].get("generation") != states["after"].get("generation"):
        raise CapacityEvidenceError("MCDMA generation changed during a round")
    for side in ("mac", "spark"):
        if states["before"][side].get("pid") != states["after"][side].get("pid"):
            raise CapacityEvidenceError(f"MCDMA {side} daemon changed during a round")
    failures = {
        side: int(states["after"][side]["failures"]) - int(states["before"][side]["failures"])
        for side in ("mac", "spark")
    }
    if any(value != 0 for value in failures.values()):
        raise CapacityEvidenceError("MCDMA failure counters changed during a round")
    calls = {
        side: int(states["after"][side]["calls"]) - int(states["before"][side]["calls"])
        for side in ("mac", "spark")
    }
    if calls["mac"] != calls["spark"] or calls["mac"] < 0:
        raise CapacityEvidenceError("MCDMA call counters do not contain an equal nonnegative delta")
    return calls


def _validate_round(
    envelope: Mapping[str, Any], *, parallel: int, mac_parallel: int,
    prompt_sha256: str, token_sha256: str, text_sha256: str,
    max_new_tokens: int, expected_transfer_id: str, expected_cache_bytes: int,
    expected_frame_lengths: Sequence[int], expected_frame_count: int,
    max_frame_bytes: int,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    benchmark = envelope.get("benchmark")
    if not isinstance(benchmark, dict):
        raise CapacityEvidenceError("round envelope has no benchmark object")
    arm = envelope.get("arm")
    if arm not in {"A", "B"} or benchmark.get("arm") != arm:
        raise CapacityEvidenceError("round arm identity differs")
    if benchmark.get("event") != "glm_paired_capacity_round" or benchmark.get("schema") != 2:
        raise CapacityEvidenceError("round has the wrong benchmark schema")
    if envelope.get("ordinal") != benchmark.get("run"):
        raise CapacityEvidenceError("round envelope ordinal differs from the worker run")
    if benchmark.get("parallel") != parallel:
        raise CapacityEvidenceError("round used the wrong total width")
    allocation = benchmark.get("allocation")
    expected_allocation = (
        {"spark": parallel, "mac": 0}
        if arm == "A"
        else {"spark": parallel - mac_parallel, "mac": mac_parallel}
    )
    if allocation != expected_allocation:
        raise CapacityEvidenceError("round used the wrong allocation")
    if benchmark.get("token_sha256") != token_sha256:
        raise CapacityEvidenceError("round token hash differs from the frozen reference")
    if benchmark.get("text_sha256") != text_sha256:
        raise CapacityEvidenceError("round text hash differs from the frozen reference")
    if benchmark.get("prompt_sha256") != prompt_sha256:
        raise CapacityEvidenceError("round prompt hash differs from the frozen cache")
    if benchmark.get("max_new_tokens") != max_new_tokens:
        raise CapacityEvidenceError("round used the wrong fixed reply length")
    lanes = benchmark.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != parallel:
        raise CapacityEvidenceError("round has the wrong number of lane records")
    if any(not isinstance(lane, Mapping) for lane in lanes):
        raise CapacityEvidenceError("round contains a malformed lane record")
    expected_topology = [
        ("spark", index) for index in range(expected_allocation["spark"])
    ] + [
        ("mac", index) for index in range(expected_allocation["mac"])
    ]
    if any(
        lane.get("kind") not in {"spark", "mac"}
        or type(lane.get("lane")) is not int
        for lane in lanes
    ):
        raise CapacityEvidenceError("round lane identity is malformed")
    observed_topology = [(lane["kind"], lane["lane"]) for lane in lanes]
    if observed_topology != expected_topology:
        raise CapacityEvidenceError("round lane topology differs from the declared allocation")
    if any(lane.get("token_sha256") != token_sha256 for lane in lanes):
        raise CapacityEvidenceError("a lane token hash differs from the frozen reference")
    text_hashes = {lane.get("text_sha256") for lane in lanes}
    if text_hashes != {text_sha256}:
        raise CapacityEvidenceError("lane output bytes differ")
    if any(lane.get("tokens") != max_new_tokens for lane in lanes):
        raise CapacityEvidenceError("a lane stopped before the fixed reply length")
    completion_tokens = benchmark.get("completion_tokens")
    if completion_tokens != parallel * max_new_tokens:
        raise CapacityEvidenceError("round did not complete the exact fixed token count")
    seconds = _finite(benchmark.get("seconds"), "round wall time", positive=True)
    first_tokens: list[float] = []
    completions: list[float] = []
    for index, lane in enumerate(lanes):
        first = _finite(lane.get("first_token_seconds"), f"lane {index} TTFT", positive=True)
        completion = _finite(
            lane.get("completion_seconds"), f"lane {index} completion", positive=True
        )
        if completion < first:
            raise CapacityEvidenceError(f"lane {index} completed before its first token")
        first_tokens.append(first)
        completions.append(completion)
    if seconds < max(completions) and not math.isclose(
        seconds, max(completions), rel_tol=1e-9, abs_tol=1e-9
    ):
        raise CapacityEvidenceError("round wall time is shorter than a lane completion")
    _require_derived_metric(
        benchmark,
        "aggregate_pipeline_tokens_per_second",
        completion_tokens / seconds,
    )
    _require_derived_metric(benchmark, "max_ttft_seconds", max(first_tokens))
    _require_derived_metric(
        benchmark, "p95_completion_seconds", _percentile(completions, 0.95)
    )
    _require_derived_metric(benchmark, "max_completion_seconds", max(completions))
    calls = _counter_delta(envelope)
    if arm == "A" and calls != {"mac": 0, "spark": 0}:
        raise CapacityEvidenceError("two-Spark baseline made an MCDMA call")
    if arm == "B":
        mac = benchmark.get("mac")
        if not isinstance(mac, Mapping):
            raise CapacityEvidenceError("candidate has no Mac transfer evidence")
        if mac.get("cache_wire_transfers") != 1:
            raise CapacityEvidenceError("candidate did not use exactly one cache transfer")
        if mac.get("cache_reuse_copies") != mac_parallel - 1:
            raise CapacityEvidenceError("candidate did not reuse the transferred cache")
        if mac.get("prompt_sha256") != prompt_sha256:
            raise CapacityEvidenceError("candidate transferred the wrong prompt cache")
        transfer_id = mac.get("transfer_id")
        if transfer_id != expected_transfer_id:
            raise CapacityEvidenceError("candidate cache transfer identity differs from the frozen reference")
        cache_bytes = mac.get("cache_bytes_total")
        lengths = mac.get("cache_frame_lengths")
        if cache_bytes != expected_cache_bytes:
            raise CapacityEvidenceError("candidate cache byte count differs from the frozen reference")
        if (
            mac.get("cache_frame_count") != expected_frame_count
            or not isinstance(lengths, list)
            or len(lengths) != expected_frame_count
            or any(type(value) is not int or not 0 < value <= max_frame_bytes for value in lengths)
            or sum(lengths) != cache_bytes
            or lengths != list(expected_frame_lengths)
        ):
            raise CapacityEvidenceError("candidate cache frame evidence differs from the frozen reference")
    attestation = envelope.get("native_attestation")
    if not isinstance(attestation, dict) or attestation.get("schema") != 1 \
            or attestation.get("accepted") is not True:
        raise CapacityEvidenceError("round has no accepted native-container attestation")
    for role in ("head", "worker"):
        peer = attestation.get(role)
        if not isinstance(peer, Mapping):
            raise CapacityEvidenceError(f"round has no {role} container attestation")
        for field in (
            "container_id", "configuration_sha256", "runtime_configuration_sha256",
            "started_at", "image_id",
        ):
            if not peer.get(field):
                raise CapacityEvidenceError(f"round {role} attestation lacks {field}")
        if peer.get("restart_count") != 0:
            raise CapacityEvidenceError(f"round {role} container restart count is not zero")
    replay = attestation.get("replay")
    if not isinstance(replay, Mapping):
        raise CapacityEvidenceError("round has no cache-replay container attestation")
    for field in (
        "container_id", "configuration_sha256", "runtime_configuration_sha256",
        "started_at", "image_id",
    ):
        if not replay.get(field):
            raise CapacityEvidenceError(f"round replay attestation lacks {field}")
    if replay.get("restart_count") != 0 or replay.get("prompt_sha256") != prompt_sha256:
        raise CapacityEvidenceError("round replay container identity is invalid")
    return benchmark, calls["mac"], attestation


def _pair_rounds(rounds: Sequence[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index in range(0, len(rounds), 2):
        block = rounds[index:index + 2]
        if len(block) != 2 or {item["arm"] for item in block} != {"A", "B"}:
            raise CapacityEvidenceError("counterbalanced sequence must contain four adjacent A/B pairs")
        baseline = next(item for item in block if item["arm"] == "A")
        candidate = next(item for item in block if item["arm"] == "B")
        pairs.append((baseline, candidate))
    return pairs


def evaluate(
    envelopes: Sequence[Mapping[str, Any]],
    *,
    parallel: int,
    mac_parallel: int,
    prompt_sha256: str,
    token_sha256: str,
    text_sha256: str,
    max_new_tokens: int,
    expected_transfer_id: str,
    expected_cache_bytes: int,
    expected_frame_lengths: Sequence[int],
    preparation_timings: Mapping[str, Any],
    expected_mcdma_calls: int = 5,
    expected_frame_count: int = 3,
    max_frame_bytes: int = 64 * 1024 * 1024,
    native_cuda_lanes: int = 4,
) -> dict[str, Any]:
    """Validate evidence and apply the paired speed and tail gates."""

    if not 1 <= parallel <= 16 or not 1 <= mac_parallel <= min(8, parallel):
        raise CapacityEvidenceError("allocation is outside the supported benchmark range")
    for label, value in (
        ("prompt_sha256", prompt_sha256),
        ("token_sha256", token_sha256),
        ("text_sha256", text_sha256),
    ):
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise CapacityEvidenceError(f"{label} must be a lowercase SHA-256")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise CapacityEvidenceError("max_new_tokens must be a positive integer")
    expected_timing_fields = {
        "reference_wall_seconds",
        "cuda_prefill_head_seconds",
        "cuda_prefill_worker_seconds",
        "mcdma_transfer_seconds",
        "mac_import_seconds",
    }
    if set(preparation_timings) != expected_timing_fields:
        raise CapacityEvidenceError("preparation timing evidence has the wrong fields")
    normalized_timings = {
        field: _finite(preparation_timings[field], field, positive=True)
        for field in sorted(expected_timing_fields)
    }
    if not expected_transfer_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in expected_transfer_id
    ):
        raise CapacityEvidenceError("expected_transfer_id is malformed")
    if type(expected_cache_bytes) is not int or expected_cache_bytes <= 0:
        raise CapacityEvidenceError("expected_cache_bytes must be a positive integer")
    if type(expected_mcdma_calls) is not int or expected_mcdma_calls <= 0:
        raise CapacityEvidenceError("expected_mcdma_calls must be a positive integer")
    if type(expected_frame_count) is not int or expected_frame_count <= 0:
        raise CapacityEvidenceError("expected_frame_count must be a positive integer")
    if expected_mcdma_calls != expected_frame_count + 2:
        raise CapacityEvidenceError("MCDMA call contract must be frame count plus prepare and release")
    if (
        not isinstance(expected_frame_lengths, Sequence)
        or isinstance(expected_frame_lengths, (str, bytes))
        or len(expected_frame_lengths) != expected_frame_count
        or any(type(value) is not int or value <= 0 for value in expected_frame_lengths)
        or sum(expected_frame_lengths) != expected_cache_bytes
    ):
        raise CapacityEvidenceError("frozen cache frame evidence is malformed")
    if type(max_frame_bytes) is not int or max_frame_bytes <= 0:
        raise CapacityEvidenceError("max_frame_bytes must be a positive integer")
    if type(native_cuda_lanes) is not int or native_cuda_lanes <= 0:
        raise CapacityEvidenceError("native_cuda_lanes must be a positive integer")
    expected_schedule = [("warmup", "A", -2), ("warmup", "B", -1)] + [
        ("measured", arm, run) for run, arm in enumerate(SEQUENCE, start=1)
    ]
    observed_schedule = [
        (value.get("phase"), value.get("arm"), value.get("ordinal")) for value in envelopes
    ]
    if observed_schedule != expected_schedule:
        raise CapacityEvidenceError("benchmark round order, arm, or ordinal differs from the fixed schedule")

    normalized: list[dict[str, Any]] = []
    warm_candidate_calls: int | None = None
    previous_after: Mapping[str, Any] | None = None
    for envelope in envelopes:
        current_before = envelope.get("mcdma_before")
        if previous_after is not None and current_before != previous_after:
            raise CapacityEvidenceError("MCDMA counters or daemon identity are discontinuous between rounds")
        benchmark, calls, attestation = _validate_round(
            envelope,
            parallel=parallel,
            mac_parallel=mac_parallel,
            prompt_sha256=prompt_sha256,
            token_sha256=token_sha256,
            text_sha256=text_sha256,
            max_new_tokens=max_new_tokens,
            expected_transfer_id=expected_transfer_id,
            expected_cache_bytes=expected_cache_bytes,
            expected_frame_lengths=expected_frame_lengths,
            expected_frame_count=expected_frame_count,
            max_frame_bytes=max_frame_bytes,
        )
        item = dict(benchmark)
        item["phase"] = envelope.get("phase")
        item["mcdma_call_delta"] = calls
        item["native_attestation"] = attestation
        normalized.append(item)
        previous_after = envelope.get("mcdma_after")
        if item["phase"] == "warmup" and item["arm"] == "B":
            warm_candidate_calls = calls
    if warm_candidate_calls != expected_mcdma_calls:
        raise CapacityEvidenceError(
            f"candidate warmup made {warm_candidate_calls} MCDMA calls; "
            f"expected exactly {expected_mcdma_calls}"
        )
    candidate_calls = [
        item["mcdma_call_delta"]
        for item in normalized
        if item["phase"] == "measured" and item["arm"] == "B"
    ]
    if any(value != warm_candidate_calls for value in candidate_calls):
        raise CapacityEvidenceError("measured candidate MCDMA calls differ from the warmup contract")
    completion_counts = {int(item["completion_tokens"]) for item in normalized}
    if len(completion_counts) != 1:
        raise CapacityEvidenceError("rounds completed different token counts")
    text_hashes = {
        lane["text_sha256"] for item in normalized for lane in item["lanes"]
    }
    if len(text_hashes) != 1:
        raise CapacityEvidenceError("output bytes changed across benchmark rounds")
    for role in ("head", "worker", "replay"):
        identities = {
            json.dumps(item["native_attestation"][role], sort_keys=True)
            for item in normalized
        }
        if len(identities) != 1:
            raise CapacityEvidenceError(f"{role} container identity or configuration changed across rounds")

    measured_rounds = [item for item in normalized if item["phase"] == "measured"]
    pairs = _pair_rounds(measured_rounds)
    log_gains: list[float] = []
    gains: list[float] = []
    pair_records: list[dict[str, Any]] = []
    reasons: list[str] = []
    for index, (baseline, candidate) in enumerate(pairs, start=1):
        base_rate = _finite(
            baseline["aggregate_pipeline_tokens_per_second"], "baseline throughput", positive=True
        )
        candidate_rate = _finite(
            candidate["aggregate_pipeline_tokens_per_second"], "candidate throughput", positive=True
        )
        gain = candidate_rate / base_rate - 1.0
        gains.append(gain)
        log_gains.append(math.log(candidate_rate / base_rate))
        latencies: dict[str, Any] = {}
        if gain < MINIMUM_GAIN_FRACTION:
            reasons.append(f"pair {index} throughput gain was below 3%")
        for metric in LATENCY_METRICS:
            baseline_value = _finite(baseline[metric], f"baseline {metric}", positive=True)
            candidate_value = _finite(candidate[metric], f"candidate {metric}", positive=True)
            regression = candidate_value / baseline_value - 1.0
            latencies[metric] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
                "regression_fraction": regression,
                "accepted": regression <= MAX_LATENCY_REGRESSION_FRACTION,
            }
            if regression > MAX_LATENCY_REGRESSION_FRACTION:
                reasons.append(f"pair {index} {metric} regressed by more than 2%")
        pair_records.append({
            "pair": index,
            "order": baseline["arm"] + candidate["arm"] if baseline["run"] < candidate["run"] else candidate["arm"] + baseline["arm"],
            "baseline_tokens_per_second": base_rate,
            "candidate_tokens_per_second": candidate_rate,
            "gain_fraction": gain,
            "throughput_accepted": gain >= MINIMUM_GAIN_FRACTION,
            "latency": latencies,
        })

    mean_log_gain = statistics.mean(log_gains)
    standard_error = statistics.stdev(log_gains) / math.sqrt(len(log_gains))
    lower_log_gain = mean_log_gain - CONFIDENCE_MULTIPLIER * standard_error
    lower_gain = math.exp(lower_log_gain) - 1.0
    if lower_gain < MINIMUM_GAIN_FRACTION:
        reasons.append("paired lower confidence gain was below 3%")
    median_gain = statistics.median(gains)
    if median_gain < MINIMUM_GAIN_FRACTION:
        reasons.append("median paired gain was below 3%")

    accepted = not reasons
    baseline_rounds = [item for item in measured_rounds if item["arm"] == "A"]
    candidate_rounds = [item for item in measured_rounds if item["arm"] == "B"]
    return {
        "schema": 1,
        "event": "glm_paired_capacity_acceptance",
        "accepted": accepted,
        "claim_scope": "steady-state-reused-shared-prefix",
        "baseline_scope": "fresh-pinned-portable-two-spark-same-checkpoint",
        "prefix_preparation_in_timed_round": False,
        "preparation_timings": normalized_timings,
        "sequence": SEQUENCE,
        "warmup_sequence": "AB",
        "parallel": parallel,
        "native_cuda_lanes": native_cuda_lanes,
        "baseline_client_requests": parallel,
        "allocation": {"spark": parallel - mac_parallel, "mac": mac_parallel},
        "minimum_gain_fraction": MINIMUM_GAIN_FRACTION,
        "maximum_latency_regression_fraction": MAX_LATENCY_REGRESSION_FRACTION,
        "confidence_multiplier": CONFIDENCE_MULTIPLIER,
        "candidate_mcdma_calls_per_round": warm_candidate_calls,
        "expected_mcdma_calls_per_candidate_round": expected_mcdma_calls,
        "token_sha256": token_sha256,
        "text_sha256": text_sha256,
        "prompt_sha256": prompt_sha256,
        "transfer_id": expected_transfer_id,
        "cache_bytes_total": expected_cache_bytes,
        "cache_frame_lengths": list(expected_frame_lengths),
        "max_new_tokens": max_new_tokens,
        "baseline_median_tokens_per_second": statistics.median(
            float(item["aggregate_pipeline_tokens_per_second"]) for item in baseline_rounds
        ),
        "candidate_median_tokens_per_second": statistics.median(
            float(item["aggregate_pipeline_tokens_per_second"]) for item in candidate_rounds
        ),
        "mean_paired_gain_fraction": math.exp(mean_log_gain) - 1.0,
        "median_paired_gain_fraction": median_gain,
        "lower_confidence_gain_fraction": lower_gain,
        "native_configuration_sha256": {
            role: normalized[0]["native_attestation"][role]["configuration_sha256"]
            for role in ("head", "worker", "replay")
        },
        "native_container_ids": {
            role: normalized[0]["native_attestation"][role]["container_id"]
            for role in ("head", "worker", "replay")
        },
        "native_started_at": {
            role: normalized[0]["native_attestation"][role]["started_at"]
            for role in ("head", "worker", "replay")
        },
        "pairs": pair_records,
        "reasons": reasons,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise CapacityEvidenceError(f"line {number} is not a JSON object")
        values.append(value)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rounds", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--parallel", type=int, required=True)
    parser.add_argument("--mac-parallel", type=int, required=True)
    parser.add_argument("--expected-mcdma-calls", type=int, default=5)
    parser.add_argument("--expected-frame-count", type=int, default=3)
    parser.add_argument("--native-cuda-lanes", type=int, default=4)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    if (
        not isinstance(reference, dict)
        or reference.get("event") != "glm_concurrent_reference"
        or reference.get("schema") != 2
    ):
        raise CapacityEvidenceError("reference file has the wrong schema")
    result = evaluate(
        load_jsonl(args.rounds),
        parallel=args.parallel,
        mac_parallel=args.mac_parallel,
        prompt_sha256=str(reference.get("prompt_sha256", "")),
        token_sha256=str(reference.get("token_sha256", "")),
        text_sha256=str(reference.get("text_sha256", "")),
        max_new_tokens=reference.get("max_new_tokens"),
        expected_transfer_id=str(reference.get("transfer_id", "")),
        expected_cache_bytes=reference.get("cache_bytes_total"),
        expected_frame_lengths=reference.get("cache_frame_lengths") or [],
        preparation_timings=reference.get("preparation_timings") or {},
        expected_mcdma_calls=args.expected_mcdma_calls,
        expected_frame_count=args.expected_frame_count,
        native_cuda_lanes=args.native_cuda_lanes,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
