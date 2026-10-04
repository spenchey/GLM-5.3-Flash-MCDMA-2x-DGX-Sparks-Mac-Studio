"""Fail-closed scoring for the one-request Spark-prefill/MCDMA/Mac-decode goal.

This evaluator intentionally cannot consume the older capacity benchmark.  It
compares one native two-Spark request (arm A) with one cache-handoff request
(arm B), using adjacent counterbalanced pairs and client-observed completion
time as the primary metric.
"""

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
CONFIDENCE_MULTIPLIER = 2.353  # two-sided 95% t critical value, df=3
CONTEXT_TARGETS = (2048, 8192, 32768, 65536, 131072)
TAIL_METRICS = ("inter_token_p95_seconds", "inter_token_p99_seconds")


class SingleStreamEvidenceError(ValueError):
    """Raised when evidence is incomplete, inconsistent, or unsafe."""


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SingleStreamEvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        suffix = " and positive" if positive else ""
        raise SingleStreamEvidenceError(f"{label} must be finite{suffix}")
    return result


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise SingleStreamEvidenceError(f"{label} must be a positive integer")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise SingleStreamEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _identity(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    identity = benchmark.get("identity")
    if not isinstance(identity, Mapping):
        raise SingleStreamEvidenceError("round has no frozen identity")
    expected = {
        "model_id", "model_revision", "checkpoint_sha256", "tokenizer_sha256",
        "prompt_sha256", "prompt_tokens", "max_new_tokens", "sampling",
        "thinking", "draft_policy", "temperature", "state",
    }
    if set(identity) != expected:
        raise SingleStreamEvidenceError("round identity fields differ from the frozen contract")
    for field in ("checkpoint_sha256", "tokenizer_sha256", "prompt_sha256"):
        _sha256(identity[field], field)
    for field in ("model_id", "model_revision", "sampling", "draft_policy", "state"):
        if not isinstance(identity[field], str) or not identity[field]:
            raise SingleStreamEvidenceError(f"identity {field} must be a non-empty string")
    _positive_int(identity["prompt_tokens"], "prompt_tokens")
    _positive_int(identity["max_new_tokens"], "max_new_tokens")
    if identity["sampling"] != "greedy" or identity["temperature"] != 0:
        raise SingleStreamEvidenceError("single-stream acceptance requires greedy temperature-zero output")
    if identity["thinking"] is not False:
        raise SingleStreamEvidenceError("single-stream acceptance requires thinking off")
    if identity["state"] not in {"cold", "warm"}:
        raise SingleStreamEvidenceError("identity state must be cold or warm")
    return dict(identity)


def _safety(envelope: Mapping[str, Any]) -> dict[str, int]:
    value = envelope.get("safety")
    if not isinstance(value, Mapping):
        raise SingleStreamEvidenceError("round has no safety evidence")
    expected = {"ooms", "restarts", "mcdma_failures", "swap_bytes"}
    if set(value) != expected:
        raise SingleStreamEvidenceError("round safety evidence has the wrong fields")
    result: dict[str, int] = {}
    for field in sorted(expected):
        item = value[field]
        if type(item) is not int or item < 0:
            raise SingleStreamEvidenceError(f"safety {field} must be a nonnegative integer")
        result[field] = item
    if any(result.values()):
        raise SingleStreamEvidenceError("round recorded an OOM, restart, MCDMA failure, or swap use")
    return result


def _mcdma_delta(envelope: Mapping[str, Any]) -> int:
    states: dict[str, Mapping[str, Any]] = {}
    for when in ("before", "after"):
        state = envelope.get(f"mcdma_{when}")
        if not isinstance(state, Mapping):
            raise SingleStreamEvidenceError(f"round has no MCDMA {when} state")
        if state.get("link") != "up" or bool(state.get("inflight")):
            raise SingleStreamEvidenceError(f"MCDMA was not up and idle {when}")
        if type(state.get("generation")) is not int or int(state["generation"]) <= 0:
            raise SingleStreamEvidenceError(f"MCDMA generation was malformed {when}")
        for side in ("mac", "spark"):
            peer = state.get(side)
            if (
                not isinstance(peer, Mapping)
                or peer.get("alive") is not True
                or type(peer.get("calls")) is not int
                or type(peer.get("failures")) is not int
                or int(peer["calls"]) < 0
                or int(peer["failures"]) < 0
            ):
                raise SingleStreamEvidenceError(f"MCDMA {side} evidence was malformed {when}")
        states[when] = state
    if states["before"]["generation"] != states["after"]["generation"]:
        raise SingleStreamEvidenceError("MCDMA generation changed during the request")
    calls = {
        side: int(states["after"][side]["calls"]) - int(states["before"][side]["calls"])
        for side in ("mac", "spark")
    }
    failures = {
        side: int(states["after"][side]["failures"]) - int(states["before"][side]["failures"])
        for side in ("mac", "spark")
    }
    if calls["mac"] != calls["spark"] or calls["mac"] < 0:
        raise SingleStreamEvidenceError("MCDMA call deltas do not agree")
    if any(value != 0 for value in failures.values()):
        raise SingleStreamEvidenceError("MCDMA failure counters changed")
    return calls["mac"]


def _phase_timings(benchmark: Mapping[str, Any], arm: str, total: float) -> dict[str, float]:
    raw = benchmark.get("phases")
    if not isinstance(raw, Mapping):
        raise SingleStreamEvidenceError("round has no phase timings")
    expected = (
        {"prefill_seconds", "decode_seconds"}
        if arm == "A"
        else {
            "prefill_seconds", "export_seconds", "transfer_seconds",
            "import_seconds", "decode_seconds",
        }
    )
    if set(raw) != expected:
        raise SingleStreamEvidenceError(f"arm {arm} phase timing fields are incomplete")
    phases = {name: _finite(raw[name], name, positive=True) for name in sorted(expected)}
    # Client time can include queueing and serialization, but phase accounting
    # must never claim more elapsed time than the request actually took.
    if sum(phases.values()) > total * 1.02:
        raise SingleStreamEvidenceError("phase timings exceed client-observed completion time")
    return phases


def _candidate_cache(
    benchmark: Mapping[str, Any], identity: Mapping[str, Any], phases: Mapping[str, float]
) -> dict[str, Any]:
    cache = benchmark.get("cache")
    if not isinstance(cache, Mapping):
        raise SingleStreamEvidenceError("candidate has no cache-transfer evidence")
    expected = {
        "bytes_total", "bytes_per_prompt_token", "precision", "copy_count",
        "complete_sparse_state", "transfer_calls", "effective_gbps",
    }
    if set(cache) != expected:
        raise SingleStreamEvidenceError("candidate cache evidence has the wrong fields")
    total = _positive_int(cache["bytes_total"], "cache bytes_total")
    per_token = _finite(cache["bytes_per_prompt_token"], "cache bytes_per_prompt_token", positive=True)
    expected_per_token = total / int(identity["prompt_tokens"])
    if not math.isclose(per_token, expected_per_token, rel_tol=1e-9, abs_tol=1e-9):
        raise SingleStreamEvidenceError("cache bytes per token differ from total cache bytes")
    if not isinstance(cache["precision"], str) or not cache["precision"]:
        raise SingleStreamEvidenceError("cache precision is missing")
    if type(cache["copy_count"]) is not int or cache["copy_count"] < 0:
        raise SingleStreamEvidenceError("cache copy_count must be a nonnegative integer")
    calls = _positive_int(cache["transfer_calls"], "cache transfer_calls")
    transfer_seconds = phases["transfer_seconds"]
    gbps = _finite(cache["effective_gbps"], "cache effective_gbps", positive=True)
    expected_gbps = total * 8 / transfer_seconds / 1_000_000_000
    if not math.isclose(gbps, expected_gbps, rel_tol=1e-6, abs_tol=1e-9):
        raise SingleStreamEvidenceError("cache effective Gbit/s differs from bytes and transfer time")
    if int(identity["prompt_tokens"]) > 2048 and cache["complete_sparse_state"] is not True:
        raise SingleStreamEvidenceError("long-context candidate omitted sparse-attention state")
    return {
        "bytes_total": total,
        "bytes_per_prompt_token": per_token,
        "precision": cache["precision"],
        "copy_count": cache["copy_count"],
        "complete_sparse_state": bool(cache["complete_sparse_state"]),
        "transfer_calls": calls,
        "effective_gbps": gbps,
    }


def _validate_round(envelope: Mapping[str, Any]) -> dict[str, Any]:
    benchmark = envelope.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise SingleStreamEvidenceError("round has no benchmark object")
    arm = envelope.get("arm")
    if arm not in {"A", "B"} or benchmark.get("arm") != arm:
        raise SingleStreamEvidenceError("round arm identity differs")
    if benchmark.get("schema") != 1 or benchmark.get("event") != "glm_single_stream_round":
        raise SingleStreamEvidenceError("round has the wrong single-stream schema")
    if envelope.get("ordinal") != benchmark.get("run"):
        raise SingleStreamEvidenceError("round ordinal differs from benchmark run")
    if benchmark.get("parallel") != 1:
        raise SingleStreamEvidenceError("single-stream acceptance requires parallel=1")
    identity = _identity(benchmark)
    completion_tokens = _positive_int(benchmark.get("completion_tokens"), "completion_tokens")
    if completion_tokens != identity["max_new_tokens"] or benchmark.get("finish_reason") != "length":
        raise SingleStreamEvidenceError("round did not complete the fixed output length")
    token_sha256 = _sha256(benchmark.get("token_sha256"), "token_sha256")
    text_sha256 = _sha256(benchmark.get("text_sha256"), "text_sha256")
    total = _finite(benchmark.get("total_seconds"), "total_seconds", positive=True)
    ttft = _finite(benchmark.get("ttft_seconds"), "ttft_seconds", positive=True)
    decode = _finite(benchmark.get("decode_seconds"), "decode_seconds", positive=True)
    if ttft > total or decode > total * 1.02:
        raise SingleStreamEvidenceError("request timing boundaries are inconsistent")
    decode_tps = _finite(benchmark.get("decode_tokens_per_second"), "decode_tokens_per_second", positive=True)
    expected_tps = max(0, completion_tokens - 1) / decode
    if not math.isclose(decode_tps, expected_tps, rel_tol=1e-6, abs_tol=1e-9):
        raise SingleStreamEvidenceError("decode rate differs from tokens and decode time")
    tails = {
        name: _finite(benchmark.get(name), name, positive=True) for name in TAIL_METRICS
    }
    if tails["inter_token_p99_seconds"] < tails["inter_token_p95_seconds"]:
        raise SingleStreamEvidenceError("p99 token delay is smaller than p95")
    admitted = _positive_int(benchmark.get("context_window_admitted"), "context_window_admitted")
    if admitted < int(identity["prompt_tokens"]) + completion_tokens:
        raise SingleStreamEvidenceError("request exceeds the admitted context window")
    phases = _phase_timings(benchmark, arm, total)
    _safety(envelope)
    calls = _mcdma_delta(envelope)
    cache = None
    if arm == "A":
        if calls != 0 or benchmark.get("cache") is not None:
            raise SingleStreamEvidenceError("native two-Spark baseline used MCDMA cache transfer")
    else:
        cache = _candidate_cache(benchmark, identity, phases)
        if calls != cache["transfer_calls"]:
            raise SingleStreamEvidenceError("MCDMA counters differ from candidate transfer calls")
    return {
        "arm": arm,
        "identity": identity,
        "token_sha256": token_sha256,
        "text_sha256": text_sha256,
        "total_seconds": total,
        "ttft_seconds": ttft,
        "decode_seconds": decode,
        "decode_tokens_per_second": decode_tps,
        **tails,
        "context_window_admitted": admitted,
        "phases": phases,
        "cache": cache,
        "mcdma_calls": calls,
    }


def _pairs(rounds: Sequence[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index in range(0, len(rounds), 2):
        block = rounds[index:index + 2]
        if len(block) != 2 or {item["arm"] for item in block} != {"A", "B"}:
            raise SingleStreamEvidenceError("measured sequence is not four adjacent A/B pairs")
        pairs.append((
            next(item for item in block if item["arm"] == "A"),
            next(item for item in block if item["arm"] == "B"),
        ))
    return pairs


def _paired_log_summary(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], field: str,
) -> dict[str, float]:
    ratios = [math.log(float(candidate[field]) / float(baseline[field])) for baseline, candidate in pairs]
    mean = statistics.mean(ratios)
    se = statistics.stdev(ratios) / math.sqrt(len(ratios)) if len(ratios) > 1 else math.inf
    lower_ratio = math.exp(mean - CONFIDENCE_MULTIPLIER * se)
    upper_ratio = math.exp(mean + CONFIDENCE_MULTIPLIER * se)
    return {
        "candidate_over_baseline_ratio": math.exp(mean),
        "ratio_confidence_low": lower_ratio,
        "ratio_confidence_high": upper_ratio,
        "median_baseline": statistics.median(float(item[0][field]) for item in pairs),
        "median_candidate": statistics.median(float(item[1][field]) for item in pairs),
    }


def break_even(
    *, prefill_seconds: float, spark_decode_tokens_per_second: float,
    mac_decode_tokens_per_second: float, handoff_seconds: float,
    output_tokens: int,
) -> dict[str, float | bool]:
    """Calculate whether measured component ceilings can possibly clear 3%."""

    prefill = _finite(prefill_seconds, "prefill_seconds", positive=True)
    spark_rate = _finite(
        spark_decode_tokens_per_second, "spark_decode_tokens_per_second", positive=True
    )
    mac_rate = _finite(mac_decode_tokens_per_second, "mac_decode_tokens_per_second", positive=True)
    handoff = _finite(handoff_seconds, "handoff_seconds", positive=True)
    tokens = _positive_int(output_tokens, "output_tokens")
    baseline = prefill + tokens / spark_rate
    candidate = prefill + handoff + tokens / mac_rate
    gain = 1.0 - candidate / baseline
    maximum_handoff = 0.97 * baseline - prefill - tokens / mac_rate
    return {
        "baseline_seconds": baseline,
        "candidate_seconds": candidate,
        "gain_fraction": gain,
        "maximum_handoff_seconds_for_three_percent": maximum_handoff,
        "compute_ceiling_can_pass": gain >= MINIMUM_GAIN_FRACTION,
    }


def evaluate(envelopes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate and score one cold or warm single-request campaign."""

    warmups = [item for item in envelopes if item.get("phase") == "warmup"]
    measured_raw = [item for item in envelopes if item.get("phase") == "measured"]
    if [item.get("arm") for item in warmups] != ["A", "B"]:
        raise SingleStreamEvidenceError("campaign requires one A then one B warmup")
    if "".join(str(item.get("arm")) for item in measured_raw) != SEQUENCE:
        raise SingleStreamEvidenceError(f"measured arms must be {SEQUENCE}")
    if [item.get("ordinal") for item in measured_raw] != list(range(1, len(SEQUENCE) + 1)):
        raise SingleStreamEvidenceError("measured ordinals must be 1 through 8")
    validated = [_validate_round(item) for item in envelopes]
    warmup_count = len(warmups)
    measured = validated[warmup_count:]
    identities = {json.dumps(item["identity"], sort_keys=True) for item in validated}
    if len(identities) != 1:
        raise SingleStreamEvidenceError("campaign changed model, prompt, generation, or state identity")
    token_hashes = {item["token_sha256"] for item in validated}
    text_hashes = {item["text_sha256"] for item in validated}
    if len(token_hashes) != 1 or len(text_hashes) != 1:
        raise SingleStreamEvidenceError("baseline and candidate output bytes or token IDs differ")
    pairs = _pairs(measured)
    total = _paired_log_summary(pairs, "total_seconds")
    total["estimated_gain_fraction"] = 1.0 - total["candidate_over_baseline_ratio"]
    total["conservative_gain_fraction"] = 1.0 - total["ratio_confidence_high"]
    total_pass = total["conservative_gain_fraction"] >= MINIMUM_GAIN_FRACTION

    secondary: dict[str, dict[str, float | bool]] = {}
    for field in ("ttft_seconds", *TAIL_METRICS):
        summary = _paired_log_summary(pairs, field)
        summary["conservative_regression_fraction"] = summary["ratio_confidence_high"] - 1.0
        summary["passed"] = (
            summary["conservative_regression_fraction"] <= MAX_LATENCY_REGRESSION_FRACTION
        )
        secondary[field] = summary
    identity = validated[0]["identity"]
    candidate_caches = [item["cache"] for item in measured if item["arm"] == "B"]
    cache_identities = {json.dumps(item, sort_keys=True) for item in candidate_caches}
    if len(cache_identities) != 1:
        raise SingleStreamEvidenceError("candidate cache shape or transfer accounting changed")
    accepted = bool(total_pass and all(bool(item["passed"]) for item in secondary.values()))
    return {
        "schema": 1,
        "event": "glm_single_stream_acceptance",
        "accepted": accepted,
        "sequence": SEQUENCE,
        "identity": identity,
        "token_sha256": token_hashes.pop(),
        "text_sha256": text_hashes.pop(),
        "primary": total,
        "secondary": secondary,
        "cache": candidate_caches[0],
        "context_window_admitted": min(item["context_window_admitted"] for item in validated),
        "mcdma_calls": sum(item["mcdma_calls"] for item in measured),
    }


def evaluate_context_sweep(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require accepted cold and warm campaigns at every frozen context target."""

    observed: dict[tuple[int, str], Mapping[str, Any]] = {}
    for result in results:
        if result.get("event") != "glm_single_stream_acceptance":
            raise SingleStreamEvidenceError("context sweep contains a non-acceptance record")
        identity = result.get("identity")
        if not isinstance(identity, Mapping):
            raise SingleStreamEvidenceError("context sweep result has no identity")
        key = (int(identity.get("prompt_tokens", 0)), str(identity.get("state", "")))
        if key in observed:
            raise SingleStreamEvidenceError("context sweep has a duplicate prompt/state result")
        observed[key] = result
    expected = {(tokens, state) for tokens in CONTEXT_TARGETS for state in ("cold", "warm")}
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise SingleStreamEvidenceError(f"context sweep differs (missing={missing}, extra={extra})")
    if any(item.get("accepted") is not True for item in observed.values()):
        raise SingleStreamEvidenceError("at least one context/state campaign did not pass")
    return {
        "schema": 1,
        "event": "glm_single_stream_context_acceptance",
        "accepted": True,
        "contexts": list(CONTEXT_TARGETS),
        "states": ["cold", "warm"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path, help="JSONL round evidence")
    args = parser.parse_args()
    envelopes = [json.loads(line) for line in args.evidence.read_text().splitlines() if line.strip()]
    print(json.dumps(evaluate(envelopes), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
