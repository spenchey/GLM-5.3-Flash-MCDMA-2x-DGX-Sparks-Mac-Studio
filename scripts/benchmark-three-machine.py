#!/usr/bin/env python3
"""Measure one deterministic three-machine request shape with full identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

from experiments.three_machine import protocol


MODEL = "GLM-5.3-Flash-TensorFold-MCDMA-3Machine"
CLIENT_SERVER_ABSOLUTE_TOLERANCE_S = 0.050
IDENTITY_FIELDS = (
    "deployment_id", "commit", "project_manifest", "config_sha256", "image_id",
    "link_generation",
)


def require_reference_hash(path: Path, expected: str | None) -> str:
    if not isinstance(expected, str) or len(expected) != 64 \
            or any(char not in "0123456789abcdef" for char in expected):
        raise SystemExit("THREE_MACHINE_NATIVE_REFERENCE_SHA256 must be a full lowercase SHA-256")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit("native reference file SHA-256 differs from the pinned value")
    return observed


def get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def request(url: str, prompt: str, max_tokens: int, timeout: float) -> tuple[dict[str, Any], float]:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "stream": False,
    }, separators=(",", ":")).encode()
    call = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    began = time.perf_counter()
    with urllib.request.urlopen(call, timeout=timeout) as response:
        body = json.load(response)
    return body, time.perf_counter() - began


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[rank]


def summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def normalize(response: dict[str, Any], wall_s: float, run: int, max_tokens: int,
              require_length: bool, expected_prompt_tokens: int | None = None) -> dict[str, Any]:
    choice = response["choices"][0]
    usage = response["usage"]
    metrics = response["three_machine_metrics"]
    finish_reason = choice["finish_reason"]
    prompt_count = int(usage["prompt_tokens"])
    if expected_prompt_tokens is not None and prompt_count != expected_prompt_tokens:
        raise RuntimeError("API prompt-token count differs from the native reference")
    if require_length and (usage["completion_tokens"] != max_tokens or finish_reason != "length"):
        raise RuntimeError("fixed benchmark ended before its registered token count")
    token_ids = [int(value) for value in metrics["generated_token_ids"]]
    if len(token_ids) != usage["completion_tokens"]:
        raise RuntimeError("API token IDs differ from the completion token count")
    steps = metrics.get("steps") or []
    if not steps:
        raise RuntimeError("API returned no per-step timing evidence")
    prompt_steps = [step for step in steps if step["prompt"]]
    decode_steps_raw = [step for step in steps if not step["prompt"]]
    expected_prompt_steps = math.ceil(prompt_count / protocol.WINDOW_MAX_ROWS)
    decode_rounds = int(metrics.get("decode_rounds", -1))
    if len(prompt_steps) != expected_prompt_steps:
        raise RuntimeError(
            f"API returned {len(prompt_steps)} prompt windows, expected {expected_prompt_steps}"
        )
    if len(decode_steps_raw) != decode_rounds:
        raise RuntimeError(
            f"API returned {len(decode_steps_raw)} decode windows, expected {decode_rounds}"
        )
    expected_steps = expected_prompt_steps + decode_rounds
    if len(steps) != expected_steps:
        raise RuntimeError(f"API returned {len(steps)} steps, expected {expected_steps}")
    if any(not 1 <= int(step.get("rows", 0)) <= protocol.WINDOW_MAX_ROWS for step in steps):
        raise RuntimeError(
            f"API returned a window outside the MCDMA 1..{protocol.WINDOW_MAX_ROWS} row limit"
        )
    if sum(int(step.get("accepted_tokens", 0)) for step in steps) != int(usage["completion_tokens"]):
        raise RuntimeError("accepted window tokens differ from the completion token count")
    verified_drafts = sum(int(step.get("verified_drafts", 0)) for step in decode_steps_raw)
    accepted_drafts = sum(int(step.get("accepted_drafts", 0)) for step in decode_steps_raw)
    if verified_drafts != int(metrics.get("verified_drafts", -1)):
        raise RuntimeError("per-window verified drafts differ from request metrics")
    if accepted_drafts != int(metrics.get("accepted_drafts", -1)):
        raise RuntimeError("per-window accepted drafts differ from request metrics")
    if float(metrics.get("metal_s", 0)) <= 0 or any(float(step.get("metal_s", 0)) <= 0 for step in steps):
        raise RuntimeError("Mac Metal participation was not measured on every step")
    decode_steps = [float(step["total_s"]) for step in decode_steps_raw]
    all_steps = [float(step["total_s"]) for step in steps]
    steady = decode_steps or all_steps
    decode_total_s = sum(decode_steps)
    if decode_rounds and decode_total_s <= 0:
        raise RuntimeError("API returned no positive decode-window time")
    median_step = statistics.median(steady)
    hiccups = sum(value > 2 * median_step for value in steady)
    stalls = sum(value > max(4 * median_step, 0.150) for value in steady)
    request_total_s = float(metrics.get("request_to_completion_s", 0))
    request_first_token_s = float(metrics.get("request_to_first_token_s", 0))
    if request_total_s <= 0 or request_first_token_s <= 0:
        raise RuntimeError("API returned no request-receipt timing evidence")
    wall_delta_s = abs(wall_s - request_total_s)
    wall_delta_fraction = wall_delta_s / wall_s
    if wall_delta_s > max(CLIENT_SERVER_ABSOLUTE_TOLERANCE_S, 0.02 * wall_s):
        raise RuntimeError(
            "client wall and server request time differ by more than "
            f"50 ms or 2% ({wall_delta_s:.6f}s, {wall_delta_fraction:.2%})"
        )
    accounted_step_s = sum(all_steps)
    unaccounted_request_s = max(0.0, request_total_s - accounted_step_s)
    unaccounted_stall_limit_s = max(4 * median_step, 0.150)
    if unaccounted_request_s > unaccounted_stall_limit_s:
        raise RuntimeError(
            "request time not accounted for by measured steps exceeds the stall limit"
        )
    text = choice["message"]["content"]
    return {
        "run": run,
        "wall_s": wall_s,
        "client_overhead_s": max(0.0, wall_s - request_total_s),
        "client_server_wall_delta_s": wall_delta_s,
        "client_server_wall_delta_fraction": wall_delta_fraction,
        "accounted_step_s": accounted_step_s,
        "unaccounted_request_s": unaccounted_request_s,
        "unaccounted_stall_limit_s": unaccounted_stall_limit_s,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_count,
        "completion_tokens": int(usage["completion_tokens"]),
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_ids": token_ids,
        "token_ids_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "expected_mcdma_calls": 2 + 2 * expected_steps,
        "decode_total_s": decode_total_s,
        "decode_tokens_per_s": (
            max(0, int(usage["completion_tokens"]) - 1) / decode_total_s
            if decode_total_s else 0.0
        ),
        "decode_step_s": summary(steady),
        "final_128_decode_step_s": summary(steady[-128:]),
        "stall_count": stalls,
        "hiccup_count": hiccups,
        "request_first_token_s": request_first_token_s,
        "request_total_s": request_total_s,
        **metrics,
    }


def health_url(api_url: str) -> str:
    marker = "/v1/chat/completions"
    if not api_url.endswith(marker):
        raise ValueError("benchmark URL must end in /v1/chat/completions")
    return api_url[:-len(marker)] + "/health"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=Path("experiments/three_machine/native-token-references.json"),
    )
    parser.add_argument("--reference-id", default="B1-decode")
    parser.add_argument(
        "--reference-sha256",
        default=os.environ.get("THREE_MACHINE_NATIVE_REFERENCE_SHA256"),
    )
    parser.add_argument("--expected-native-image-id", required=True)
    parser.add_argument("--warmup-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    if args.repeats < 3 or args.warmups < 1:
        raise SystemExit("repeats must be >=3 and warmups >=1")
    if not 0 <= args.warmup_tolerance <= 1:
        raise SystemExit("warmup tolerance must be between zero and one")
    reference_file_sha256 = require_reference_hash(args.reference_file, args.reference_sha256)
    references = json.loads(args.reference_file.read_text())
    source = references.get("source")
    if not isinstance(source, dict) or source.get("image_id") != args.expected_native_image_id:
        raise SystemExit("native reference was not captured from the pinned TensorFold image")
    reference = (references.get("entries") or {}).get(args.reference_id)
    if not isinstance(reference, dict) or reference.get("suite") != "benchmark":
        raise SystemExit(f"missing native benchmark reference {args.reference_id}")
    prompt = reference.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise SystemExit("benchmark prompt must not be empty")
    reference_max_tokens = int(reference.get("max_tokens", 0))
    if reference_max_tokens < 1:
        raise SystemExit("native benchmark reference has no valid max_tokens")
    if args.max_tokens is not None and args.max_tokens != reference_max_tokens:
        raise SystemExit("requested max_tokens differs from the frozen reference")
    max_tokens = reference_max_tokens
    expected_prompt_tokens = reference.get("prompt_tokens")
    if type(expected_prompt_tokens) is not int or expected_prompt_tokens < 1:
        raise SystemExit("native benchmark reference has no valid prompt_tokens")
    expected_token_ids_sha256 = reference.get("token_ids_sha256")
    expected_output_sha256 = reference.get("output_sha256")
    for label, value in (
        ("token IDs", expected_token_ids_sha256),
        ("output", expected_output_sha256),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise SystemExit(f"native benchmark reference has no {label} hash")
    if hashlib.sha256(prompt.encode()).hexdigest() != reference.get("prompt_sha256"):
        raise SystemExit("native benchmark prompt hash is invalid")
    require_length = reference.get("finish_reason") == "length"

    health_before = get_json(health_url(args.url), min(args.timeout, 10))
    if health_before.get("status") != "ready":
        raise RuntimeError("three-machine API was not ready before the benchmark")
    for field in IDENTITY_FIELDS:
        if health_before.get(field) in (None, "", "unknown"):
            raise RuntimeError(f"health is missing the deployed {field}")

    warmups = []
    for index in range(args.warmups):
        response, wall_s = request(args.url, prompt, max_tokens, args.timeout)
        warmups.append(normalize(
            response, wall_s, -(args.warmups - index), max_tokens,
            require_length, expected_prompt_tokens,
        ))

    health_after_warmup = get_json(health_url(args.url), min(args.timeout, 10))
    if health_after_warmup.get("status") != "ready":
        raise RuntimeError("three-machine API was not ready after warmup")
    for field in IDENTITY_FIELDS:
        if health_after_warmup.get(field) != health_before.get(field):
            raise RuntimeError(f"deployed {field} changed during warmup")

    runs = []
    for index in range(args.repeats):
        response, wall_s = request(args.url, prompt, max_tokens, args.timeout)
        runs.append(normalize(
            response, wall_s, index + 1, max_tokens, require_length,
            expected_prompt_tokens,
        ))

    health_after = get_json(health_url(args.url), min(args.timeout, 10))
    if health_after.get("status") != "ready":
        raise RuntimeError("three-machine API was not ready after the benchmark")
    for field in IDENTITY_FIELDS:
        if health_after.get(field) != health_before.get(field):
            raise RuntimeError(f"deployed {field} changed during the benchmark")
    if int(health_after.get("failed_requests", -1)) != int(health_before.get("failed_requests", -2)):
        raise RuntimeError("failed request count changed during the benchmark")
    if int(health_before.get("failed_requests", -1)) != 0:
        raise RuntimeError("service had a nonzero failed request count before the benchmark")
    expected_requests = args.warmups + args.repeats
    if (int(health_after.get("completed_requests", -1))
            - int(health_before.get("completed_requests", -1))) != expected_requests:
        raise RuntimeError("completed request count does not match the benchmark request count")

    warm_memory = health_after_warmup.get("memory") or {}
    after_memory = health_after.get("memory") or {}
    for key in ("mlx_active_bytes", "mlx_cache_bytes", "mlx_peak_bytes"):
        if type(warm_memory.get(key)) is not int or type(after_memory.get(key)) is not int:
            raise RuntimeError(f"health is missing numeric {key} memory evidence")
    warm_held = int(warm_memory["mlx_active_bytes"]) + int(warm_memory["mlx_cache_bytes"])
    after_held = int(after_memory["mlx_active_bytes"]) + int(after_memory["mlx_cache_bytes"])
    memory_limit = max(64 << 20, int(warm_held * 0.01))
    if after_held - warm_held > memory_limit:
        raise RuntimeError("Mac MLX memory grew beyond the post-warmup limit")

    all_runs = warmups + runs
    output_hashes = {item["output_sha256"] for item in all_runs}
    token_hashes = {item["token_ids_sha256"] for item in all_runs}
    if len(output_hashes) != 1 or len(token_hashes) != 1:
        raise RuntimeError("greedy output changed across identical requests")
    output_sha256 = next(iter(output_hashes))
    if output_sha256 != expected_output_sha256:
        raise RuntimeError("output differs from the registered reference hash")
    token_ids_sha256 = next(iter(token_hashes))
    if token_ids_sha256 != expected_token_ids_sha256:
        raise RuntimeError("token IDs differ from the registered native reference")
    if any(item["stall_count"] for item in all_runs):
        raise RuntimeError("at least one decode step exceeded the registered stall limit")
    if any(item["hiccup_count"] > 2 for item in all_runs):
        raise RuntimeError("at least one run had more than two decode hiccups")
    if any(item["decode_step_s"]["p99"] > 1.3 * item["decode_step_s"]["median"] for item in all_runs):
        raise RuntimeError("at least one run exceeded the p99 decode-tail limit")

    measured_step_median = statistics.median(
        item["decode_step_s"]["median"] for item in runs
    )
    warmup_step_median = statistics.median(
        item["decode_step_s"]["median"] for item in warmups
    )
    warmup_delta = abs(warmup_step_median - measured_step_median) / measured_step_median
    if warmup_delta > args.warmup_tolerance:
        raise RuntimeError(
            f"warmup step median differs from measured runs by {warmup_delta:.2%}"
        )

    result = {
        "schema": 2,
        "captured_at_unix_ns": time.time_ns(),
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "reference_prompt_tokens": expected_prompt_tokens,
        "max_tokens": max_tokens,
        "warmup_tolerance": args.warmup_tolerance,
        "warmup_delta_fraction": warmup_delta,
        "output_sha256": output_sha256,
        "token_ids_sha256": token_ids_sha256,
        "reference_id": args.reference_id,
        "reference_file_sha256": reference_file_sha256,
        "reference_token_ids_sha256": expected_token_ids_sha256,
        "health_before": health_before,
        "health_after_warmup": health_after_warmup,
        "health_after": health_after,
        "mlx_post_warmup_growth_bytes": after_held - warm_held,
        "expected_mcdma_calls": sum(item["expected_mcdma_calls"] for item in all_runs),
        "wall_s": summary([float(item["wall_s"]) for item in runs]),
        "elapsed_s": summary([float(item["elapsed_s"]) for item in runs]),
        "first_token_s": summary([float(item["request_first_token_s"]) for item in runs]),
        "tokens_per_s": summary([float(item["tokens_per_s"]) for item in runs]),
        "decode_tokens_per_s": summary([
            float(item["decode_tokens_per_s"]) for item in runs
        ]),
        "decode_step_median_s": summary([
            float(item["decode_step_s"]["median"]) for item in runs
        ]),
        "decode_step_max_s": summary([
            float(item["decode_step_s"]["max"]) for item in runs
        ]),
        "final_128_decode_step_median_s": summary([
            float(item["final_128_decode_step_s"]["median"]) for item in runs
        ]),
        "warmups": warmups,
        "runs": runs,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
