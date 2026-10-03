#!/usr/bin/env python3
"""Measure steady, first-after-idle, and second-after-idle request latency."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


MODEL = "GLM-5.3-Flash-TensorFold-MCDMA-3Machine"
IDENTITY_FIELDS = (
    "deployment_id", "commit", "project_manifest", "config_sha256", "image_id",
    "link_generation",
)


def get(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post(url: str, prompt: str, max_tokens: int, timeout: float) -> dict[str, Any]:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "stream": False,
    }, separators=(",", ":")).encode()
    request = urllib.request.Request(url, payload, {"Content-Type": "application/json"})
    began = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    result["_client_wall_s"] = time.perf_counter() - began
    return result


def health_url(api_url: str) -> str:
    marker = "/v1/chat/completions"
    if not api_url.endswith(marker):
        raise ValueError("idle URL must end in /v1/chat/completions")
    return api_url[:-len(marker)] + "/health"


def request_evidence(
    url: str, prompt: str, max_tokens: int, expected_prompt_tokens: int,
    expected_token_hash: str, timeout: float,
) -> dict[str, Any]:
    response = post(url, prompt, max_tokens, timeout)
    metrics = response["three_machine_metrics"]
    usage = response["usage"]
    token_ids = [int(value) for value in metrics["generated_token_ids"]]
    token_hash = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode()
    ).hexdigest()
    if token_hash != expected_token_hash:
        raise RuntimeError("idle request token IDs differ from native reference")
    if int(usage["prompt_tokens"]) != expected_prompt_tokens:
        raise RuntimeError("idle request prompt-token count differs from native reference")
    steps = metrics.get("steps")
    if not isinstance(steps, list) or not steps:
        raise RuntimeError("idle request returned no protocol steps")
    request_time = float(metrics.get("request_to_completion_s", 0))
    if request_time <= 0:
        raise RuntimeError("idle request has no server completion time")
    return {
        "request_id": int(metrics["request_id"]),
        "request_s": request_time,
        "client_wall_s": float(response["_client_wall_s"]),
        "expected_mcdma_calls": 2 + 2 * len(steps),
        "steps": [
            {"step": int(step["step"]), "prompt": bool(step["prompt"])}
            for step in steps
        ],
        "token_ids_sha256": token_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--reference-id", default="B1-decode")
    parser.add_argument("--expected-native-image-id", required=True)
    parser.add_argument("--idle-seconds", type=float, default=8 * 60 * 60)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.idle_seconds <= 0:
        raise SystemExit("idle-seconds must be positive")
    raw = args.reference_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.reference_sha256:
        raise SystemExit("native reference file SHA-256 differs from the pinned value")
    references = json.loads(raw)
    if (references.get("source") or {}).get("image_id") != args.expected_native_image_id:
        raise SystemExit("native reference was not captured from the pinned TensorFold image")
    reference = (references.get("entries") or {}).get(args.reference_id)
    if not isinstance(reference, dict) or reference.get("suite") != "benchmark":
        raise SystemExit("idle benchmark reference is missing")
    prompt = reference.get("prompt")
    expected_prompt_tokens = reference.get("prompt_tokens")
    max_tokens = reference.get("max_tokens")
    expected_token_hash = reference.get("token_ids_sha256")
    if not isinstance(prompt, str) or type(expected_prompt_tokens) is not int \
            or type(max_tokens) is not int or not isinstance(expected_token_hash, str):
        raise SystemExit("idle benchmark reference is malformed")

    before = get(health_url(args.url), 10)
    if before.get("status") != "ready" or before.get("failed_requests") != 0:
        raise RuntimeError("three-machine API is not clean and ready before idle measurement")
    for field in IDENTITY_FIELDS:
        if before.get(field) in (None, "", "unknown"):
            raise RuntimeError(f"idle health is missing {field}")
    steady = [
        request_evidence(
            args.url, prompt, max_tokens, expected_prompt_tokens,
            expected_token_hash, args.timeout,
        )
        for _ in range(3)
    ]
    time.sleep(args.idle_seconds)
    first = request_evidence(
        args.url, prompt, max_tokens, expected_prompt_tokens,
        expected_token_hash, args.timeout,
    )
    second = request_evidence(
        args.url, prompt, max_tokens, expected_prompt_tokens,
        expected_token_hash, args.timeout,
    )
    after = get(health_url(args.url), 10)
    for field in IDENTITY_FIELDS:
        if after.get(field) != before.get(field):
            raise RuntimeError(f"idle deployment {field} changed")
    if after.get("status") != "ready" or after.get("failed_requests") != 0:
        raise RuntimeError("three-machine API failed during idle measurement")
    if int(after["completed_requests"]) - int(before["completed_requests"]) != 5:
        raise RuntimeError("idle request count does not match health counters")
    requests = [*steady, first, second]
    print(json.dumps({
        "schema": 1,
        "health_before": before,
        "health_after": after,
        "steady_runs_s": [value["request_s"] for value in steady],
        "idle_seconds": args.idle_seconds,
        "first_after_idle_s": first["request_s"],
        "second_after_idle_s": second["request_s"],
        "expected_mcdma_calls": sum(value["expected_mcdma_calls"] for value in requests),
        "requests": requests,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
