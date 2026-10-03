#!/usr/bin/env python3
"""Measure the pinned native two-Spark service against one frozen reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


def get(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    began = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    return body, time.perf_counter() - began


def token_ids(response: dict[str, Any]) -> list[int]:
    values = (response.get("tensorfold") or {}).get("token_ids")
    if values is None:
        values = response["choices"][0].get("token_ids")
    if not isinstance(values, list) or not values or any(type(value) is not int for value in values):
        raise RuntimeError("native completion returned no token IDs")
    return values


def digest_tokens(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


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


def health_ready(health: dict[str, Any]) -> bool:
    return (
        health.get("ok") is True and health.get("backend") == "tensorfold"
    ) or health.get("status") in {"ok", "ready"}


def normalize(response: dict[str, Any], wall_s: float, reference: dict[str, Any], run: int) -> dict[str, Any]:
    if response.get("model") != "GLM-5.3-Flash-EXL3":
        raise RuntimeError("native response returned a different model")
    runtime = response.get("tensorfold")
    if not isinstance(runtime, dict) or runtime.get("drafts") is not True:
        raise RuntimeError("native response does not prove MTP drafting was enabled")
    usage = response["usage"]
    ids = token_ids(response)
    text = response["choices"][0]["message"]["content"]
    if int(usage["prompt_tokens"]) != int(reference["prompt_tokens"]):
        raise RuntimeError("native prompt-token count differs from the frozen reference")
    if int(usage["completion_tokens"]) != len(ids):
        raise RuntimeError("native completion count differs from returned token IDs")
    if digest_tokens(ids) != reference["token_ids_sha256"]:
        raise RuntimeError("MTP token IDs differ from the MTP-off native reference")
    if hashlib.sha256(text.encode()).hexdigest() != reference["output_sha256"]:
        raise RuntimeError("MTP text differs from the MTP-off native reference")
    if response["choices"][0]["finish_reason"] != reference["finish_reason"]:
        raise RuntimeError("native finish reason differs from the frozen reference")
    completion_tokens = len(ids)
    decode_s = float(runtime.get("decode_s", 0))
    if decode_s <= 0:
        raise RuntimeError("native response returned no positive decode time")
    return {
        "run": run,
        "wall_s": wall_s,
        "completion_tokens": completion_tokens,
        "wall_tokens_per_s": completion_tokens / wall_s,
        "decode_s": decode_s,
        "decode_tokens_per_s": max(0, completion_tokens - 1) / decode_s,
        "token_ids_sha256": digest_tokens(ids),
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "tensorfold": runtime,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--reference-id", default="B1-decode")
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.round < 1 or args.warmups < 1 or args.repeats < 3:
        raise SystemExit("round and warmups must be positive; repeats must be at least three")

    reference_bytes = args.reference_file.read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != args.reference_sha256:
        raise SystemExit("native reference file differs from its pinned SHA-256")
    references = json.loads(reference_bytes)
    reference = (references.get("entries") or {}).get(args.reference_id)
    if not isinstance(reference, dict) or reference.get("suite") != "benchmark":
        raise SystemExit("requested native benchmark reference is missing")
    attestation_bytes = args.attestation.read_bytes()
    attestation = json.loads(attestation_bytes)
    if attestation.get("image_id") != args.expected_image_id or attestation.get("drafter") != "mtp":
        raise SystemExit("native attestation does not prove the pinned MTP service")

    base = args.url.rstrip("/")
    health_before = get(base + "/health", 10)
    if not health_ready(health_before):
        raise RuntimeError("native service was not healthy before the benchmark")
    payload = {
        "model": "GLM-5.3-Flash-EXL3",
        "messages": [{"role": "user", "content": reference["prompt"]}],
        "max_tokens": int(reference["max_tokens"]),
        "temperature": 0,
        "draft": True,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    warmups = []
    for index in range(args.warmups):
        response, wall_s = post(base + "/v1/chat/completions", payload, args.timeout)
        warmups.append(normalize(response, wall_s, reference, -(args.warmups - index)))
    runs = []
    for index in range(args.repeats):
        response, wall_s = post(base + "/v1/chat/completions", payload, args.timeout)
        runs.append(normalize(response, wall_s, reference, index + 1))
    health_after = get(base + "/health", 10)
    if not health_ready(health_after) or health_after.get("backend") != health_before.get("backend"):
        raise RuntimeError("native service changed or became unhealthy during the benchmark")

    result = {
        "schema": 1,
        "kind": "native-two-spark-benchmark-round",
        "round": args.round,
        "captured_at_unix_ns": time.time_ns(),
        "reference_id": args.reference_id,
        "reference_sha256": args.reference_sha256,
        "attestation_sha256": hashlib.sha256(attestation_bytes).hexdigest(),
        "source": attestation,
        "health_before": health_before,
        "health_after": health_after,
        "wall_s": summary([item["wall_s"] for item in runs]),
        "wall_tokens_per_s": summary([item["wall_tokens_per_s"] for item in runs]),
        "decode_tokens_per_s": summary([item["decode_tokens_per_s"] for item in runs]),
        "warmups": warmups,
        "runs": runs,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
