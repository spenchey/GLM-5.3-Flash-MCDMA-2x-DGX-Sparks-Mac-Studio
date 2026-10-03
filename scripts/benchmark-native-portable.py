#!/usr/bin/env python3
"""Measure concurrent capacity of the same portable model on two Sparks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import statistics
import time
import urllib.request
from typing import Any


def get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def token_ids(response: dict[str, Any]) -> list[int]:
    values = (response.get("tensorfold") or {}).get("token_ids")
    if values is None:
        values = response["choices"][0].get("token_ids")
    if not isinstance(values, list) or not values or any(type(value) is not int for value in values):
        raise RuntimeError("native completion returned no token IDs")
    return values


def digest(values: list[int]) -> str:
    encoded = b"".join(int(value).to_bytes(4, "big", signed=False) for value in values)
    return hashlib.sha256(encoded).hexdigest()


def run_batch(
    url: str,
    payload: dict[str, Any],
    *,
    parallel: int,
    timeout: float,
    run: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(post, url, payload, timeout) for _ in range(parallel)]
        responses = [future.result() for future in futures]
    seconds = time.perf_counter() - started
    ids = [token_ids(response) for response in responses]
    hashes = {digest(values) for values in ids}
    if len(hashes) != 1:
        raise RuntimeError("identical concurrent native requests returned different tokens")
    prompt_counts = {int(response["usage"]["prompt_tokens"]) for response in responses}
    if len(prompt_counts) != 1:
        raise RuntimeError("native concurrent requests reported different prompt lengths")
    completion_tokens = sum(len(values) for values in ids)
    if any(
        int(response["usage"]["completion_tokens"]) != len(values)
        for response, values in zip(responses, ids, strict=True)
    ):
        raise RuntimeError("native completion accounting differs from returned token IDs")
    if any(response["choices"][0]["finish_reason"] != "length" for response in responses):
        raise RuntimeError("native benchmark stopped before the fixed reply length")
    return {
        "run": run,
        "parallel": parallel,
        "seconds": seconds,
        "prompt_tokens": prompt_counts.pop(),
        "completion_tokens": completion_tokens,
        "aggregate_pipeline_tokens_per_second": completion_tokens / seconds,
        "token_sha256": hashes.pop(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="GLM-5.3-Flash-Portable")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    if not 1 <= args.parallel <= 16:
        parser.error("--parallel must be between 1 and 16")
    if args.max_tokens < 1 or args.warmups < 1 or args.repeats < 3:
        parser.error("max-tokens must be positive, warmups >= 1, and repeats >= 3")

    base = args.url.rstrip("/")
    health_before = get_json(base + "/health", min(args.timeout, 10.0))
    if health_before.get("status") not in {"ok", "ready"} and health_before.get("ok") is not True:
        raise RuntimeError("native portable service was not healthy before measurement")
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "draft": True,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    warmups = [
        run_batch(
            base + "/v1/chat/completions",
            payload,
            parallel=args.parallel,
            timeout=args.timeout,
            run=-(args.warmups - index),
        )
        for index in range(args.warmups)
    ]
    runs = [
        run_batch(
            base + "/v1/chat/completions",
            payload,
            parallel=args.parallel,
            timeout=args.timeout,
            run=index + 1,
        )
        for index in range(args.repeats)
    ]
    hashes = {item["token_sha256"] for item in warmups + runs}
    if len(hashes) != 1:
        raise RuntimeError("native portable output changed across benchmark rounds")
    health_after = get_json(base + "/health", min(args.timeout, 10.0))
    if health_after.get("status") not in {"ok", "ready"} and health_after.get("ok") is not True:
        raise RuntimeError("native portable service was not healthy after measurement")
    rates = [float(item["aggregate_pipeline_tokens_per_second"]) for item in runs]
    print(json.dumps({
        "schema": 1,
        "kind": "portable-native-two-spark-concurrency",
        "parallel": args.parallel,
        "max_tokens": args.max_tokens,
        "token_sha256": hashes.pop(),
        "aggregate_pipeline_tokens_per_second": {
            "median": statistics.median(rates),
            "min": min(rates),
            "max": max(rates),
        },
        "health_before": health_before,
        "health_after": health_after,
        "warmups": warmups,
        "runs": runs,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
