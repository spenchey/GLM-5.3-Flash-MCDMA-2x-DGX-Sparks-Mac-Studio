#!/usr/bin/env python3
"""Measure the one-request ceiling of an OpenAI-compatible TensorFold service."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any, Mapping

from experiments.three_machine.concurrent_capacity import get_json, post_stream


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} was not numeric")
    result = float(value)
    if positive and result <= 0:
        raise RuntimeError(f"{label} was not positive")
    return result


def normalize(run: Mapping[str, Any]) -> dict[str, Any]:
    runtime = run.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("TensorFold returned no runtime evidence")
    tokens = int(run["tokens"])
    prefill = runtime.get("prefill_s", runtime.get("prefill_seconds"))
    decode = runtime.get("decode_s")
    if decode is None:
        rate = runtime.get("tokens_per_second")
        if rate is not None and float(rate) > 0 and tokens > 1:
            decode = (tokens - 1) / float(rate)
    prefill_seconds = _number(prefill, "server prefill time", positive=True)
    decode_seconds = _number(decode, "server decode time", positive=True)
    chunk_times = [float(value) for value in run.get("client_chunk_seconds", [])]
    chunk_gaps = [right - left for left, right in zip(chunk_times, chunk_times[1:])]
    return {
        "total_seconds": _number(run["seconds"], "client total time", positive=True),
        "ttft_seconds": _number(run["first_token_seconds"], "client TTFT", positive=True),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (tokens - 1) / decode_seconds,
        "completion_tokens": tokens,
        "token_ids": list(run["token_ids"]),
        "token_sha256": run["token_sha256"],
        "text_sha256": run["text_sha256"],
        "finish_reason": run["finish_reason"],
        "client_chunk_seconds": chunk_times,
        "client_chunk_gaps_seconds": chunk_gaps,
        "runtime": dict(runtime),
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="GLM-5.3-Flash-Portable")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--draft", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_tokens <= 1 or args.warmups < 1 or args.repeats < 3:
        parser.error("max-tokens must be >1, warmups >=1, and repeats >=3")

    base = args.url.rstrip("/")
    health_before = get_json(base + "/health", min(args.timeout, 10.0))
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "draft": args.draft,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    raw = [
        post_stream(base + "/v1/chat/completions", payload, args.timeout)
        for _ in range(args.warmups + args.repeats)
    ]
    runs = [normalize(item) for item in raw]
    identities = {(item["token_sha256"], item["text_sha256"]) for item in runs}
    if len(identities) != 1:
        raise RuntimeError("identical one-request rounds returned different output")
    if any(item["completion_tokens"] != args.max_tokens for item in runs):
        raise RuntimeError("one-request round stopped before the fixed output length")
    health_after = get_json(base + "/health", min(args.timeout, 10.0))
    measured = runs[args.warmups:]
    result = {
        "schema": 1,
        "event": "glm_native_single_stream_ceiling",
        "parallel": 1,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "draft": args.draft,
        "token_sha256": measured[0]["token_sha256"],
        "text_sha256": measured[0]["text_sha256"],
        "total_seconds": _summary([item["total_seconds"] for item in measured]),
        "ttft_seconds": _summary([item["ttft_seconds"] for item in measured]),
        "prefill_seconds": _summary([item["prefill_seconds"] for item in measured]),
        "decode_seconds": _summary([item["decode_seconds"] for item in measured]),
        "decode_tokens_per_second": _summary([
            item["decode_tokens_per_second"] for item in measured
        ]),
        "health_before": health_before,
        "health_after": health_after,
        "warmups": runs[:args.warmups],
        "runs": measured,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
