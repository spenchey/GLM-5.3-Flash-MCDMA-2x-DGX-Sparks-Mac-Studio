#!/usr/bin/env python3
"""Reproduce MiaAI/sparkDash's one-request GLM decode protocol.

The benchmark is intentionally client-observed.  It reports complete request
time as the primary latency and preserves TTFT plus post-first-token decode
speed as diagnostic boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from typing import Any, Callable, Iterable, Mapping
import urllib.request


PROMPTS = {
    "structured": (
        "Count from 1 to 200. Output only the numbers, separated by spaces. "
        "No other text."
    ),
    "prose": (
        "Write a detailed step-by-step explanation of how a hash map works, "
        "including collision handling, resizing, and time complexity. Be thorough."
    ),
    "code": (
        "binary_search\n"
        "def binary_search(nums, target) -> int: index of target in a sorted list, "
        "or -1.\n"
        "Output only Python source. No comments, no docstrings, no markdown fences. "
        "Then add tests and the helpers this needs. Keep writing code."
    ),
}

CODE_WARMUP_PROMPT = (
    "warmup_noop\n"
    "def warmup_noop(x): return x unchanged.\n"
    "Output only Python source. No comments, no docstrings, no markdown fences. "
    "Then add tests and the helpers this needs. Keep writing code."
)


class BenchmarkError(RuntimeError):
    """The endpoint did not satisfy the pinned benchmark contract."""


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise BenchmarkError("cannot summarize an empty metric")
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def parse_sse(
    lines: Iterable[bytes],
    started: float,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Parse an OpenAI stream using sparkDash's first/last content window."""

    first_at: float | None = None
    last_at: float | None = None
    chunks: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    saw_done = False
    arrivals: list[float] = []

    for raw in lines:
        raw = raw.strip()
        if not raw.startswith(b"data:"):
            continue
        data = raw[5:].strip()
        if data == b"[DONE]":
            saw_done = True
            break
        if not data:
            continue
        item = json.loads(data)
        if item.get("error"):
            raise BenchmarkError(f"stream returned an error: {item['error']}")
        if isinstance(item.get("usage"), Mapping):
            usage = dict(item["usage"])
        choices = item.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        emitted = str(delta.get("reasoning_content") or "") + str(
            delta.get("content") or ""
        )
        if emitted:
            arrived = clock()
            first_at = arrived if first_at is None else first_at
            last_at = arrived
            arrivals.append(arrived - started)
            chunks.append(emitted)
        finish_reason = choice.get("finish_reason") or finish_reason

    finished = clock()
    if not saw_done:
        raise BenchmarkError("stream ended without [DONE]")
    if first_at is None or last_at is None:
        raise BenchmarkError("stream emitted no content")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int):
        raise BenchmarkError("stream returned no integer completion token count")
    if completion_tokens < 2:
        raise BenchmarkError("stream returned fewer than two completion tokens")
    decode_seconds = last_at - first_at
    if decode_seconds <= 0:
        raise BenchmarkError("stream had no measurable decode window")
    text = "".join(chunks)
    return {
        "total_seconds": finished - started,
        "ttft_seconds": first_at - started,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (completion_tokens - 1) / decode_seconds,
        "completion_tokens": completion_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "finish_reason": finish_reason,
        "text_sha256": _digest(text),
        "text": text,
        "client_chunk_seconds": arrivals,
    }


def request_body(model: str, prompt: str, max_tokens: int, *, force_length: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {
            "enable_thinking": False,
            "thinking": False,
            "thinking_mode": "disabled",
        },
    }
    if force_length:
        body.update({"min_tokens": max_tokens, "ignore_eos": True, "stop": []})
    return body


def post_stream(url: str, body: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(dict(body), separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return parse_sse(response, started)


def get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def benchmark_type(
    *,
    base_url: str,
    model: str,
    prompt_type: str,
    max_tokens: int,
    warmup_tokens: int,
    repeats: int,
    timeout: float,
) -> dict[str, Any]:
    prompt = PROMPTS[prompt_type]
    warmup_prompt = CODE_WARMUP_PROMPT if prompt_type == "code" else prompt
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    warmup = post_stream(
        endpoint,
        request_body(model, warmup_prompt, warmup_tokens, force_length=False),
        timeout,
    )
    runs = [
        post_stream(
            endpoint,
            request_body(model, prompt, max_tokens, force_length=True),
            timeout,
        )
        for _ in range(repeats)
    ]
    if any(run["completion_tokens"] != max_tokens for run in runs):
        raise BenchmarkError(f"{prompt_type}: a measured run ended before {max_tokens} tokens")
    hashes = {run["text_sha256"] for run in runs}
    if len(hashes) != 1:
        raise BenchmarkError(f"{prompt_type}: greedy measured runs returned different bytes")
    return {
        "prompt_type": prompt_type,
        "prompt": prompt,
        "prompt_sha256": _digest(prompt),
        "output_text_sha256": runs[0]["text_sha256"],
        "total_seconds": _summary([float(run["total_seconds"]) for run in runs]),
        "ttft_seconds": _summary([float(run["ttft_seconds"]) for run in runs]),
        "decode_seconds": _summary([float(run["decode_seconds"]) for run in runs]),
        "decode_tokens_per_second": _summary(
            [float(run["decode_tokens_per_second"]) for run in runs]
        ),
        "warmup": warmup,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=sorted(PROMPTS),
        default=["prose", "structured", "code"],
    )
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.max_tokens < 64 or args.warmup_tokens < 1 or args.repeats < 3:
        parser.error("max-tokens >=64, warmup-tokens >=1, and repeats >=3 are required")

    base = args.url.rstrip("/")
    short_timeout = min(args.timeout, 10.0)
    health_before = get_json(base + "/health", short_timeout)
    models = get_json(base + "/v1/models", short_timeout)
    results = [
        benchmark_type(
            base_url=base,
            model=args.model,
            prompt_type=prompt_type,
            max_tokens=args.max_tokens,
            warmup_tokens=args.warmup_tokens,
            repeats=args.repeats,
            timeout=args.timeout,
        )
        for prompt_type in args.types
    ]
    health_after = get_json(base + "/health", short_timeout)
    print(
        json.dumps(
            {
                "schema": 1,
                "event": "mia_sparkdash_c1_control",
                "protocol": {
                    "sparkdash_commit": "b4228a330a7877dcb5a30516500d57e26affa45a",
                    "temperature": 0,
                    "top_p": 1,
                    "thinking": False,
                    "warmup_tokens": args.warmup_tokens,
                    "max_tokens": args.max_tokens,
                    "repeats": args.repeats,
                },
                "model_requested": args.model,
                "models_response": models,
                "health_before": health_before,
                "health_after": health_after,
                "results": results,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
