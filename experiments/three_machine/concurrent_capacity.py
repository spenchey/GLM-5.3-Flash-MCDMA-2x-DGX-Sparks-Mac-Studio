"""Run same-width native and MCDMA-fed capacity rounds in one process.

The JSON-lines worker keeps the Mac model loaded while an outer harness
counterbalances the two arms and samples the MCDMA counters around every
round. All latency values are observed at the client. In the candidate arm,
Mac TTFT includes cache transfer and cache import.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Sequence
import urllib.request

from experiments.three_machine.cache_handoff import HandoffError, prompt_digest
from experiments.three_machine.cuda_prefill import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from experiments.three_machine.mac_decode import (
    _load_runtime,
    _verify_snapshot_path,
    render_user_prompt,
    run_decode,
    run_handoff_round,
)


NativePost = Callable[[str, dict[str, Any], float], dict[str, Any]]


def _barrier_clock(parties: int) -> tuple[threading.Barrier, dict[str, float]]:
    """Return a barrier whose release defines the one clock for a round."""

    origin: dict[str, float] = {}

    def mark() -> None:
        origin["value"] = time.perf_counter()

    return threading.Barrier(parties, action=mark), origin


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _worker_pidfile(path: Path, worker_id: str):
    """Publish one exact remote worker identity and remove only that identity."""

    if not worker_id or any(value not in "abcdefghijklmnopqrstuvwxyz0123456789-" for value in worker_id):
        raise HandoffError("worker ID contains unsupported characters")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    record = {
        "schema": 1,
        "pid": os.getpid(),
        "worker_id": worker_id,
        "module": "experiments.three_machine.concurrent_capacity",
    }
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError as exc:
        raise HandoffError(f"paired worker PID file already exists: {path}") from exc
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        yield record
    finally:
        try:
            current = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            current = None
        if current == record:
            path.unlink()


def _runtime_evidence(args: argparse.Namespace, tensorfold_module: Any) -> dict[str, Any]:
    manifest_script = Path(__file__).resolve().parents[2] / "scripts" / "tree-manifest.py"
    source_sha256 = subprocess.check_output(
        [
            sys.executable,
            str(manifest_script),
            "--exclude-relative",
            ".tensorfold-metal-stage.json",
            str(args.tensorfold_source),
        ],
        text=True,
    ).strip()
    python_sha256 = _sha256_file(Path(sys.executable).resolve())
    python_env = (
        platform.python_version()
        + ",mlx="
        + importlib.metadata.version("mlx")
        + ",mlx-lm="
        + importlib.metadata.version("mlx-lm")
        + ",numpy="
        + importlib.metadata.version("numpy")
    )
    marker = json.loads((args.tensorfold_source / ".tensorfold-metal-stage.json").read_text())
    module_file = getattr(tensorfold_module, "__file__", None)
    module_version = getattr(tensorfold_module, "__version__", None)
    if not isinstance(module_file, str) or not module_file:
        raise HandoffError("imported TensorFold has no module path")
    module_path = Path(module_file).resolve()
    source_path = args.tensorfold_source.resolve()
    try:
        module_path.relative_to(source_path)
    except ValueError as exc:
        raise HandoffError("imported TensorFold is outside the frozen source tree") from exc
    expected_version = str(marker.get("upstream_tag", "")).removeprefix("v")
    if not expected_version or module_version != expected_version:
        raise HandoffError("imported TensorFold version differs from the frozen source marker")
    if source_sha256 != args.expected_tensorfold_tree_sha256:
        raise HandoffError("Mac TensorFold source tree differs from the frozen benchmark pin")
    if python_sha256 != args.expected_python_sha256:
        raise HandoffError("Mac Python executable differs from the frozen benchmark pin")
    if python_env != args.expected_python_env:
        raise HandoffError("Mac Python/MLX environment differs from the frozen benchmark pin")
    return {
        "tensorfold_tree_sha256": source_sha256,
        "tensorfold_upstream_tag": marker.get("upstream_tag"),
        "tensorfold_upstream_commit": marker.get("upstream_commit"),
        "tensorfold_module_path": str(module_path),
        "tensorfold_version": module_version,
        "python_sha256": python_sha256,
        "python_env": python_env,
    }


def get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""

    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _parse_sse_lines(lines: Sequence[bytes], started: float) -> dict[str, Any]:
    """Parse one OpenAI-compatible stream and preserve client timings."""

    first_at: float | None = None
    visible: list[str] = []
    reasoning: list[str] = []
    usage: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    finish_reason: str | None = None
    saw_done = False
    for raw in lines:
        if not raw.startswith(b"data: "):
            continue
        data = raw[6:].strip()
        if data == b"[DONE]":
            saw_done = True
            break
        item = json.loads(data)
        if item.get("error"):
            raise HandoffError(f"native TensorFold stream failed: {item['error']}")
        if item.get("usage"):
            usage = dict(item["usage"])
        if item.get("tensorfold"):
            stats = dict(item["tensorfold"])
        choices = item.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        emitted = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
        if emitted and first_at is None:
            first_at = time.perf_counter()
        if delta.get("content"):
            visible.append(str(delta["content"]))
        if delta.get("reasoning_content"):
            reasoning.append(str(delta["reasoning_content"]))
        finish_reason = choice.get("finish_reason") or finish_reason
    finished = time.perf_counter()
    if not saw_done:
        raise HandoffError("native TensorFold stream ended without [DONE]")
    if first_at is None:
        raise HandoffError("native TensorFold stream emitted no token")
    tokens = stats.get("token_ids")
    if not isinstance(tokens, list) or not tokens or any(type(value) is not int for value in tokens):
        raise HandoffError("native TensorFold stream returned no token IDs")
    if int(usage.get("completion_tokens", -1)) != len(tokens):
        raise HandoffError("native TensorFold stream token accounting differs")
    if finish_reason != "length":
        raise HandoffError("native TensorFold stopped before the fixed reply length")
    text = "".join(reasoning + visible)
    return {
        "seconds": finished - started,
        "first_token_seconds": first_at - started,
        "tokens": len(tokens),
        "token_ids": [int(value) for value in tokens],
        "token_sha256": prompt_digest(tokens),
        "text": text,
        "text_sha256": _text_digest(text),
        "finish_reason": finish_reason,
    }


def post_stream(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Measure true client TTFT and completion time from an SSE response."""

    streamed = dict(payload)
    streamed["stream"] = True
    streamed["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        url,
        data=json.dumps(streamed, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return _parse_sse_lines(response, started)


def _payload(prompt: str, max_new_tokens: int) -> dict[str, Any]:
    return {
        "model": "GLM-5.3-Flash-Portable",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_new_tokens),
        "temperature": 0,
        "draft": True,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _validate_native_lane(
    lane: dict[str, Any], expected_tokens: Sequence[int], expected_text_sha256: str
) -> None:
    if lane.get("token_ids") != list(expected_tokens):
        raise HandoffError("native TensorFold tokens differ from the frozen reference")
    if lane.get("text_sha256") != expected_text_sha256:
        raise HandoffError("native TensorFold bytes differ from the frozen reference")


def _round_summary(
    *,
    arm: str,
    run: int,
    parallel: int,
    allocation: dict[str, int],
    seconds: float,
    lanes: list[dict[str, Any]],
    expected_hash: str,
    expected_text_hash: str,
    prompt_sha256: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    completions = [float(item["completion_seconds"]) for item in lanes]
    ttfts = [float(item["first_token_seconds"]) for item in lanes]
    completion_tokens = sum(int(item["tokens"]) for item in lanes)
    if len(lanes) != parallel or not lanes:
        raise HandoffError("capacity round completed the wrong number of lanes")
    if seconds <= 0 or completion_tokens <= 0:
        raise HandoffError("capacity round returned no measurable work")
    return {
        "schema": 2,
        "event": "glm_paired_capacity_round",
        "arm": arm,
        "run": run,
        "parallel": parallel,
        "allocation": allocation,
        "seconds": seconds,
        "completion_tokens": completion_tokens,
        "max_new_tokens": max_new_tokens,
        "aggregate_pipeline_tokens_per_second": completion_tokens / seconds,
        "max_ttft_seconds": max(ttfts),
        "p95_completion_seconds": _percentile(completions, 0.95),
        "max_completion_seconds": max(completions),
        "token_sha256": expected_hash,
        "text_sha256": expected_text_hash,
        "prompt_sha256": prompt_sha256,
        "lanes": lanes,
    }


def run_native_round(
    *,
    prompt: str,
    native_url: str,
    max_new_tokens: int,
    timeout: float,
    local_reference: dict[str, Any],
    run: int,
    parallel: int,
    native_post: NativePost = post_stream,
) -> dict[str, Any]:
    """Run the same-width two-Spark arm with client-observed streaming time."""

    if not 1 <= parallel <= 16:
        raise ValueError("native parallelism must be between 1 and 16")
    expected = list(local_reference["tokens"])
    expected_hash = prompt_digest(expected)
    expected_text_hash = _text_digest(str(local_reference["text"]))
    payload = _payload(prompt, max_new_tokens)
    prompt_sha256 = str(local_reference["prompt_sha256"])
    barrier, origin = _barrier_clock(parallel)

    def lane(index: int) -> dict[str, Any]:
        barrier.wait(timeout=min(timeout, 30.0))
        request_started = time.perf_counter()
        result = native_post(
            native_url.rstrip("/") + "/v1/chat/completions", payload, timeout
        )
        _validate_native_lane(result, expected, expected_text_hash)
        return {
            "kind": "spark",
            "lane": index,
            "tokens": int(result["tokens"]),
            "token_sha256": result["token_sha256"],
            "text_sha256": result["text_sha256"],
            "first_token_seconds": (
                request_started - origin["value"] + float(result["first_token_seconds"])
            ),
            "completion_seconds": request_started - origin["value"] + float(result["seconds"]),
        }

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(lane, index) for index in range(parallel)]
        lanes = [future.result() for future in futures]
    seconds = time.perf_counter() - origin["value"]
    return _round_summary(
        arm="A",
        run=run,
        parallel=parallel,
        allocation={"spark": parallel, "mac": 0},
        seconds=seconds,
        lanes=lanes,
        expected_hash=expected_hash,
        expected_text_hash=expected_text_hash,
        prompt_sha256=prompt_sha256,
        max_new_tokens=max_new_tokens,
    )


def run_round(
    runtime: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    *,
    prompt: str,
    native_url: str,
    mailbox_name: str,
    max_new_tokens: int,
    timeout: float,
    model_id: str,
    model_revision: str,
    local_reference: dict[str, Any],
    run: int,
    spark_parallel: int,
    mac_parallel: int,
    native_post: NativePost = post_stream,
    mac_run: Callable[..., dict[str, Any]] = run_handoff_round,
) -> dict[str, Any]:
    """Run one same-width Spark plus MCDMA-fed Mac candidate arm."""

    if not 0 <= spark_parallel <= 8 or not 1 <= mac_parallel <= 8:
        raise ValueError("Spark parallelism must be 0..8 and Mac parallelism 1..8")

    expected = list(local_reference["tokens"])
    expected_hash = prompt_digest(expected)
    expected_text_hash = _text_digest(str(local_reference["text"]))
    payload = _payload(prompt, max_new_tokens)
    prompt_sha256 = str(local_reference["prompt_sha256"])
    barrier, origin = _barrier_clock(spark_parallel + 1)

    def native_lane(index: int) -> dict[str, Any]:
        barrier.wait(timeout=min(timeout, 30.0))
        request_started = time.perf_counter()
        result = native_post(
            native_url.rstrip("/") + "/v1/chat/completions", payload, timeout
        )
        _validate_native_lane(result, expected, expected_text_hash)
        return {
            "kind": "spark",
            "lane": index,
            "tokens": int(result["tokens"]),
            "token_sha256": result["token_sha256"],
            "text_sha256": result["text_sha256"],
            "first_token_seconds": (
                request_started - origin["value"] + float(result["first_token_seconds"])
            ),
            "completion_seconds": request_started - origin["value"] + float(result["seconds"]),
        }

    def mac_lane() -> dict[str, Any]:
        barrier.wait(timeout=min(timeout, 30.0))
        return mac_run(
            runtime,
            tokenizer,
            prompt_ids,
            parallel=mac_parallel,
            max_new_tokens=max_new_tokens,
            mailbox_name=mailbox_name,
            timeout=timeout,
            model_id=model_id,
            model_revision=model_revision,
            local_reference=local_reference,
            timing_origin=origin["value"],
        )

    with ThreadPoolExecutor(max_workers=spark_parallel + 1) as pool:
        native_futures = [pool.submit(native_lane, index) for index in range(spark_parallel)]
        mac_future = pool.submit(mac_lane)
        native_lanes = [future.result() for future in native_futures]
        mac = mac_future.result()
    seconds = time.perf_counter() - origin["value"]
    mac_results = [mac["result"]] if mac_parallel == 1 else list(mac["batch"]["results"])
    if len(mac_results) != mac_parallel:
        raise HandoffError("MCDMA-fed Mac completed the wrong number of lanes")
    mac_lanes: list[dict[str, Any]] = []
    for index, item in enumerate(mac_results):
        tokens = list(item["tokens"])
        if tokens != expected or item.get("token_sha256") != expected_hash:
            raise HandoffError("MCDMA-fed Mac tokens differ from the frozen reference")
        text_hash = _text_digest(str(item["text"]))
        if text_hash != expected_text_hash:
            raise HandoffError("MCDMA-fed Mac bytes differ from the frozen reference")
        first = float(item["first_token_from_origin_seconds"])
        completion = float(item["completion_from_origin_seconds"])
        mac_lanes.append({
            "kind": "mac",
            "lane": index,
            "tokens": len(tokens),
            "token_sha256": expected_hash,
            "text_sha256": text_hash,
            "first_token_seconds": first,
            "completion_seconds": completion,
        })
    lanes = native_lanes + mac_lanes
    parallel = spark_parallel + mac_parallel
    if sum(int(item["tokens"]) for item in lanes) != parallel * max_new_tokens:
        raise HandoffError("candidate did not complete the frozen token count")
    result = _round_summary(
        arm="B",
        run=run,
        parallel=parallel,
        allocation={"spark": spark_parallel, "mac": mac_parallel},
        seconds=seconds,
        lanes=lanes,
        expected_hash=expected_hash,
        expected_text_hash=expected_text_hash,
        prompt_sha256=prompt_sha256,
        max_new_tokens=max_new_tokens,
    )
    result["mac"] = {
        "cache_wire_transfers": mac.get("cache_wire_transfers"),
        "cache_reuse_copies": mac.get("cache_reuse_copies"),
        "transfer_id": mac.get("transfer_id"),
        "prompt_sha256": mac.get("prompt_sha256"),
        "cache_bytes_total": mac.get("cache_bytes_total"),
        "cache_frame_count": mac.get("cache_frame_count"),
        "cache_frame_lengths": mac.get("cache_frame_lengths"),
        "transfer_seconds": mac.get("transfer_seconds"),
        "import_seconds": mac.get("import_seconds"),
        "reclaim_seconds": mac.get("reclaim_seconds"),
        "pipeline_seconds": mac.get("pipeline_seconds"),
    }
    return result


def _healthy(value: dict[str, Any]) -> bool:
    return value.get("status") in {"ok", "ready"} or value.get("ok") is True


def _serve_worker_impl(args: argparse.Namespace, protocol_stdout: Any) -> int:
    model_dir = _verify_snapshot_path(args.model, args.model_revision)
    import tensorfold

    runtime_evidence = _runtime_evidence(args, tensorfold)
    runtime, tokenizer = _load_runtime(model_dir)
    prompt_ids = render_user_prompt(tokenizer, args.prompt)
    local_reference = run_decode(
        runtime,
        tokenizer,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        stream_id="paired-capacity-reference",
    )
    prompt_sha256 = prompt_digest(prompt_ids)
    local_hash = prompt_digest(local_reference["tokens"])
    local_text_hash = _text_digest(str(local_reference["text"]))
    if prompt_sha256 != args.expected_prompt_sha256:
        raise HandoffError("Mac rendered prompt differs from the frozen cache reference")
    if local_hash != args.expected_token_sha256:
        raise HandoffError("Mac reference tokens differ from the frozen two-Spark reference")
    if local_text_hash != args.expected_text_sha256:
        raise HandoffError("Mac reference bytes differ from the frozen reference")
    if len(local_reference["tokens"]) != args.max_new_tokens:
        raise HandoffError("Mac reference stopped before the frozen token count")
    local_reference["prompt_sha256"] = prompt_sha256
    health_before = get_json(args.native_url.rstrip("/") + "/health", 10.0)
    if not _healthy(health_before):
        raise HandoffError("native TensorFold was not healthy before the paired benchmark")
    print(json.dumps({
        "event": "glm_paired_capacity_ready",
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": args.max_new_tokens,
        "token_sha256": local_hash,
        "text_sha256": local_text_hash,
        "prompt_sha256": prompt_sha256,
        "runtime": runtime_evidence,
        "worker_id": args.worker_id,
        "health_before": health_before,
    }, sort_keys=True), file=protocol_stdout, flush=True)

    for line in sys.stdin:
        command = json.loads(line)
        if command == {"command": "stop"}:
            health_after = get_json(args.native_url.rstrip("/") + "/health", 10.0)
            if not _healthy(health_after):
                raise HandoffError("native TensorFold was not healthy after the paired benchmark")
            print(json.dumps({
                "event": "glm_paired_capacity_stopped", "health_after": health_after
            }, sort_keys=True), file=protocol_stdout, flush=True)
            return 0
        if set(command) != {"arm", "run", "total_parallel", "mac_parallel"}:
            raise HandoffError("paired benchmark worker received a malformed command")
        arm = command["arm"]
        run = int(command["run"])
        total = int(command["total_parallel"])
        mac_parallel = int(command["mac_parallel"])
        if not 1 <= total <= 16 or not 1 <= mac_parallel <= min(8, total):
            raise HandoffError("paired benchmark worker received invalid parallelism")
        if arm == "A":
            result = run_native_round(
                prompt=args.prompt,
                native_url=args.native_url,
                max_new_tokens=args.max_new_tokens,
                timeout=args.timeout,
                local_reference=local_reference,
                run=run,
                parallel=total,
            )
        elif arm == "B":
            if total - mac_parallel > 8:
                raise HandoffError("candidate Spark parallelism exceeds eight")
            result = run_round(
                runtime,
                tokenizer,
                prompt_ids,
                prompt=args.prompt,
                native_url=args.native_url,
                mailbox_name=args.mailbox,
                max_new_tokens=args.max_new_tokens,
                timeout=args.timeout,
                model_id=args.model_id,
                model_revision=args.model_revision,
                local_reference=local_reference,
                run=run,
                spark_parallel=total - mac_parallel,
                mac_parallel=mac_parallel,
            )
        else:
            raise HandoffError("paired benchmark arm must be A or B")
        print(json.dumps(result, sort_keys=True), file=protocol_stdout, flush=True)
    raise HandoffError("paired benchmark worker input closed before the stop command")


def _serve_worker(args: argparse.Namespace) -> int:
    """Keep every TensorFold notice off the JSON-lines protocol stream."""

    protocol_stdout = sys.stdout
    with redirect_stdout(sys.stderr):
        return _serve_worker_impl(args, protocol_stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native-url", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mailbox", default="p06cfr")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--expected-prompt-sha256", required=True)
    parser.add_argument("--expected-token-sha256", required=True)
    parser.add_argument("--expected-text-sha256", required=True)
    parser.add_argument("--tensorfold-source", type=Path, required=True)
    parser.add_argument("--expected-tensorfold-tree-sha256", required=True)
    parser.add_argument("--expected-python-sha256", required=True)
    parser.add_argument("--expected-python-env", required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    for label in (
        "expected_prompt_sha256",
        "expected_token_sha256",
        "expected_text_sha256",
        "expected_tensorfold_tree_sha256",
        "expected_python_sha256",
    ):
        value = getattr(args, label)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            parser.error(f"{label.replace('_', '-')} must be a lowercase SHA-256")
    if not args.worker:
        parser.error("the fair capacity harness requires --worker")
    with _worker_pidfile(args.pid_file, args.worker_id):
        return _serve_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
