"""Mac-side GLM decode from a verified two-Spark prompt cache.

The Sparks compute the prompt once.  This client retrieves the versioned cache
over MCDMA, refuses incomplete or mismatched state, imports it into TensorFold's
MLX cache objects, and leaves only the final prompt token plus reply generation
for the Mac.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence

import numpy as np

from experiments.three_machine.cache_handoff import (
    ChunkAssembler,
    DEFAULT_FRAME_BYTES,
    HandoffError,
    HandoffManifest,
    MAX_FRAME_BYTES,
    MAX_HEADER_BYTES,
    Request,
    encode_request,
    prompt_digest,
    raise_if_error,
    unpack_ack,
    unpack_manifest,
)
from experiments.three_machine.cuda_prefill import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from experiments.three_machine.mailbox import ClientMailbox


@dataclass(frozen=True)
class DecodeJob:
    stream_id: str
    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    cache: list[Any]
    cached_tokens: int


def fetch_cache(
    mailbox: Any,
    tokens: Sequence[int],
    *,
    model_id: str,
    model_revision: str,
    timeout: float,
    chunk_bytes: int = DEFAULT_FRAME_BYTES,
) -> tuple[HandoffManifest, ChunkAssembler, tuple[int, ...]]:
    """Fetch, verify, and release one immutable cache transfer."""

    ids = tuple(int(token) for token in tokens)
    digest = prompt_digest(ids)
    raw = mailbox.call(encode_request(Request(
        operation="prepare",
        tokens=ids,
        prompt_sha256=digest,
        model_id=model_id,
        model_revision=model_revision,
    )), timeout)
    raise_if_error(raw)
    manifest = unpack_manifest(raw)

    def release() -> None:
        answer = mailbox.call(encode_request(Request(
            operation="release", transfer_id=manifest.transfer_id
        )), timeout)
        if unpack_ack(answer, "release") != manifest.transfer_id:
            raise HandoffError("cache release acknowledgement has the wrong transfer identity")

    try:
        manifest.require_identity(
            model_id=model_id,
            model_revision=model_revision,
            prompt_sha256=digest,
            cached_tokens=len(ids) - 1,
        )
        assembler = ChunkAssembler(manifest)
        framing_bytes = 4 + 4 + MAX_HEADER_BYTES
        reply_limit = int(getattr(mailbox, "max_reply", MAX_FRAME_BYTES + framing_bytes))
        effective_frame_bytes = min(
            int(chunk_bytes), MAX_FRAME_BYTES, reply_limit - framing_bytes
        )
        if effective_frame_bytes <= 0:
            raise HandoffError("MCDMA reply mailbox is too small for a cache frame")
        total = sum(spec.nbytes for spec in manifest.tensors)
        frame_lengths: list[int] = []
        offset = 0
        while offset < total:
            length = min(effective_frame_bytes, total - offset)
            answer = mailbox.call(encode_request(Request(
                operation="frame",
                transfer_id=manifest.transfer_id,
                offset=offset,
                length=length,
            )), timeout)
            raise_if_error(answer)
            assembler.accept_frame(answer)
            frame_lengths.append(length)
            offset += length
        assembler.verify_complete()
    except BaseException as exc:
        try:
            release()
        except BaseException as release_exc:
            raise HandoffError(
                f"cache transfer failed ({exc}); its release also failed ({release_exc})"
            ) from exc
        raise

    release()
    return manifest, assembler, tuple(frame_lengths)


def _numpy_tensor(assembler: ChunkAssembler, name: str) -> tuple[np.ndarray, bool]:
    spec = assembler.manifest.tensor(name)
    if spec.dtype == "bfloat16":
        expected = int(np.prod(spec.shape, dtype=np.int64)) * np.dtype(np.uint16).itemsize
        raw = assembler.buffer_for(name)
        if raw.nbytes != expected:
            raise HandoffError(f"bfloat16 tensor {name!r} byte count does not match its shape")
        return np.frombuffer(raw, dtype=np.uint16).reshape(spec.shape), True
    return assembler.array(name), False


def _mlx_tensor(assembler: ChunkAssembler, name: str, mx: Any) -> Any:
    value, bfloat16 = _numpy_tensor(assembler, name)
    result = mx.array(value)
    return result.view(mx.bfloat16) if bfloat16 else result


def _cache_tensor_names(cache: Sequence[Any]) -> set[str]:
    if len(cache) < 2:
        raise HandoffError("TensorFold returned an incomplete GLM cache")
    names: set[str] = set()
    for layer, target in enumerate(cache[:-1]):
        if hasattr(target, "conv") and hasattr(target, "ssm"):
            names.update((f"layers.{layer}.conv", f"layers.{layer}.ssm"))
        elif all(hasattr(target, field) for field in ("keys", "ik", "ig", "pool")):
            names.add(f"layers.{layer}.keys")
        else:
            raise HandoffError(f"GLM cache layer {layer} has an unsupported cache type")
    if not all(hasattr(cache[-1], field) for field in ("keys", "ik", "ig", "pool", "drafted")):
        raise HandoffError("TensorFold returned an unsupported MTP cache")
    names.add("mtp.keys")
    return names


def import_cache(
    runtime: Any,
    manifest: HandoffManifest,
    assembler: ChunkAssembler,
    *,
    mx_module: Any = None,
) -> list[Any]:
    """Build TensorFold MLX cache objects from a fully verified handoff."""

    if assembler.manifest != manifest:
        raise HandoffError("cache assembler and manifest identities differ")
    manifest.validate()
    if manifest.long_context:
        raise HandoffError("sparse long-context state is not supported by cache handoff v1")
    if mx_module is None:
        import mlx.core as mx_module
    mx = mx_module

    cache = runtime.make_cache()
    expected = _cache_tensor_names(cache)
    actual = {item.name for item in manifest.tensors}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise HandoffError(f"cache tensor set differs (missing={missing}, extra={extra})")

    cached = int(manifest.cached_tokens)
    index_width = int(getattr(runtime.args, "index_head_dim", 128))
    kpool = int(getattr(runtime.args, "index_kpool", 4))
    dense_limit = int(getattr(runtime.args, "index_topk", 2048))
    latent_width = int(getattr(runtime.args, "kv_lora_rank", 512))
    linear_heads = int(getattr(runtime.args, "linear_num_heads", 64))
    linear_dim = int(getattr(runtime.args, "linear_head_dim", 128))
    linear_conv = int(getattr(runtime.args, "linear_conv", 4))
    if cached >= dense_limit:
        raise HandoffError(
            f"cache has {cached} prefix tokens; handoff v1 must leave the Mac below "
            f"the {dense_limit}-token dense-attention limit"
        )
    if index_width <= 0 or kpool <= 0:
        raise HandoffError("GLM sparse-index geometry is invalid")

    evaluations: list[Any] = []
    for layer, target in enumerate(cache[:-1]):
        if hasattr(target, "conv") and hasattr(target, "ssm"):
            target.conv = _mlx_tensor(assembler, f"layers.{layer}.conv", mx)
            recurrent = _mlx_tensor(assembler, f"layers.{layer}.ssm", mx)
            expected_conv = (linear_conv - 1, 3 * linear_heads * linear_dim)
            expected_recurrent = (linear_heads, linear_dim, linear_dim)
            if tuple(target.conv.shape) != expected_conv or tuple(recurrent.shape) != expected_recurrent:
                raise HandoffError(
                    f"KDA layer {layer} geometry differs from the local GLM runtime"
                )
            # CUDA has no stream batch axis; MLX KDA state always carries one.
            target.ssm = recurrent[None]
            target.offset = cached
            evaluations.extend((target.conv, target.ssm))
            continue

        target.keys = _mlx_tensor(assembler, f"layers.{layer}.keys", mx)
        if tuple(target.keys.shape) != (cached, latent_width):
            raise HandoffError(f"attention layer {layer} key geometry differs from the local GLM runtime")
        dtype = target.keys.dtype
        target.ik = mx.zeros((cached, index_width), dtype=dtype)
        target.ig = mx.zeros((cached, index_width), dtype=dtype)
        target.pool = mx.zeros((-(-cached // kpool), index_width), dtype=dtype)
        target.offset = cached
        evaluations.extend((target.keys, target.ik, target.ig, target.pool))

    mtp = cache[-1]
    mtp.keys = _mlx_tensor(assembler, "mtp.keys", mx)
    if tuple(mtp.keys.shape) != (cached, latent_width):
        raise HandoffError("MTP key geometry differs from the local GLM runtime")
    dtype = mtp.keys.dtype
    mtp.ik = mx.zeros((cached, index_width), dtype=dtype)
    mtp.ig = mx.zeros((cached, index_width), dtype=dtype)
    mtp.pool = mx.zeros((-(-cached // kpool), index_width), dtype=dtype)
    mtp.offset = cached
    mtp.drafted = 0
    evaluations.extend((mtp.keys, mtp.ik, mtp.ig, mtp.pool))
    mx.eval(*evaluations)
    return cache


def _eos_ids(tokenizer: Any) -> frozenset[int]:
    value = getattr(tokenizer, "eos_token_ids", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    return frozenset(int(item) for item in values if item is not None)


def render_user_prompt(tokenizer: Any, prompt: str) -> list[int]:
    """Render the same thinking-off chat prompt as TensorFold's HTTP server."""

    from tensorfold.server.text import render_prompt_ids

    return render_prompt_ids(
        tokenizer,
        [{"role": "user", "content": prompt}],
        enable_thinking=False,
        add_generation_prompt=True,
    )


def _load_runtime(model_dir: Path) -> tuple[Any, Any]:
    """Load through TensorFold's public GLM entry point.

    The public loader applies the Metal working-set limit that keeps the large
    GLM weights resident between rounds. Calling ``runtime.load`` directly
    skips that protection and can make an otherwise identical round page for
    minutes.
    """

    from tensorfold.families.glm5_next import load

    return load(Path(model_dir), mtp_drafts=3)


def _new_engine(runtime: Any, *, imported: bool) -> Any:
    from tensorfold.engine.lane_engine import LaneEngine
    from tensorfold.families.glm5_next import engine_settings

    engine = LaneEngine(runtime, **engine_settings(runtime))
    if imported:
        # A cross-host prefix need not end on the local prompt planner's chunk
        # boundary.  TensorFold's cache contract still validates its position.
        engine.prefill_plan = None
    return engine


def _release_engine(engine: Any) -> None:
    """Release the completed round and detach every stream from its engine."""

    try:
        engine.release_rounds()
    finally:
        engine.reset()


def _reclaim_mlx_memory() -> None:
    """Return freed round buffers to MLX before the next measured round."""

    import mlx.core as mx

    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def run_decode(
    runtime: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    cache: list[Any] | None = None,
    cached_tokens: int = 0,
    stream_id: str = "request",
    timing_origin: float | None = None,
) -> dict[str, Any]:
    from tensorfold.engine.lane_engine import LaneStream

    started = time.perf_counter()
    origin = started if timing_origin is None else float(timing_origin)
    if origin > started:
        raise HandoffError("decode timing origin is in the future")
    dense_limit = int(getattr(runtime.args, "index_topk", 2048))
    if cache is not None and len(prompt_ids) + int(max_new_tokens) > dense_limit:
        raise HandoffError(
            f"prompt plus reply can reach {len(prompt_ids) + int(max_new_tokens)} tokens; "
            f"cache handoff v1 is limited to {dense_limit}"
        )
    engine = _new_engine(runtime, imported=cache is not None)
    try:
        stream = LaneStream(
            stream_id=stream_id,
            prompt_ids=[int(token) for token in prompt_ids],
            max_new_tokens=int(max_new_tokens),
            eos_ids=_eos_ids(tokenizer),
            drafts=True,
        )
        engine.add_stream(stream, cache=cache, cached_tokens=int(cached_tokens))
        first_at: float | None = None
        while engine.active_count:
            engine.step()
            if first_at is None and stream.emitted:
                first_at = time.perf_counter()
        finished = time.perf_counter()
        tokens = [int(token) for token in stream.emitted]
        first_at = finished if first_at is None else first_at
        total_seconds = finished - started
        generation_seconds = finished - first_at
        subsequent_tokens = max(0, len(tokens) - 1)
        subsequent_rate = subsequent_tokens / generation_seconds if generation_seconds > 0 else 0.0
        output = {
            "tokens": tokens,
            "token_sha256": prompt_digest(tokens),
            "text": tokenizer.decode(tokens, skip_special_tokens=False),
            "first_token_seconds": first_at - started,
            "first_token_from_origin_seconds": first_at - origin,
            "completion_from_origin_seconds": finished - origin,
            "generation_seconds": generation_seconds,
            "total_seconds": total_seconds,
            "subsequent_tokens_per_second": subsequent_rate,
            "decode_tokens_per_second": subsequent_rate,
            "end_to_end_tokens_per_second": len(tokens) / total_seconds if total_seconds > 0 else 0.0,
            "finish_reason": stream.finish_reason,
        }
    finally:
        try:
            _release_engine(engine)
        finally:
            del engine
            _reclaim_mlx_memory()
    return output


def run_decode_batch(
    runtime: Any,
    tokenizer: Any,
    jobs: Sequence[DecodeJob],
    *,
    timing_origin: float | None = None,
) -> dict[str, Any]:
    """Decode several imported Spark prefixes in one MLX batch."""

    from tensorfold.engine.lane_engine import LaneStream

    started = time.perf_counter()
    origin = started if timing_origin is None else float(timing_origin)
    if origin > started:
        raise HandoffError("decode timing origin is in the future")
    if not jobs:
        raise ValueError("at least one decode job is required")
    if len({job.stream_id for job in jobs}) != len(jobs):
        raise ValueError("decode job stream IDs must be unique")
    dense_limit = int(getattr(runtime.args, "index_topk", 2048))
    streams = []
    for job in jobs:
        if job.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if len(job.prompt_ids) + job.max_new_tokens > dense_limit:
            raise HandoffError(
                f"stream {job.stream_id!r} can reach "
                f"{len(job.prompt_ids) + job.max_new_tokens} tokens; "
                f"cache handoff v1 is limited to {dense_limit}"
            )
        streams.append(LaneStream(
            stream_id=job.stream_id,
            prompt_ids=list(job.prompt_ids),
            max_new_tokens=job.max_new_tokens,
            eos_ids=_eos_ids(tokenizer),
            drafts=True,
        ))

    engine = _new_engine(runtime, imported=True)
    try:
        for stream, job in zip(streams, jobs, strict=True):
            engine.add_stream(stream, cache=job.cache, cached_tokens=job.cached_tokens)
        ready_at = time.perf_counter()
        first_at: dict[str, float] = {}
        finished_at: dict[str, float] = {}
        while engine.active_count:
            engine.step()
            observed_at = time.perf_counter()
            for stream in streams:
                if stream.emitted and stream.stream_id not in first_at:
                    first_at[stream.stream_id] = observed_at
                if stream.finished and stream.stream_id not in finished_at:
                    finished_at[stream.stream_id] = observed_at
        finished = time.perf_counter()

        results = []
        for stream in streams:
            tokens = [int(token) for token in stream.emitted]
            first = first_at.get(stream.stream_id, finished)
            stream_finished = finished_at.get(stream.stream_id, finished)
            generation_seconds = stream_finished - first
            subsequent_tokens = max(0, len(tokens) - 1)
            results.append({
                "stream_id": stream.stream_id,
                "tokens": tokens,
                "token_sha256": prompt_digest(tokens),
                "text": tokenizer.decode(tokens, skip_special_tokens=False),
                "first_token_seconds": first - started,
                "first_token_from_origin_seconds": first - origin,
                "completion_from_origin_seconds": stream_finished - origin,
                "generation_seconds": generation_seconds,
                "subsequent_tokens_per_second": (
                    subsequent_tokens / generation_seconds if generation_seconds > 0 else 0.0
                ),
                "finish_reason": stream.finish_reason,
            })
        first_batch_token_at = min(first_at.values(), default=finished)
        decode_seconds = finished - first_batch_token_at
        total_seconds = finished - started
        token_count = sum(len(result["tokens"]) for result in results)
        decode_token_count = sum(max(0, len(result["tokens"]) - 1) for result in results)
        output = {
            "parallel": len(results),
            "tokens": token_count,
            "decode_tokens": decode_token_count,
            "batch_ready_seconds": ready_at - started,
            "first_token_seconds": first_batch_token_at - started,
            "batch_decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
            "aggregate_decode_tokens_per_second": (
                decode_token_count / decode_seconds if decode_seconds > 0 else 0.0
            ),
            "end_to_end_tokens_per_second": token_count / total_seconds if total_seconds > 0 else 0.0,
            "results": results,
        }
    finally:
        try:
            _release_engine(engine)
        finally:
            del engine
            _reclaim_mlx_memory()
    return output


def _verify_snapshot_path(model_dir: Path, revision: str) -> Path:
    path = model_dir.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {path}")
    if len(path.name) == 40 and all(char in "0123456789abcdef" for char in path.name.lower()):
        if path.name.lower() != revision.lower():
            raise HandoffError("local model snapshot directory differs from the requested revision")
    return path


def run_handoff_round(
    runtime: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    *,
    parallel: int,
    max_new_tokens: int,
    mailbox_name: str,
    timeout: float,
    model_id: str,
    model_revision: str,
    local_reference: dict[str, Any] | None,
    chunk_bytes: int = DEFAULT_FRAME_BYTES,
    timing_origin: float | None = None,
) -> dict[str, Any]:
    """Run one measured Spark-prefill, MCDMA-transfer, Mac-decode round."""

    entered = time.perf_counter()
    origin = entered if timing_origin is None else float(timing_origin)
    if origin > entered:
        raise HandoffError("handoff timing origin is in the future")
    if not 1 <= parallel <= 8:
        raise ValueError("parallel must be between 1 and 8")
    transfer_started = time.perf_counter()
    with ClientMailbox(mailbox_name, ready_timeout_s=min(timeout, 30.0)) as mailbox:
        manifest, assembler, frame_lengths = fetch_cache(
            mailbox,
            prompt_ids,
            model_id=model_id,
            model_revision=model_revision,
            timeout=timeout,
            chunk_bytes=chunk_bytes,
        )
    transfer_seconds = time.perf_counter() - transfer_started
    import_started = time.perf_counter()
    base_cache = import_cache(runtime, manifest, assembler)
    if parallel == 1:
        caches = [base_cache]
    else:
        from tensorfold.engine.lane_engine import LaneEngine

        # Every request has the same immutable prompt prefix. TensorFold's
        # cache copier shares whole MLX arrays and detaches only views, so one
        # verified MCDMA transfer safely seeds all concurrent lanes.
        caches = [base_cache] + [
            LaneEngine.copy_single_cache(base_cache) for _ in range(parallel - 1)
        ]
    import_seconds = time.perf_counter() - import_started
    if parallel == 1:
        result = run_decode(
            runtime, tokenizer, prompt_ids,
            max_new_tokens=max_new_tokens,
            cache=caches[0],
            cached_tokens=manifest.cached_tokens,
            stream_id="mcdma-handoff",
            timing_origin=origin,
        )
        batch = None
        generated = [result]
        decode_seconds = float(result["total_seconds"])
    else:
        batch = run_decode_batch(runtime, tokenizer, [
            DecodeJob(
                stream_id=f"mcdma-handoff-{index}",
                prompt_ids=tuple(int(token) for token in prompt_ids),
                max_new_tokens=max_new_tokens,
                cache=cache,
                cached_tokens=manifest.cached_tokens,
            )
            for index, cache in enumerate(caches)
        ], timing_origin=origin)
        result = batch["results"][0]
        generated = batch["results"]
        decode_seconds = float(batch["total_seconds"])
    completion_tokens = sum(len(item["tokens"]) for item in generated)
    cache_bytes = sum(item.nbytes for item in manifest.tensors)
    exact_token_match = None
    if local_reference is not None:
        exact_token_match = all(
            local_reference["tokens"] == item["tokens"] for item in generated
        )

    reclaim_started = time.perf_counter()
    del caches
    del base_cache
    del assembler
    _reclaim_mlx_memory()
    reclaim_seconds = time.perf_counter() - reclaim_started
    pipeline_seconds = time.perf_counter() - origin
    output = {
        "event": "glm_mcdma_decode",
        "model_id": model_id,
        "model_revision": model_revision,
        "prompt_sha256": manifest.prompt_sha256,
        "prompt_tokens": len(prompt_ids),
        "parallel": parallel,
        "cached_tokens": manifest.cached_tokens,
        "transfer_id": manifest.transfer_id,
        "cache_bytes_per_request": cache_bytes,
        "cache_bytes_total": cache_bytes,
        "cache_frame_count": len(frame_lengths),
        "cache_frame_lengths": list(frame_lengths),
        "cache_wire_transfers": 1,
        "cache_reuse_copies": parallel - 1,
        "chunk_bytes": int(chunk_bytes),
        "transfer_seconds": transfer_seconds,
        "import_seconds": import_seconds,
        "decode_seconds": decode_seconds,
        "reclaim_seconds": reclaim_seconds,
        "pipeline_seconds": pipeline_seconds,
        "completion_tokens": completion_tokens,
        "aggregate_pipeline_tokens_per_second": (
            completion_tokens / pipeline_seconds if pipeline_seconds > 0 else 0.0
        ),
        "result": result,
    }
    if batch is not None:
        output["batch"] = batch
    if local_reference is not None:
        output["local_reference"] = local_reference
        output["exact_token_match"] = exact_token_match
        if not output["exact_token_match"]:
            raise HandoffError("Spark-prefilled and Mac-prefilled replies differ")
    return output


def _run_workload(args: argparse.Namespace, model_dir: Path) -> dict[str, Any]:
    runtime, tokenizer = _load_runtime(model_dir)
    prompt_ids = render_user_prompt(tokenizer, args.prompt)
    if len(prompt_ids) < 2:
        raise HandoffError("the rendered prompt must contain at least two tokens")

    local = None
    if args.compare_local:
        local = run_decode(
            runtime, tokenizer, prompt_ids,
            max_new_tokens=args.max_new_tokens,
            stream_id="local-reference",
        )

    rounds = [
        run_handoff_round(
            runtime,
            tokenizer,
            prompt_ids,
            parallel=args.parallel,
            max_new_tokens=args.max_new_tokens,
            mailbox_name=args.mailbox,
            timeout=args.timeout,
            model_id=args.model_id,
            model_revision=args.model_revision,
            local_reference=local,
            chunk_bytes=args.chunk_mib * 1024 * 1024,
        )
        for _ in range(args.warmups + args.repeats)
    ]
    if args.warmups == 0 and args.repeats == 1:
        return rounds[0]
    warmups = rounds[:args.warmups]
    measured = rounds[args.warmups:]
    rates = [float(item["aggregate_pipeline_tokens_per_second"]) for item in measured]
    return {
        "event": "glm_mcdma_benchmark",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "prompt_tokens": len(prompt_ids),
        "parallel": args.parallel,
        "max_new_tokens": args.max_new_tokens,
        "chunk_mib": args.chunk_mib,
        "warmup_count": args.warmups,
        "repeat_count": args.repeats,
        "aggregate_pipeline_tokens_per_second": {
            "median": statistics.median(rates),
            "min": min(rates),
            "max": max(rates),
        },
        "local_reference": local,
        "warmups": warmups,
        "runs": measured,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--mailbox", default="p06cfr")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--compare-local", action="store_true")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--chunk-mib", type=int, default=DEFAULT_FRAME_BYTES // (1024 * 1024))
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if not 1 <= args.parallel <= 8:
        parser.error("--parallel must be between 1 and 8")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be nonnegative and --repeats must be positive")
    if not 1 <= args.chunk_mib <= MAX_FRAME_BYTES // (1024 * 1024):
        parser.error(f"--chunk-mib must be between 1 and {MAX_FRAME_BYTES // (1024 * 1024)}")
    model_dir = _verify_snapshot_path(Path(args.model), args.model_revision)

    # TensorFold reports model tuning information to stdout. Keep stdout a
    # single JSON record for proof and benchmark consumers while preserving
    # those notices on stderr.
    with redirect_stdout(sys.stderr):
        output = _run_workload(args, model_dir)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
