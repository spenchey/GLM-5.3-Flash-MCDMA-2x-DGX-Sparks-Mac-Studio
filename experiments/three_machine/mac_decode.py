"""Mac-side GLM decode from a verified two-Spark prompt cache.

The Sparks compute the prompt once.  This client retrieves the versioned cache
over MCDMA, refuses incomplete or mismatched state, imports it into TensorFold's
MLX cache objects, and leaves reply generation to the Mac.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass, field
import gc
import hashlib
import json
import math
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
from experiments.three_machine.kda_intermediates import (
    KDA_INTERMEDIATE_STAGES,
    kda_intermediate_names,
)
from experiments.three_machine.mailbox import ClientMailbox
from experiments.three_machine.progressive_handoff import (
    DEFAULT_PROGRESSIVE_FRAME_BYTES,
    ProgressiveAssembler,
    ProgressiveManifest,
    ProgressiveRequest,
    encode_progressive_request,
    new_job_id,
    progressive_frame_ranges,
    unpack_plan,
    unpack_progressive_ack,
    unpack_seal,
)


@dataclass(frozen=True)
class DecodeJob:
    stream_id: str
    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    cache: list[Any]
    cached_tokens: int


@dataclass
class ReusablePrefix:
    """One verified Spark prefix retained on the Mac between requests."""

    manifest: HandoffManifest | ProgressiveManifest
    cache: list[Any]
    frame_lengths: tuple[int, ...]
    transfer_seconds: float
    import_seconds: float
    final_hidden: Any = None
    handoff_telemetry: dict[str, Any] = field(default_factory=dict)


def fetch_cache(
    mailbox: Any,
    tokens: Sequence[int],
    *,
    model_id: str,
    model_revision: str,
    timeout: float,
    chunk_bytes: int = DEFAULT_FRAME_BYTES,
    telemetry: dict[str, Any] | None = None,
) -> tuple[HandoffManifest, ChunkAssembler, tuple[int, ...]]:
    """Fetch, verify, and release one immutable cache transfer."""

    fetch_started = time.perf_counter()
    stats = telemetry if telemetry is not None else {}
    ids = tuple(int(token) for token in tokens)
    digest = prompt_digest(ids)
    prepare_started = time.perf_counter()
    raw = mailbox.call(encode_request(Request(
        operation="prepare",
        tokens=ids,
        prompt_sha256=digest,
        model_id=model_id,
        model_revision=model_revision,
    )), timeout)
    stats["prepare_call_seconds"] = time.perf_counter() - prepare_started
    manifest_started = time.perf_counter()
    raise_if_error(raw)
    manifest = unpack_manifest(raw)
    stats["manifest_decode_seconds"] = time.perf_counter() - manifest_started

    def release() -> None:
        release_started = time.perf_counter()
        answer = mailbox.call(encode_request(Request(
            operation="release", transfer_id=manifest.transfer_id
        )), timeout)
        stats["release_call_seconds"] = (
            float(stats.get("release_call_seconds", 0.0))
            + time.perf_counter() - release_started
        )
        if unpack_ack(answer, "release") != manifest.transfer_id:
            raise HandoffError("cache release acknowledgement has the wrong transfer identity")

    try:
        manifest.require_identity(
            model_id=model_id,
            model_revision=model_revision,
            prompt_sha256=digest,
            cached_tokens=len(ids),
            mtp_cached_tokens=len(ids) - 1,
        )
        if manifest.spark_first_token is None:
            raise HandoffError("Spark handoff did not bind its reference first token")
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
        frame_call_seconds = 0.0
        frame_assemble_seconds = 0.0
        frame_reply_bytes = 0
        offset = 0
        while offset < total:
            length = min(effective_frame_bytes, total - offset)
            request = encode_request(Request(
                operation="frame",
                transfer_id=manifest.transfer_id,
                offset=offset,
                length=length,
            ))
            frame_call_started = time.perf_counter()
            consume = getattr(mailbox, "call_consume", None)
            if callable(consume):
                assembled = 0.0

                def accept(answer: memoryview) -> int:
                    nonlocal assembled
                    assemble_started = time.perf_counter()
                    raise_if_error(answer)
                    assembler.accept_frame(answer)
                    assembled += time.perf_counter() - assemble_started
                    return len(answer)

                reply_bytes = consume(request, accept, timeout)
                call_elapsed = time.perf_counter() - frame_call_started
                frame_call_seconds += max(0.0, call_elapsed - assembled)
                frame_assemble_seconds += assembled
            else:
                answer = mailbox.call(request, timeout)
                frame_call_seconds += time.perf_counter() - frame_call_started
                reply_bytes = len(answer)
                assemble_started = time.perf_counter()
                raise_if_error(answer)
                assembler.accept_frame(answer)
                frame_assemble_seconds += time.perf_counter() - assemble_started
            frame_reply_bytes += reply_bytes
            frame_lengths.append(length)
            offset += length
        verify_started = time.perf_counter()
        assembler.verify_complete()
        stats.update({
            "archive_bytes": total,
            "effective_frame_bytes": effective_frame_bytes,
            "frame_count": len(frame_lengths),
            "frame_payload_bytes": sum(frame_lengths),
            "frame_reply_bytes": frame_reply_bytes,
            "frame_call_seconds": frame_call_seconds,
            "frame_assemble_seconds": frame_assemble_seconds,
            "verify_seconds": time.perf_counter() - verify_started,
        })
    except BaseException as exc:
        try:
            release()
        except BaseException as release_exc:
            raise HandoffError(
                f"cache transfer failed ({exc}); its release also failed ({release_exc})"
            ) from exc
        raise

    release()
    stats["total_seconds"] = time.perf_counter() - fetch_started
    return manifest, assembler, tuple(frame_lengths)


def fetch_progressive_cache(
    mailbox: Any,
    tokens: Sequence[int],
    *,
    model_id: str,
    model_revision: str,
    timeout: float,
    chunk_bytes: int = DEFAULT_PROGRESSIVE_FRAME_BYTES,
    telemetry: dict[str, Any] | None = None,
) -> tuple[ProgressiveManifest, ProgressiveAssembler, tuple[int, ...]]:
    """Receive final-chunk cache state while the Sparks continue prefilling."""

    fetch_started = time.perf_counter()
    stats = telemetry if telemetry is not None else {}
    ids = tuple(int(token) for token in tokens)
    digest = prompt_digest(ids)
    generation = int(getattr(mailbox, "generation", 0))
    if generation <= 0:
        raise HandoffError("progressive handoff needs a live MCDMA generation")
    job_id = new_job_id()
    prepare = ProgressiveRequest(
        operation="prepare",
        generation=generation,
        job_id=job_id,
        tokens=ids,
        prompt_sha256=digest,
        model_id=model_id,
        model_revision=model_revision,
    )
    prepare_started = time.perf_counter()
    raw = mailbox.call(encode_progressive_request(prepare), timeout)
    stats["prepare_call_seconds"] = time.perf_counter() - prepare_started
    raise_if_error(raw)
    plan_started = time.perf_counter()
    plan = unpack_plan(raw)
    stats["plan_decode_seconds"] = time.perf_counter() - plan_started
    plan.require_identity(
        model_id=model_id,
        model_revision=model_revision,
        prompt_sha256=digest,
        cached_tokens=len(ids),
        mtp_cached_tokens=len(ids) - 1,
        generation=generation,
    )
    if plan.job_id != job_id:
        raise HandoffError("progressive plan belongs to another preparation attempt")
    assembler = ProgressiveAssembler(plan)
    seal = None

    def release() -> None:
        release_request = ProgressiveRequest(
            operation="release",
            generation=generation,
            job_id=job_id,
            plan_id=plan.plan_id,
            transfer_id=None if seal is None else seal.transfer_id,
        )
        release_started = time.perf_counter()
        answer = mailbox.call(encode_progressive_request(release_request), timeout)
        stats["release_call_seconds"] = (
            float(stats.get("release_call_seconds", 0.0))
            + time.perf_counter() - release_started
        )
        acknowledged = unpack_progressive_ack(answer, release_request)
        if acknowledged != release_request.transfer_id:
            raise HandoffError("progressive release acknowledgement has the wrong transfer identity")

    try:
        framing_bytes = 4 + 4 + MAX_HEADER_BYTES
        reply_limit = int(getattr(mailbox, "max_reply", MAX_FRAME_BYTES + framing_bytes))
        effective_frame_bytes = min(
            int(chunk_bytes),
            MAX_FRAME_BYTES,
            reply_limit - framing_bytes,
        )
        if effective_frame_bytes <= 0:
            raise HandoffError("MCDMA reply mailbox is too small for a progressive cache frame")
        frame_lengths: list[int] = []
        frame_call_seconds = 0.0
        frame_assemble_seconds = 0.0
        frame_reply_bytes = 0
        frame_samples: list[dict[str, int | float]] = []
        for offset, length in progressive_frame_ranges(plan, effective_frame_bytes):
            request = encode_progressive_request(ProgressiveRequest(
                operation="frame",
                generation=generation,
                job_id=job_id,
                plan_id=plan.plan_id,
                offset=offset,
                length=length,
            ))
            frame_call_started = time.perf_counter()
            consume = getattr(mailbox, "call_consume", None)
            if callable(consume):
                assembled = 0.0

                def accept(answer: memoryview) -> int:
                    nonlocal assembled
                    assemble_started = time.perf_counter()
                    raise_if_error(answer)
                    assembler.accept_frame(answer)
                    assembled += time.perf_counter() - assemble_started
                    return len(answer)

                reply_bytes = consume(request, accept, timeout)
                call_elapsed = time.perf_counter() - frame_call_started
                call_seconds = max(0.0, call_elapsed - assembled)
                assemble_seconds = assembled
            else:
                answer = mailbox.call(request, timeout)
                call_seconds = time.perf_counter() - frame_call_started
                reply_bytes = len(answer)
                assemble_started = time.perf_counter()
                raise_if_error(answer)
                assembler.accept_frame(answer)
                assemble_seconds = time.perf_counter() - assemble_started
            frame_call_seconds += call_seconds
            frame_assemble_seconds += assemble_seconds
            frame_reply_bytes += int(reply_bytes)
            frame_lengths.append(length)
            frame_samples.append({
                "offset": offset,
                "length": length,
                "reply_bytes": int(reply_bytes),
                "call_seconds": call_seconds,
                "assemble_seconds": assemble_seconds,
            })

        seal_request = ProgressiveRequest(
            operation="seal",
            generation=generation,
            job_id=job_id,
            plan_id=plan.plan_id,
        )
        seal_started = time.perf_counter()
        raw_seal = mailbox.call(encode_progressive_request(seal_request), timeout)
        stats["seal_call_seconds"] = time.perf_counter() - seal_started
        raise_if_error(raw_seal)
        verify_started = time.perf_counter()
        seal = unpack_seal(raw_seal)
        assembler.accept_seal(seal)
        manifest = assembler.manifest
        stats.update({
            "archive_bytes": plan.total_nbytes,
            "effective_frame_bytes": effective_frame_bytes,
            "frame_count": len(frame_lengths),
            "frame_payload_bytes": sum(frame_lengths),
            "frame_reply_bytes": frame_reply_bytes,
            "frames": frame_samples,
            "frame_strategy": "execution_order_tensor_groups",
            "frame_call_seconds": frame_call_seconds,
            "frame_assemble_seconds": frame_assemble_seconds,
            "verify_seconds": time.perf_counter() - verify_started,
            "protocol": plan.protocol,
            "job_id": job_id,
            "plan_id": plan.plan_id,
            "transfer_id": manifest.transfer_id,
        })
    except BaseException as exc:
        try:
            release()
        except BaseException as release_exc:
            raise HandoffError(
                f"progressive cache transfer failed ({exc}); its release also failed ({release_exc})"
            ) from exc
        raise

    release()
    stats["total_seconds"] = time.perf_counter() - fetch_started
    return manifest, assembler, tuple(frame_lengths)


def _numpy_tensor(
    assembler: ChunkAssembler | ProgressiveAssembler, name: str
) -> tuple[np.ndarray, bool]:
    spec = assembler.manifest.tensor(name)
    if spec.dtype == "bfloat16":
        expected = int(np.prod(spec.shape, dtype=np.int64)) * np.dtype(np.uint16).itemsize
        raw = assembler.buffer_for(name)
        if raw.nbytes != expected:
            raise HandoffError(f"bfloat16 tensor {name!r} byte count does not match its shape")
        return np.frombuffer(raw, dtype=np.uint16).reshape(spec.shape), True
    return assembler.array(name), False


def _mlx_tensor(
    assembler: ChunkAssembler | ProgressiveAssembler, name: str, mx: Any
) -> Any:
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
    if all(hasattr(cache[-1], field) for field in ("keys", "ik", "ig", "pool", "drafted")):
        names.add("mtp.keys")
    elif not (hasattr(cache[-1], "drafter") and callable(getattr(cache[-1], "get", None))):
        raise HandoffError("TensorFold returned an unsupported draft cache")
    return names


def _cache_state_tensors(cache: Sequence[Any]) -> dict[str, Any]:
    """Name the cache arrays that have a cross-host handoff representation."""

    if len(cache) < 2:
        raise HandoffError("TensorFold returned an incomplete GLM cache")
    tensors: dict[str, Any] = {}
    for layer, target in enumerate(cache[:-1]):
        if hasattr(target, "conv") and hasattr(target, "ssm"):
            tensors[f"layers.{layer}.conv"] = target.conv
            tensors[f"layers.{layer}.ssm"] = target.ssm
        elif all(hasattr(target, field) for field in ("keys", "ik", "ig", "pool")):
            tensors[f"layers.{layer}.keys"] = target.keys
        else:
            raise HandoffError(f"GLM cache layer {layer} has an unsupported cache type")
    draft = cache[-1]
    if all(hasattr(draft, field) for field in ("keys", "ik", "ig", "pool", "drafted")):
        tensors["mtp.keys"] = draft.keys
    elif not (hasattr(draft, "drafter") and callable(getattr(draft, "get", None))):
        raise HandoffError("TensorFold returned an unsupported draft cache")
    missing = sorted(name for name, value in tensors.items() if value is None)
    if missing:
        raise HandoffError(f"GLM cache tensors are uninitialized: {missing}")
    return tensors


def _float32_numpy(value: Any, *, mx_module: Any = None) -> np.ndarray:
    """Materialize one MLX or NumPy tensor as finite-comparable float32 values."""

    converted = value
    if mx_module is not None and hasattr(value, "astype"):
        converted = value.astype(mx_module.float32)
        mx_module.eval(converted)
    return np.asarray(converted, dtype=np.float32)


def compare_tensor_values(
    imported: Any,
    local: Any,
    *,
    mx_module: Any = None,
) -> dict[str, Any]:
    """Return exact and numerical distance for two semantically equal tensors."""

    imported_shape = tuple(int(item) for item in imported.shape)
    local_shape = tuple(int(item) for item in local.shape)
    result: dict[str, Any] = {
        "imported_shape": list(imported_shape),
        "local_shape": list(local_shape),
        "imported_dtype": str(imported.dtype),
        "local_dtype": str(local.dtype),
        "shape_match": imported_shape == local_shape,
    }
    if imported_shape != local_shape:
        result.update({
            "element_count": 0,
            "exact_elements": 0,
            "exact_fraction": 0.0,
            "max_abs": None,
            "mean_abs": None,
            "relative_l2": None,
            "cosine_similarity": None,
        })
        return result

    imported_values = np.ascontiguousarray(
        _float32_numpy(imported, mx_module=mx_module)
    )
    local_values = np.ascontiguousarray(
        _float32_numpy(local, mx_module=mx_module)
    )
    imported_flat = imported_values.reshape(-1)
    local_flat = local_values.reshape(-1)
    count = int(imported_flat.size)
    exact = 0
    max_abs = 0.0
    absolute_sum = 0.0
    imported_squared = 0.0
    local_squared = 0.0
    difference_squared = 0.0
    dot = 0.0
    # Cache arrays can hold tens of millions of values.  Accumulate in bounded
    # float64 chunks so the diagnostic is accurate without duplicating the
    # entire cache in double precision.
    for start in range(0, count, 1_048_576):
        end = min(start + 1_048_576, count)
        imported_chunk = imported_flat[start:end].astype(np.float64)
        local_chunk = local_flat[start:end].astype(np.float64)
        if not (
            np.all(np.isfinite(imported_chunk))
            and np.all(np.isfinite(local_chunk))
        ):
            raise HandoffError("cache comparison contains non-finite values")
        difference = imported_chunk - local_chunk
        absolute = np.abs(difference)
        exact += int(np.count_nonzero(imported_chunk == local_chunk))
        if absolute.size:
            max_abs = max(max_abs, float(np.max(absolute)))
        absolute_sum += float(np.sum(absolute, dtype=np.float64))
        imported_squared += float(np.dot(imported_chunk, imported_chunk))
        local_squared += float(np.dot(local_chunk, local_chunk))
        difference_squared += float(np.dot(difference, difference))
        dot += float(np.dot(imported_chunk, local_chunk))
    imported_norm = math.sqrt(imported_squared)
    local_norm = math.sqrt(local_squared)
    difference_norm = math.sqrt(difference_squared)
    denominator = imported_norm * local_norm
    cosine = (
        max(-1.0, min(1.0, dot / denominator))
        if denominator > 0.0 else (1.0 if difference_norm == 0.0 else None)
    )
    result.update({
        "element_count": count,
        "exact_elements": exact,
        "exact_fraction": exact / count if count else 1.0,
        "max_abs": max_abs,
        "mean_abs": absolute_sum / count if count else 0.0,
        "relative_l2": difference_norm / local_norm if local_norm > 0.0 else None,
        "cosine_similarity": cosine,
        "imported_float32_sha256": hashlib.sha256(
            imported_values.view(np.uint8)
        ).hexdigest(),
        "local_float32_sha256": hashlib.sha256(
            local_values.view(np.uint8)
        ).hexdigest(),
    })
    return result


def _compare_diagnostic_tensor_values(
    imported: Any,
    local: Any,
    *,
    mx_module: Any = None,
) -> dict[str, Any]:
    """Compare one diagnostic tensor without hiding non-finite evidence.

    Production cache comparisons remain fail-closed in
    :func:`compare_tensor_values`.  A traced kernel intermediate is evidence,
    though, so record which side contains NaN or infinity and continue through
    the remaining stages instead of losing the location of the first failure.
    """

    try:
        result = compare_tensor_values(imported, local, mx_module=mx_module)
        mismatch_summary: dict[str, Any] = {
            "count": None,
            "first_flat_index": None,
            "last_flat_index": None,
            "first": [],
            "axes": [],
        }
        if result["shape_match"]:
            imported_values = np.ascontiguousarray(
                _float32_numpy(imported, mx_module=mx_module)
            )
            local_values = np.ascontiguousarray(
                _float32_numpy(local, mx_module=mx_module)
            )
            mismatch_flat = np.flatnonzero(
                imported_values.reshape(-1) != local_values.reshape(-1)
            )
            mismatch_count = int(mismatch_flat.size)
            mismatch_summary.update({
                "count": mismatch_count,
                "first_flat_index": (
                    int(mismatch_flat[0]) if mismatch_count else None
                ),
                "last_flat_index": (
                    int(mismatch_flat[-1]) if mismatch_count else None
                ),
            })
        else:
            mismatch_count = 0
        if result["shape_match"] and mismatch_count:
            sample_flat = mismatch_flat[:16]
            sample_coordinates = np.stack(
                np.unravel_index(sample_flat, imported_values.shape), axis=1
            )
            imported_flat = imported_values.reshape(-1)
            local_flat = local_values.reshape(-1)
            mismatch_summary["first"] = [
                {
                    "index": [int(value) for value in coordinate],
                    "imported": float(imported_flat[flat_index]),
                    "local": float(local_flat[flat_index]),
                    "difference": float(
                        imported_flat[flat_index] - local_flat[flat_index]
                    ),
                }
                for flat_index, coordinate in zip(sample_flat, sample_coordinates)
            ]
            # The projection mismatch is small.  Preserve its complete axis
            # distribution without letting a broadly different later stage
            # inflate diagnostic JSON or memory use.
            if mismatch_count <= 4096:
                coordinates = np.stack(
                    np.unravel_index(mismatch_flat, imported_values.shape), axis=1
                )
                for axis in range(coordinates.shape[1]):
                    values, counts = np.unique(
                        coordinates[:, axis], return_counts=True
                    )
                    order = np.argsort(-counts, kind="stable")[:16]
                    mismatch_summary["axes"].append({
                        "axis": axis,
                        "distinct_indices": int(values.size),
                        "minimum_index": int(values[0]),
                        "maximum_index": int(values[-1]),
                        "top": [
                            {
                                "index": int(values[position]),
                                "count": int(counts[position]),
                            }
                            for position in order
                        ],
                    })
        result.update({
            "numerical_metrics_available": bool(result["shape_match"]),
            "contains_nonfinite": False,
            "imported_nan_count": 0,
            "imported_positive_inf_count": 0,
            "imported_negative_inf_count": 0,
            "local_nan_count": 0,
            "local_positive_inf_count": 0,
            "local_negative_inf_count": 0,
            "mismatch_summary": mismatch_summary,
        })
        return result
    except HandoffError as error:
        if "non-finite" not in str(error):
            raise

    imported_values = np.ascontiguousarray(
        _float32_numpy(imported, mx_module=mx_module)
    )
    local_values = np.ascontiguousarray(
        _float32_numpy(local, mx_module=mx_module)
    )
    imported_shape = tuple(int(item) for item in imported_values.shape)
    local_shape = tuple(int(item) for item in local_values.shape)
    if imported_shape != local_shape:
        raise HandoffError(
            "diagnostic tensor geometry changed while reporting non-finite values"
        )

    imported_flat = imported_values.reshape(-1)
    local_flat = local_values.reshape(-1)
    imported_bits = imported_flat.view(np.uint32)
    local_bits = local_flat.view(np.uint32)
    count = int(imported_flat.size)
    bitwise_exact = int(np.count_nonzero(imported_bits == local_bits))

    def nonfinite_counts(values: np.ndarray, prefix: str) -> dict[str, int]:
        return {
            f"{prefix}_nan_count": int(np.count_nonzero(np.isnan(values))),
            f"{prefix}_positive_inf_count": int(
                np.count_nonzero(np.isposinf(values))
            ),
            f"{prefix}_negative_inf_count": int(
                np.count_nonzero(np.isneginf(values))
            ),
        }

    return {
        "imported_shape": list(imported_shape),
        "local_shape": list(local_shape),
        "imported_dtype": str(imported.dtype),
        "local_dtype": str(local.dtype),
        "shape_match": True,
        "element_count": count,
        # Numeric equality is deliberately unavailable when either side is
        # non-finite.  Raw float32 bits remain useful forensic evidence.
        "exact_elements": None,
        "exact_fraction": None,
        "bitwise_exact_elements": bitwise_exact,
        "bitwise_exact_fraction": bitwise_exact / count if count else 1.0,
        "max_abs": None,
        "mean_abs": None,
        "relative_l2": None,
        "cosine_similarity": None,
        "imported_float32_sha256": hashlib.sha256(
            imported_values.view(np.uint8)
        ).hexdigest(),
        "local_float32_sha256": hashlib.sha256(
            local_values.view(np.uint8)
        ).hexdigest(),
        "numerical_metrics_available": False,
        "contains_nonfinite": True,
        **nonfinite_counts(imported_flat, "imported"),
        **nonfinite_counts(local_flat, "local"),
    }


class MacKDAIntermediateTrace:
    """Capture the actual fused Metal prompt path for one KDA layer."""

    def __init__(self, runtime: Any, layer_index: int, mx_module: Any) -> None:
        self.runtime = runtime
        self.layer_index = int(layer_index)
        self.mx = mx_module
        layers = runtime.model.layers
        if not 0 <= self.layer_index < len(layers):
            raise HandoffError(f"Mac KDA trace layer {self.layer_index} is out of range")
        self.target = layers[self.layer_index].attn
        if not layers[self.layer_index].is_linear:
            raise HandoffError(f"Mac layer {self.layer_index} is not KDA")
        self.kda_class = type(self.target)
        from tensorfold.kernels.glm.flash.v1 import prompt as prompt_module

        self.prompt = prompt_module
        self._original_call = self.kda_class.__call__
        self._original_prompt = self.kda_class._prompt
        self._original_pre = prompt_module.kda_pre
        self._original_post = prompt_module.kda_post
        self._active = False
        self._captures: dict[str, Any] = {}
        self._prompt_calls = 0

    def _capture(self, stage: str, value: Any) -> None:
        if stage in self._captures:
            raise HandoffError(f"Mac KDA intermediate {stage!r} was captured more than once")
        self._captures[stage] = self.mx.contiguous(value)

    def __enter__(self) -> "MacKDAIntermediateTrace":
        trace = self

        def traced_call(owner, x, caches, lengths, decode):
            if owner is trace.target:
                if decode:
                    raise HandoffError("Mac KDA intermediate tracing observed a decode call")
                trace._capture("input", x)
            return trace._original_call(owner, x, caches, lengths, decode)

        def traced_prompt(owner, projection, cache):
            if owner is not trace.target:
                return trace._original_prompt(owner, projection, cache)
            if trace._active:
                raise HandoffError("Mac KDA intermediate tracing re-entered its target layer")
            trace._prompt_calls += 1
            trace._capture("projection", projection)
            trace._active = True
            try:
                result = trace._original_prompt(owner, projection, cache)
                trace._capture("state", cache.ssm)
                return result
            finally:
                trace._active = False

        def traced_pre(owner, projection, conv, forget_projection, *args, **kwargs):
            result = trace._original_pre(
                owner,
                projection,
                conv,
                forget_projection,
                *args,
                **kwargs,
            )
            if trace._active:
                q, key, value, decay, beta = result
                # CUDA's retained serial scratch does not expose q.  Keep the
                # shared stages only so the first reported mismatch is honest.
                del q
                trace._capture("forget_projection", forget_projection)
                trace._capture("key", key)
                trace._capture("value", value)
                trace._capture("decay", decay)
                trace._capture("beta", beta)
            return result

        def traced_post(owner, value, output_gate_projection, *args, **kwargs):
            result = trace._original_post(
                owner,
                value,
                output_gate_projection,
                *args,
                **kwargs,
            )
            if trace._active:
                trace._capture("output_gate_projection", output_gate_projection)
                trace._capture("output", result)
            return result

        self.kda_class.__call__ = traced_call
        self.kda_class._prompt = traced_prompt
        self.prompt.kda_pre = traced_pre
        self.prompt.kda_post = traced_post
        return self

    def __exit__(self, *_args) -> None:
        self.kda_class.__call__ = self._original_call
        self.kda_class._prompt = self._original_prompt
        self.prompt.kda_pre = self._original_pre
        self.prompt.kda_post = self._original_post

    def captured(self) -> dict[str, Any]:
        if self._prompt_calls != 1:
            raise HandoffError(
                "Mac KDA intermediate tracing requires exactly one fused target-layer prompt chunk"
            )
        missing = sorted(set(KDA_INTERMEDIATE_STAGES) - set(self._captures))
        if missing:
            raise HandoffError(f"Mac KDA intermediate capture is incomplete: {missing}")
        self.mx.eval(*self._captures.values())
        return dict(self._captures)


def _canonical_mac_kda_intermediate(stage: str, value: Any) -> Any:
    """Remove MLX's one-stream batch axis where CUDA stores no such axis."""

    if stage in {"key", "value", "decay", "beta", "state"}:
        if len(value.shape) < 1 or int(value.shape[0]) != 1:
            raise HandoffError(f"Mac KDA intermediate {stage!r} has no one-stream axis")
        return value[0]
    return value


def compare_kda_intermediates(
    assembler: ChunkAssembler | ProgressiveAssembler,
    local_values: dict[str, Any],
    *,
    layer_index: int,
    mx_module: Any,
) -> dict[str, Any]:
    """Compare aligned Spark and Mac values in execution order."""

    names = kda_intermediate_names(layer_index)
    stages: list[dict[str, Any]] = []
    first_difference = None
    for stage in KDA_INTERMEDIATE_STAGES:
        imported = _mlx_tensor(assembler, names[stage], mx_module)
        local = _canonical_mac_kda_intermediate(stage, local_values[stage])
        comparison = _compare_diagnostic_tensor_values(
            imported,
            local,
            mx_module=mx_module,
        )
        exact = bool(
            comparison["shape_match"]
            and not comparison["contains_nonfinite"]
            and comparison.get("exact_elements") == comparison.get("element_count")
        )
        item = {"stage": stage, "exact": exact, **comparison}
        stages.append(item)
        if first_difference is None and not exact:
            first_difference = stage
    return {
        "layer": int(layer_index),
        "stage_count": len(stages),
        "exact_stage_count": sum(1 for item in stages if item["exact"]),
        "first_difference": first_difference,
        "stages": stages,
    }


def import_prefix(
    runtime: Any,
    manifest: HandoffManifest | ProgressiveManifest,
    assembler: ChunkAssembler | ProgressiveAssembler,
    *,
    mx_module: Any = None,
    diagnostic_kda_layer: int | None = None,
) -> tuple[list[Any], Any]:
    """Build TensorFold MLX cache objects and final prompt row."""

    if assembler.manifest != manifest:
        raise HandoffError("cache assembler and manifest identities differ")
    manifest.validate()
    if manifest.long_context:
        raise HandoffError("sparse long-context state is not supported by cache handoff v2")
    if mx_module is None:
        import mlx.core as mx_module
    mx = mx_module

    cache = runtime.make_cache()
    expected = _cache_tensor_names(cache) | {"prompt.final_hidden"}
    actual = {item.name for item in manifest.tensors}
    # The Spark exporter always includes its MTP prefix.  A Mac DFlash runtime ignores that one tensor and
    # initializes its own small draft state from the first locally verified decode row.
    allowed = expected | ({"mtp.keys"} if "mtp.keys" not in expected else set())
    if diagnostic_kda_layer is not None:
        allowed |= set(kda_intermediate_names(diagnostic_kda_layer).values())
    if actual != allowed:
        missing = sorted(allowed - actual)
        extra = sorted(actual - allowed)
        raise HandoffError(f"cache tensor set differs (missing={missing}, extra={extra})")

    cached = int(manifest.cached_tokens)
    mtp_cached = int(manifest.mtp_cached_tokens)
    index_width = int(getattr(runtime.args, "index_head_dim", 128))
    kpool = int(getattr(runtime.args, "index_kpool", 4))
    dense_limit = int(getattr(runtime.args, "index_topk", 2048))
    latent_width = int(getattr(runtime.args, "kv_lora_rank", 512))
    linear_heads = int(getattr(runtime.args, "linear_num_heads", 64))
    linear_dim = int(getattr(runtime.args, "linear_head_dim", 128))
    linear_conv = int(getattr(runtime.args, "linear_conv", 4))
    hidden_width = int(getattr(runtime.args, "hidden_size", 0))
    if cached >= dense_limit:
        raise HandoffError(
            f"cache has {cached} prefix tokens; handoff v2 must leave the Mac below "
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
    if "mtp.keys" in expected:
        mtp.keys = _mlx_tensor(assembler, "mtp.keys", mx)
        if tuple(mtp.keys.shape) != (mtp_cached, latent_width):
            raise HandoffError("MTP key geometry differs from the local GLM runtime")
        dtype = mtp.keys.dtype
        mtp.ik = mx.zeros((mtp_cached, index_width), dtype=dtype)
        mtp.ig = mx.zeros((mtp_cached, index_width), dtype=dtype)
        mtp.pool = mx.zeros((-(-mtp_cached // kpool), index_width), dtype=dtype)
        mtp.offset = mtp_cached
        mtp.drafted = 0
        evaluations.extend((mtp.keys, mtp.ik, mtp.ig, mtp.pool))
    final_hidden = _mlx_tensor(assembler, "prompt.final_hidden", mx)
    if hidden_width <= 0 or tuple(final_hidden.shape) != (1, hidden_width):
        raise HandoffError("final prompt row geometry differs from the local GLM runtime")
    evaluations.append(final_hidden)
    mx.eval(*evaluations)
    return cache, final_hidden


def import_cache(
    runtime: Any,
    manifest: HandoffManifest | ProgressiveManifest,
    assembler: ChunkAssembler | ProgressiveAssembler,
    *,
    mx_module: Any = None,
) -> list[Any]:
    """Compatibility wrapper returning only the imported cache."""

    cache, _final_hidden = import_prefix(
        runtime, manifest, assembler, mx_module=mx_module
    )
    return cache


def _eos_ids(tokenizer: Any) -> frozenset[int]:
    value = getattr(tokenizer, "eos_token_ids", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    return frozenset(int(item) for item in values if item is not None)


def _request_eos_ids(tokenizer: Any, *, force_length: bool) -> frozenset[int]:
    """Return the stop set for one request.

    MiaAI/sparkDash measures exactly 400 generated tokens with EOS ignored.  A
    separate helper keeps that benchmark behavior explicit without changing
    normal serving semantics.
    """

    return frozenset() if force_length else _eos_ids(tokenizer)


def _first_token_difference(left: Sequence[int], right: Sequence[int]) -> int | None:
    """Return the first unequal token position, including a length mismatch."""

    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if int(left_token) != int(right_token):
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def render_user_prompt(tokenizer: Any, prompt: str) -> list[int]:
    """Render the same thinking-off chat prompt as TensorFold's HTTP server."""

    from tensorfold.server.text import render_prompt_ids

    return render_prompt_ids(
        tokenizer,
        [{"role": "user", "content": prompt}],
        enable_thinking=False,
        add_generation_prompt=True,
    )


def _load_runtime(
    model_dir: Path,
    *,
    mtp_drafts: int = 3,
    drafter: str = "",
    drafter_bits: int = 4,
) -> tuple[Any, Any]:
    """Load through TensorFold's public GLM entry point.

    The public loader applies the Metal working-set limit that keeps the large
    GLM weights resident between rounds. Calling ``runtime.load`` directly
    skips that protection and can make an otherwise identical round page for
    minutes.
    """

    from tensorfold.families.glm5_next import load

    if not 1 <= int(mtp_drafts) <= 8:
        raise ValueError("MTP draft count must be in 1..8")
    return load(
        Path(model_dir),
        mtp_drafts=int(mtp_drafts),
        drafter=str(drafter),
        drafter_bits=int(drafter_bits),
    )


def _new_engine(runtime: Any, *, imported: bool) -> Any:
    from tensorfold.engine.lane_engine import LaneEngine
    from tensorfold.families.glm5_next import engine_settings

    engine = LaneEngine(runtime, **engine_settings(runtime))
    if imported:
        # A cross-host prefix need not end on the local prompt planner's chunk
        # boundary.  TensorFold's cache contract still validates its position.
        engine.prefill_plan = None
    return engine


def _copy_proposer(*, enabled: bool, min_match: int) -> Any:
    """Match TensorFold's normal server-side copy-draft policy."""

    if not enabled:
        return None
    if int(min_match) < 1:
        raise ValueError("copy-draft minimum match must be positive")
    from tensorfold.engine.lane_engine import SuffixLookupProposer

    return SuffixLookupProposer(min_match=int(min_match))


def _release_engine(engine: Any) -> None:
    """Release the completed round and detach every stream from its engine."""

    try:
        engine.release_rounds()
    finally:
        engine.reset()


def _proposer_telemetry(proposer: Any) -> dict[str, Any] | None:
    """Read one retained local draft head's counters without changing it."""

    report = getattr(proposer, "telemetry", None)
    if proposer is None or not callable(report):
        return None
    value = report()
    return dict(value) if isinstance(value, dict) else None


def _cache_drafter_proposer(cache: Sequence[Any]) -> Any:
    """Return a cache's measurable local draft proposer, when present."""

    if not cache:
        return None
    proposer = getattr(cache[-1], "proposer", None)
    return proposer if callable(getattr(proposer, "telemetry", None)) else None


def _engine_drafter_proposer(
    engine: Any,
    stream_id: str,
    fallback_cache: Sequence[Any],
) -> Any:
    """Find a live stream's actual working-cache proposer.

    TensorFold removes finished streams from its live table.  Keep the proposer
    itself while the stream is live so its final counters survive cleanup.
    """

    for stream, work_cache in getattr(engine, "_live", ()):
        if getattr(stream, "stream_id", None) == stream_id:
            proposer = _cache_drafter_proposer(work_cache)
            if proposer is not None:
                return proposer
    return _cache_drafter_proposer(fallback_cache)


def _cache_drafter_telemetry(cache: Sequence[Any]) -> dict[str, Any] | None:
    """Read the local draft head's counters without changing its state."""

    return _proposer_telemetry(_cache_drafter_proposer(cache))


def _reclaim_mlx_memory() -> None:
    """Return freed round buffers to MLX before the next measured round."""

    import mlx.core as mx

    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def build_local_handoff_cache(
    runtime: Any,
    prompt_ids: Sequence[int],
    *,
    prefill_rows: int = 128,
    mx_module: Any = None,
) -> list[Any]:
    """Mirror the Spark export boundary using the Mac runtime.

    The Spark exporter prefills every prompt token except the last one, then
    gives the MTP head that final prompt token before handing the cache to the
    Mac.  Reproducing that boundary locally separates a cross-backend numeric
    difference from a boundary/chunking error in the handoff protocol.
    """

    ids = [int(token) for token in prompt_ids]
    if len(ids) < 2:
        raise HandoffError("cache handoff needs at least two prompt tokens")
    if int(prefill_rows) < 1:
        raise ValueError("prefill_rows must be positive")
    if mx_module is None:
        import mlx.core as mx_module
    mx = mx_module

    cache = runtime.make_cache()
    prefix = ids[:-1]
    final_hidden = None
    for start in range(0, len(prefix), int(prefill_rows)):
        end = min(start + int(prefill_rows), len(prefix))
        runtime.hidden(mx.array([prefix[start:end]], dtype=mx.uint32), cache)
        rows = runtime.draft_rows()
        following = prefix[start + 1:end + 1]
        if following:
            runtime.absorb_draft_context(rows, following, cache)
        final_hidden = rows.reshape(-1, rows.shape[-1])[-1:]
    if final_hidden is None:
        raise HandoffError("local handoff mirror produced no prompt state")
    runtime.absorb_draft_context(final_hidden, [ids[-1]], cache)

    from tensorfold.engine.family_common import cache_arrays

    mx.eval(*cache_arrays(cache))
    return cache


def build_local_full_prompt_continuation_state(
    runtime: Any,
    prompt_ids: Sequence[int],
    first_token: int,
    *,
    prefill_rows: int = 128,
    mx_module: Any = None,
) -> tuple[list[Any], Any]:
    """Prefill the whole prompt and return its cache plus final hidden row.

    This models a lossless v2 boundary: the Sparks finish prompt prefill and
    provide the first greedy token, then the Mac resumes with that token as
    the only uncached target-model row.  Unlike handoff v1, the last prompt
    token is computed by the prompt kernel instead of the decode kernel.
    """

    ids = [int(token) for token in prompt_ids]
    if not ids:
        raise HandoffError("full-prompt continuation needs at least one prompt token")
    if int(prefill_rows) < 1:
        raise ValueError("prefill_rows must be positive")
    if mx_module is None:
        import mlx.core as mx_module
    mx = mx_module

    cache = runtime.make_cache()
    final_hidden = None
    for start in range(0, len(ids), int(prefill_rows)):
        end = min(start + int(prefill_rows), len(ids))
        runtime.hidden(mx.array([ids[start:end]], dtype=mx.uint32), cache)
        rows = runtime.draft_rows()
        following = ids[start + 1:end + 1]
        if following:
            runtime.absorb_draft_context(rows, following, cache)
        final_hidden = rows.reshape(-1, rows.shape[-1])[-1:]
    if final_hidden is None:
        raise HandoffError("full-prompt continuation produced no prompt state")
    runtime.absorb_draft_context(final_hidden, [int(first_token)], cache)

    from tensorfold.engine.family_common import cache_arrays

    mx.eval(*cache_arrays(cache))
    return cache, final_hidden


def build_local_full_prompt_continuation_cache(
    runtime: Any,
    prompt_ids: Sequence[int],
    first_token: int,
    *,
    prefill_rows: int = 128,
    mx_module: Any = None,
) -> list[Any]:
    """Compatibility wrapper returning only the aligned local cache."""

    cache, _final_hidden = build_local_full_prompt_continuation_state(
        runtime,
        prompt_ids,
        first_token,
        prefill_rows=prefill_rows,
        mx_module=mx_module,
    )
    return cache


def prepare_first_token_continuation(
    runtime: Any,
    cache: list[Any],
    final_hidden: Any,
    *,
    expected_token: int | None = None,
    mx_module: Any = None,
) -> int:
    """Draw the first greedy token on the Mac and prime the MTP cache."""

    if mx_module is None:
        import mlx.core as mx_module
    mx = mx_module
    logits = runtime.head(final_hidden)
    row = logits.reshape(-1, logits.shape[-1])[-1]
    first = mx.argmax(row)
    mx.eval(first)
    token = int(first.item())
    if expected_token is not None and token != int(expected_token):
        raise HandoffError(
            f"Spark and Mac first tokens differ ({int(expected_token)} != {token})"
        )
    runtime.absorb_draft_context(final_hidden, [token], cache)
    from tensorfold.engine.family_common import cache_arrays

    mx.eval(*cache_arrays(cache))
    return token


def _logit_summary(
    logits: Any,
    *,
    expected_token: int,
    top_count: int = 2,
    mx_module: Any = None,
) -> dict[str, Any]:
    """Return the greedy choice, margin, and expected-token rank."""

    row = _float32_numpy(logits, mx_module=mx_module).reshape(-1)
    if not 0 <= int(expected_token) < int(row.size):
        raise HandoffError("reference token is outside the model vocabulary")
    if not np.all(np.isfinite(row)):
        raise HandoffError("model logits contain non-finite values")
    count = min(max(2, int(top_count)), int(row.size))
    candidates = np.argpartition(row, -count)[-count:]
    ordered = sorted((int(index) for index in candidates), key=lambda index: (-float(row[index]), index))
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else best
    expected = int(expected_token)
    expected_value = float(row[expected])
    return {
        "token": best,
        "logit": float(row[best]),
        "second_token": second,
        "second_logit": float(row[second]),
        "margin": float(row[best] - row[second]),
        "expected_token": expected,
        "expected_logit": expected_value,
        "expected_rank": 1 + int(np.count_nonzero(row > expected_value)),
        "top": [
            {"token": index, "logit": float(row[index])}
            for index in ordered
        ],
    }


def compare_aligned_cache_states(
    imported_cache: Sequence[Any],
    local_cache: Sequence[Any],
    manifest: HandoffManifest | ProgressiveManifest,
    *,
    dense_limit: int = 2048,
    mx_module: Any = None,
) -> dict[str, Any]:
    """Compare the transferred state with a Mac-prefilled state by semantic name."""

    imported = _cache_state_tensors(imported_cache)
    local = _cache_state_tensors(local_cache)
    transferred = {
        item.name for item in manifest.tensors if item.name != "prompt.final_hidden"
    }
    common = sorted(transferred & imported.keys() & local.keys())
    skipped = sorted(transferred - set(common))
    unexpected_imported = sorted(set(imported) - transferred)
    unexpected_local = sorted(set(local) - transferred)
    tensors = []
    for name in common:
        imported_value = imported[name]
        local_value = local[name]
        # MLX grows attention caches in allocation blocks, while the Spark
        # wire format contains only active rows.  Compare the semantic prefix,
        # not unused reserved capacity.
        if name.endswith(".keys"):
            active_rows = (
                int(manifest.mtp_cached_tokens)
                if name == "mtp.keys"
                else int(manifest.cached_tokens)
            )
            if imported_value.shape[0] >= active_rows:
                imported_value = imported_value[:active_rows]
            if local_value.shape[0] >= active_rows:
                local_value = local_value[:active_rows]
        comparison = compare_tensor_values(
            imported_value, local_value, mx_module=mx_module
        )
        comparison["name"] = name
        comparison["imported_storage_shape"] = [
            int(item) for item in imported[name].shape
        ]
        comparison["local_storage_shape"] = [
            int(item) for item in local[name].shape
        ]
        tensors.append(comparison)
    comparable = [item for item in tensors if item["shape_match"]]
    exact_tensors = sum(item["exact_fraction"] == 1.0 for item in comparable)
    worst = max(
        comparable,
        key=lambda item: (
            -1.0 if item["relative_l2"] is None else float(item["relative_l2"])
        ),
        default=None,
    )
    dense_limit = int(dense_limit)
    if dense_limit < 1:
        raise ValueError("dense_limit must be positive")
    return {
        "transferred_tensor_count": len(transferred),
        "compared_tensor_count": len(tensors),
        "shape_matched_tensor_count": len(comparable),
        "exact_tensor_count": exact_tensors,
        "skipped_transferred_tensors": skipped,
        "cache_tensors_not_on_wire_imported": unexpected_imported,
        "cache_tensors_not_on_wire_local": unexpected_local,
        "prompt_tokens": int(manifest.cached_tokens),
        "dense_attention_limit": dense_limit,
        "sparse_index_state_transferred": bool(manifest.long_context),
        "sparse_index_state_used_for_this_prompt": int(manifest.cached_tokens) >= dense_limit,
        "worst_relative_l2_tensor": None if worst is None else worst["name"],
        "tensors": tensors,
    }


def trace_teacher_forced_divergence(
    runtime: Any,
    imported_cache: list[Any],
    local_cache: list[Any],
    reference_tokens: Sequence[int],
    *,
    mx_module: Any = None,
) -> dict[str, Any]:
    """Advance both caches on the same tokens until their greedy choices split."""

    if mx_module is None:
        import mlx.core as mx_module
    mx = mx_module
    tokens = [int(token) for token in reference_tokens]
    if len(tokens) < 2:
        raise HandoffError("cache-state tracing needs at least two reference tokens")
    steps: list[dict[str, Any]] = []
    first_imported_reference_difference = None
    first_local_reference_difference = None
    first_cross_backend_difference = None
    divergence = None
    for index in range(1, len(tokens)):
        fed = tokens[index - 1]
        expected = tokens[index]
        imported_hidden = runtime.hidden(
            mx.array([[fed]], dtype=mx.uint32), imported_cache
        )
        imported_logits = runtime.head(imported_hidden)
        local_hidden = runtime.hidden(
            mx.array([[fed]], dtype=mx.uint32), local_cache
        )
        local_logits = runtime.head(local_hidden)
        mx.eval(imported_hidden, imported_logits, local_hidden, local_logits)
        imported_summary = _logit_summary(
            imported_logits,
            expected_token=expected,
            mx_module=mx,
        )
        local_summary = _logit_summary(
            local_logits,
            expected_token=expected,
            mx_module=mx,
        )
        hidden = compare_tensor_values(
            imported_hidden, local_hidden, mx_module=mx
        )
        steps.append({
            "generated_token_index": index,
            "fed_token": fed,
            "expected_token": expected,
            "imported_token": imported_summary["token"],
            "imported_margin": imported_summary["margin"],
            "imported_expected_rank": imported_summary["expected_rank"],
            "local_token": local_summary["token"],
            "local_margin": local_summary["margin"],
            "local_expected_rank": local_summary["expected_rank"],
            "hidden_max_abs": hidden["max_abs"],
            "hidden_mean_abs": hidden["mean_abs"],
            "hidden_relative_l2": hidden["relative_l2"],
            "hidden_cosine_similarity": hidden["cosine_similarity"],
        })
        if imported_summary["token"] != expected and first_imported_reference_difference is None:
            first_imported_reference_difference = index
        if local_summary["token"] != expected and first_local_reference_difference is None:
            first_local_reference_difference = index
        if imported_summary["token"] != local_summary["token"]:
            first_cross_backend_difference = index
            divergence = {
                "generated_token_index": index,
                "fed_token": fed,
                "expected_token": expected,
                "imported": _logit_summary(
                    imported_logits,
                    expected_token=expected,
                    top_count=5,
                    mx_module=mx,
                ),
                "local": _logit_summary(
                    local_logits,
                    expected_token=expected,
                    top_count=5,
                    mx_module=mx,
                ),
                "hidden": hidden,
                "logits": compare_tensor_values(
                    imported_logits, local_logits, mx_module=mx
                ),
            }
            break
    return {
        "reference_token_count": len(tokens),
        "traced_prediction_count": len(steps),
        "stopped_at_first_cross_backend_difference": divergence is not None,
        "first_imported_reference_difference": first_imported_reference_difference,
        "first_local_reference_difference": first_local_reference_difference,
        "first_cross_backend_difference": first_cross_backend_difference,
        "divergence": divergence,
        "steps": steps,
    }


def _cache_transplant_plan(cache: Sequence[Any]) -> list[dict[str, Any]]:
    """Return deterministic local-state substitutions for one cache diagnosis.

    The broad variants answer which cache family is necessary or sufficient.
    Three family partitions then localize a useful range without pretending
    that independently computed layer errors are additive.
    """

    if len(cache) < 2:
        raise HandoffError("cache transplant diagnostic needs a complete GLM cache")
    kda: list[int] = []
    mla: list[int] = []
    for index, item in enumerate(cache[:-1]):
        if hasattr(item, "conv") and hasattr(item, "ssm"):
            kda.append(index)
        elif all(hasattr(item, field) for field in ("keys", "ik", "ig", "pool")):
            mla.append(index)
        else:
            raise HandoffError(f"cache layer {index} has an unsupported cache type")
    draft = len(cache) - 1
    all_indices = tuple(range(len(cache)))
    plan: list[dict[str, Any]] = [
        {"name": "all_local", "local_indices": all_indices},
        {"name": "local_kda_only", "local_indices": tuple(kda)},
        {"name": "local_mla_only", "local_indices": tuple(mla)},
        {"name": "local_draft_only", "local_indices": (draft,)},
        {"name": "local_kda_and_draft", "local_indices": (*kda, draft)},
        {"name": "local_mla_and_draft", "local_indices": (*mla, draft)},
        {"name": "local_layers_without_draft", "local_indices": (*kda, *mla)},
    ]

    for family, indices in (("kda", kda), ("mla", mla)):
        parts = min(3, len(indices))
        for part in range(parts):
            start = part * len(indices) // parts
            end = (part + 1) * len(indices) // parts
            group = tuple(indices[start:end])
            if not group:
                continue
            label = f"{family}_layers_{group[0]}_{group[-1]}"
            plan.append({"name": f"local_{label}_only", "local_indices": group})
            excluded = set(group)
            plan.append({
                "name": f"all_local_except_{label}",
                "local_indices": tuple(index for index in all_indices if index not in excluded),
            })
    return plan


def trace_cache_transplant_variants(
    runtime: Any,
    imported_cache: list[Any],
    local_cache: list[Any],
    reference_tokens: Sequence[int],
    *,
    mx_module: Any = None,
) -> list[dict[str, Any]]:
    """Locate divergent cache families by substituting Mac-local state.

    Each variant starts from fresh cache objects. Arrays may be shared at the
    immutable prefix boundary, while later decode writes replace their owning
    cache attributes. Only compact first-divergence evidence is retained.
    """

    from tensorfold.engine.lane_engine import LaneEngine

    results: list[dict[str, Any]] = []
    for variant in _cache_transplant_plan(imported_cache):
        hybrid = LaneEngine.copy_single_cache(imported_cache)
        donor = LaneEngine.copy_single_cache(local_cache)
        reference = LaneEngine.copy_single_cache(local_cache)
        local_indices = tuple(int(index) for index in variant["local_indices"])
        for index in local_indices:
            hybrid[index] = donor[index]
        try:
            trace = trace_teacher_forced_divergence(
                runtime,
                hybrid,
                reference,
                reference_tokens,
                mx_module=mx_module,
            )
            results.append({
                "name": variant["name"],
                "local_cache_indices": list(local_indices),
                "local_layer_indices": [
                    index for index in local_indices if index < len(imported_cache) - 1
                ],
                "local_draft_cache": len(imported_cache) - 1 in local_indices,
                "traced_prediction_count": trace["traced_prediction_count"],
                "first_reference_difference": trace["first_imported_reference_difference"],
                "first_local_difference": trace["first_local_reference_difference"],
                "first_cross_backend_difference": trace["first_cross_backend_difference"],
                "divergence": trace["divergence"],
            })
        finally:
            hybrid.clear()
            donor.clear()
            reference.clear()
            _reclaim_mlx_memory()
    return results


def run_cache_state_diagnostic(
    runtime: Any,
    prompt_ids: Sequence[int],
    local_reference: dict[str, Any],
    *,
    mailbox_name: str,
    timeout: float,
    model_id: str,
    model_revision: str,
    chunk_bytes: int = DEFAULT_FRAME_BYTES,
    progressive_handoff: bool = False,
    diagnose_kda_layer: int | None = None,
) -> dict[str, Any]:
    """Compare one real Spark cache with the Mac cache and trace its first split."""

    import mlx.core as mx

    telemetry: dict[str, Any] = {}
    transfer_started = time.perf_counter()
    with ClientMailbox(mailbox_name, ready_timeout_s=min(timeout, 30.0)) as mailbox:
        fetcher = fetch_progressive_cache if progressive_handoff else fetch_cache
        manifest, assembler, frame_lengths = fetcher(
            mailbox,
            prompt_ids,
            model_id=model_id,
            model_revision=model_revision,
            timeout=timeout,
            chunk_bytes=chunk_bytes,
            telemetry=telemetry,
        )
    transfer_seconds = time.perf_counter() - transfer_started
    imported_cache: list[Any] | None = None
    local_cache: list[Any] | None = None
    try:
        import_started = time.perf_counter()
        imported_cache, imported_final_hidden = import_prefix(
            runtime,
            manifest,
            assembler,
            mx_module=mx,
            diagnostic_kda_layer=diagnose_kda_layer,
        )
        import_seconds = time.perf_counter() - import_started
        imported_first_token = prepare_first_token_continuation(
            runtime,
            imported_cache,
            imported_final_hidden,
            expected_token=manifest.spark_first_token,
            mx_module=mx,
        )
        local_intermediates = None
        if diagnose_kda_layer is None:
            local_cache, local_final_hidden = build_local_full_prompt_continuation_state(
                runtime,
                prompt_ids,
                imported_first_token,
                mx_module=mx,
            )
        else:
            with MacKDAIntermediateTrace(runtime, diagnose_kda_layer, mx) as trace:
                local_cache, local_final_hidden = build_local_full_prompt_continuation_state(
                    runtime,
                    prompt_ids,
                    imported_first_token,
                    mx_module=mx,
                )
            local_intermediates = trace.captured()
        reference_tokens = [int(token) for token in local_reference.get("tokens", ())]
        if not reference_tokens:
            raise HandoffError("cache-state diagnostic has no Mac-local reference tokens")
        local_first = _logit_summary(
            runtime.head(local_final_hidden),
            expected_token=reference_tokens[0],
            top_count=5,
            mx_module=mx,
        )
        imported_first = _logit_summary(
            runtime.head(imported_final_hidden),
            expected_token=reference_tokens[0],
            top_count=5,
            mx_module=mx,
        )
        if not (
            imported_first_token
            == int(manifest.spark_first_token)
            == reference_tokens[0]
            == imported_first["token"]
            == local_first["token"]
        ):
            raise HandoffError("Spark, imported, and local first-token choices are not aligned")
        final_hidden = compare_tensor_values(
            imported_final_hidden, local_final_hidden, mx_module=mx
        )
        cache = compare_aligned_cache_states(
            imported_cache,
            local_cache,
            manifest,
            dense_limit=int(getattr(runtime.args, "index_topk", 2048)),
            mx_module=mx,
        )
        transplants = trace_cache_transplant_variants(
            runtime,
            imported_cache,
            local_cache,
            reference_tokens,
            mx_module=mx,
        )
        trace = trace_teacher_forced_divergence(
            runtime,
            imported_cache,
            local_cache,
            reference_tokens,
            mx_module=mx,
        )
        kda_intermediates = (
            None
            if local_intermediates is None
            else compare_kda_intermediates(
                assembler,
                local_intermediates,
                layer_index=diagnose_kda_layer,
                mx_module=mx,
            )
        )
        return {
            "event": "glm_mcdma_cache_state_diagnostic",
            "model_id": model_id,
            "model_revision": model_revision,
            "prompt_sha256": manifest.prompt_sha256,
            "prompt_tokens": len(prompt_ids),
            "reference_token_sha256": local_reference.get("token_sha256"),
            "reference_token_count": len(reference_tokens),
            "cache_protocol": manifest.protocol,
            "transfer_id": manifest.transfer_id,
            "cache_bytes": sum(item.nbytes for item in manifest.tensors),
            "cache_frame_count": len(frame_lengths),
            "cache_frame_lengths": list(frame_lengths),
            "transfer_seconds": transfer_seconds,
            "import_seconds": import_seconds,
            "handoff_telemetry": telemetry,
            "mcdma_archive_verified": True,
            "spark_reference_first_token": manifest.spark_first_token,
            "imported_first_token": imported_first,
            "local_first_token": local_first,
            "final_hidden": final_hidden,
            "cache_state": cache,
            "cache_transplants": transplants,
            "teacher_forced_trace": trace,
            "kda_intermediates": kda_intermediates,
        }
    finally:
        del assembler
        if imported_cache is not None:
            imported_cache.clear()
        if local_cache is not None:
            local_cache.clear()
        _reclaim_mlx_memory()


def _prepend_first_token_result(
    result: dict[str, Any],
    *,
    first_token: int,
    tokenizer: Any,
    first_at: float,
    entered: float,
    origin: float,
) -> dict[str, Any]:
    """Turn a continuation result into the complete user-visible reply."""

    tokens = [int(first_token), *[int(token) for token in result["tokens"]]]
    completion_from_origin = float(result["completion_from_origin_seconds"])
    first_from_origin = first_at - origin
    total_seconds = completion_from_origin - (entered - origin)
    generation_seconds = completion_from_origin - first_from_origin
    rate = (len(tokens) - 1) / generation_seconds if generation_seconds > 0 else 0.0
    combined = dict(result)
    combined.update({
        "tokens": tokens,
        "token_sha256": prompt_digest(tokens),
        "text": tokenizer.decode(tokens, skip_special_tokens=False),
        "first_token_seconds": first_at - entered,
        "first_token_from_origin_seconds": first_from_origin,
        "generation_seconds": generation_seconds,
        "total_seconds": total_seconds,
        "subsequent_tokens_per_second": rate,
        "decode_tokens_per_second": rate,
        "end_to_end_tokens_per_second": (
            len(tokens) / total_seconds if total_seconds > 0 else 0.0
        ),
        "first_token_source": "mac_head_from_spark_final_hidden",
    })
    return combined


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
    copy_drafts: bool = True,
    copy_min_match: int = 4,
    force_length: bool = False,
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
            f"cache handoff v2 is limited to {dense_limit}"
        )
    engine = _new_engine(runtime, imported=cache is not None)
    try:
        stream = LaneStream(
            stream_id=stream_id,
            prompt_ids=[int(token) for token in prompt_ids],
            max_new_tokens=int(max_new_tokens),
            eos_ids=_request_eos_ids(tokenizer, force_length=force_length),
            drafts=True,
        )
        stream.proposer = _copy_proposer(enabled=copy_drafts, min_match=copy_min_match)
        engine.add_stream(stream, cache=cache, cached_tokens=int(cached_tokens))
        drafter_proposer = _engine_drafter_proposer(
            engine, stream.stream_id, cache or ()
        )
        first_at: float | None = None
        while engine.active_count:
            engine.step()
            if drafter_proposer is None:
                drafter_proposer = _engine_drafter_proposer(
                    engine, stream.stream_id, cache or ()
                )
            if first_at is None and stream.emitted:
                first_at = time.perf_counter()
        finished = time.perf_counter()
        tokens = [int(token) for token in stream.emitted]
        first_at = finished if first_at is None else first_at
        total_seconds = finished - started
        generation_seconds = finished - first_at
        subsequent_tokens = max(0, len(tokens) - 1)
        subsequent_rate = subsequent_tokens / generation_seconds if generation_seconds > 0 else 0.0
        drafter_telemetry = _proposer_telemetry(drafter_proposer)
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
            "decode_rounds": int(stream.rounds),
            "drafted_tokens": int(stream.drafted),
            "accepted_draft_tokens": int(stream.accepted),
            "draft_acceptance_rate": (
                int(stream.accepted) / int(stream.drafted) if stream.drafted else 0.0
            ),
            "copy_drafts": bool(copy_drafts),
            "force_length": bool(force_length),
            "copy_draft_telemetry": (
                stream.proposer.telemetry() if stream.proposer is not None else None
            ),
            "drafter_telemetry": drafter_telemetry,
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
    force_length: bool = False,
    copy_drafts: bool = True,
    copy_min_match: int = 4,
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
                f"cache handoff v2 is limited to {dense_limit}"
            )
        stream = LaneStream(
            stream_id=job.stream_id,
            prompt_ids=list(job.prompt_ids),
            max_new_tokens=job.max_new_tokens,
            eos_ids=_request_eos_ids(tokenizer, force_length=force_length),
            drafts=True,
        )
        stream.proposer = _copy_proposer(
            enabled=copy_drafts,
            min_match=copy_min_match,
        )
        streams.append(stream)

    engine = _new_engine(runtime, imported=True)
    try:
        for stream, job in zip(streams, jobs, strict=True):
            engine.add_stream(stream, cache=job.cache, cached_tokens=job.cached_tokens)
        drafter_proposers = {
            job.stream_id: _engine_drafter_proposer(
                engine, job.stream_id, job.cache
            )
            for job in jobs
        }
        ready_at = time.perf_counter()
        first_at: dict[str, float] = {}
        finished_at: dict[str, float] = {}
        while engine.active_count:
            engine.step()
            for job in jobs:
                if drafter_proposers[job.stream_id] is None:
                    drafter_proposers[job.stream_id] = _engine_drafter_proposer(
                        engine, job.stream_id, job.cache
                    )
            observed_at = time.perf_counter()
            for stream in streams:
                if stream.emitted and stream.stream_id not in first_at:
                    first_at[stream.stream_id] = observed_at
                if stream.finished and stream.stream_id not in finished_at:
                    finished_at[stream.stream_id] = observed_at
        finished = time.perf_counter()
        drafter_telemetry = {
            job.stream_id: _proposer_telemetry(drafter_proposers[job.stream_id])
            for job in jobs
        }

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
                "drafter_telemetry": drafter_telemetry[stream.stream_id],
            })
        first_batch_token_at = min(first_at.values(), default=finished)
        decode_seconds = finished - first_batch_token_at
        total_seconds = finished - started
        token_count = sum(len(result["tokens"]) for result in results)
        decode_token_count = sum(max(0, len(result["tokens"]) - 1) for result in results)
        output = {
            "parallel": len(results),
            "force_length": bool(force_length),
            "copy_drafts": bool(copy_drafts),
            "copy_min_match": int(copy_min_match),
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


def prepare_reusable_prefix(
    runtime: Any,
    prompt_ids: Sequence[int],
    *,
    mailbox_name: str,
    timeout: float,
    model_id: str,
    model_revision: str,
    chunk_bytes: int = DEFAULT_FRAME_BYTES,
    progressive_handoff: bool = False,
) -> ReusablePrefix:
    """Prepare once on both Sparks and retain the verified cache on the Mac."""

    handoff_telemetry: dict[str, Any] = {}
    transfer_started = time.perf_counter()
    with ClientMailbox(mailbox_name, ready_timeout_s=min(timeout, 30.0)) as mailbox:
        fetcher = fetch_progressive_cache if progressive_handoff else fetch_cache
        manifest, assembler, frame_lengths = fetcher(
            mailbox,
            prompt_ids,
            model_id=model_id,
            model_revision=model_revision,
            timeout=timeout,
            chunk_bytes=chunk_bytes,
            telemetry=handoff_telemetry,
        )
    transfer_seconds = time.perf_counter() - transfer_started
    import_started = time.perf_counter()
    cache, final_hidden = import_prefix(runtime, manifest, assembler)
    import_seconds = time.perf_counter() - import_started
    del assembler
    _reclaim_mlx_memory()
    return ReusablePrefix(
        manifest=manifest,
        cache=cache,
        frame_lengths=frame_lengths,
        transfer_seconds=transfer_seconds,
        import_seconds=import_seconds,
        final_hidden=final_hidden,
        handoff_telemetry=handoff_telemetry,
    )


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
    force_length: bool = False,
    strict_local_match: bool = True,
    reusable_prefix: ReusablePrefix | None = None,
    copy_drafts: bool = True,
    copy_min_match: int = 4,
    progressive_handoff: bool = False,
) -> dict[str, Any]:
    """Run one measured Spark-prefill, MCDMA-transfer, Mac-decode round."""

    entered = time.perf_counter()
    origin = entered if timing_origin is None else float(timing_origin)
    if origin > entered:
        raise HandoffError("handoff timing origin is in the future")
    if not 1 <= parallel <= 8:
        raise ValueError("parallel must be between 1 and 8")
    if max_new_tokens < 2:
        raise ValueError("full-prompt handoff needs at least two output tokens")
    handoff_telemetry: dict[str, Any] = {}
    if reusable_prefix is None:
        transfer_started = time.perf_counter()
        with ClientMailbox(mailbox_name, ready_timeout_s=min(timeout, 30.0)) as mailbox:
            fetcher = fetch_progressive_cache if progressive_handoff else fetch_cache
            manifest, assembler, frame_lengths = fetcher(
                mailbox,
                prompt_ids,
                model_id=model_id,
                model_revision=model_revision,
                timeout=timeout,
                chunk_bytes=chunk_bytes,
                telemetry=handoff_telemetry,
            )
        transfer_seconds = time.perf_counter() - transfer_started
        import_started = time.perf_counter()
        base_cache, final_hidden = import_prefix(runtime, manifest, assembler)
        import_seconds = time.perf_counter() - import_started
        cache_wire_transfers = 1
        cache_bytes_total = sum(item.nbytes for item in manifest.tensors)
    else:
        manifest = reusable_prefix.manifest
        manifest.require_identity(
            model_id=model_id,
            model_revision=model_revision,
            prompt_sha256=prompt_digest(prompt_ids),
            cached_tokens=len(prompt_ids),
            mtp_cached_tokens=len(prompt_ids) - 1,
        )
        base_cache = reusable_prefix.cache
        final_hidden = reusable_prefix.final_hidden
        if final_hidden is None:
            raise HandoffError("reusable prefix has no final prompt row")
        frame_lengths = ()
        transfer_seconds = 0.0
        import_seconds = 0.0
        cache_wire_transfers = 0
        cache_bytes_total = 0

    if reusable_prefix is not None:
        from tensorfold.engine.lane_engine import LaneEngine

        # The retained cache must stay immutable across measured requests.
        caches = [
            LaneEngine.copy_single_cache(base_cache) for _ in range(parallel)
        ]
        cache_reuse_copies = parallel
    elif parallel == 1:
        caches = [base_cache]
        cache_reuse_copies = 0
    else:
        from tensorfold.engine.lane_engine import LaneEngine

        caches = [base_cache] + [
            LaneEngine.copy_single_cache(base_cache) for _ in range(parallel - 1)
        ]
        cache_reuse_copies = parallel - 1
    decode_started = time.perf_counter()
    first_tokens: list[int] = []
    first_at: float | None = None
    for cache in caches:
        token = prepare_first_token_continuation(
            runtime,
            cache,
            final_hidden,
            expected_token=manifest.spark_first_token,
        )
        if first_at is None:
            first_at = time.perf_counter()
        first_tokens.append(token)
    if len(set(first_tokens)) != 1 or first_at is None:
        raise HandoffError("Mac first-token continuation is not deterministic")
    first_token = first_tokens[0]
    continuation_prompt = [*[int(token) for token in prompt_ids], first_token]
    continuation_tokens = max_new_tokens - 1

    if parallel == 1:
        continued = run_decode(
            runtime, tokenizer, continuation_prompt,
            max_new_tokens=continuation_tokens,
            cache=caches[0],
            cached_tokens=manifest.cached_tokens,
            stream_id="mcdma-handoff",
            timing_origin=origin,
            force_length=force_length,
            copy_drafts=copy_drafts,
            copy_min_match=copy_min_match,
        )
        result = _prepend_first_token_result(
            continued,
            first_token=first_token,
            tokenizer=tokenizer,
            first_at=first_at,
            entered=entered,
            origin=origin,
        )
        batch = None
        generated = [result]
    else:
        batch = run_decode_batch(
            runtime,
            tokenizer,
            [
                DecodeJob(
                    stream_id=f"mcdma-handoff-{index}",
                    prompt_ids=tuple(continuation_prompt),
                    max_new_tokens=continuation_tokens,
                    cache=cache,
                    cached_tokens=manifest.cached_tokens,
                )
                for index, cache in enumerate(caches)
            ],
            timing_origin=origin,
            force_length=force_length,
            copy_drafts=copy_drafts,
            copy_min_match=copy_min_match,
        )
        finished = time.perf_counter()
        generated = [
            _prepend_first_token_result(
                item,
                first_token=first_token,
                tokenizer=tokenizer,
                first_at=first_at,
                entered=entered,
                origin=origin,
            )
            for item in batch["results"]
        ]
        batch["results"] = generated
        batch["tokens"] = sum(len(item["tokens"]) for item in generated)
        batch["decode_tokens"] = sum(len(item["tokens"]) - 1 for item in generated)
        batch["first_token_seconds"] = first_at - entered
        batch["batch_decode_seconds"] = finished - first_at
        batch["total_seconds"] = finished - entered
        batch["aggregate_decode_tokens_per_second"] = (
            batch["decode_tokens"] / batch["batch_decode_seconds"]
            if batch["batch_decode_seconds"] > 0 else 0.0
        )
        batch["end_to_end_tokens_per_second"] = (
            batch["tokens"] / batch["total_seconds"]
            if batch["total_seconds"] > 0 else 0.0
        )
        result = generated[0]
    decode_seconds = time.perf_counter() - decode_started
    drafter_telemetry = [
        value for item in generated
        if isinstance((value := item.get("drafter_telemetry")), dict)
    ]
    completion_tokens = sum(len(item["tokens"]) for item in generated)
    cache_bytes = sum(item.nbytes for item in manifest.tensors)
    exact_token_match = None
    if local_reference is not None:
        exact_token_match = all(
            local_reference["tokens"] == item["tokens"] for item in generated
        )

    reclaim_started = time.perf_counter()
    del caches
    if reusable_prefix is None:
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
        "force_length": bool(force_length),
        "copy_drafts": bool(copy_drafts),
        "copy_min_match": int(copy_min_match),
        "cached_tokens": manifest.cached_tokens,
        "mtp_cached_tokens": manifest.mtp_cached_tokens,
        "spark_reference_first_token": manifest.spark_first_token,
        "transfer_id": manifest.transfer_id,
        "cache_protocol": manifest.protocol,
        "cache_bytes_per_request": cache_bytes,
        "cache_bytes_total": cache_bytes_total,
        "cache_frame_count": len(frame_lengths),
        "cache_frame_lengths": list(frame_lengths),
        "cache_wire_transfers": cache_wire_transfers,
        "cache_reuse_copies": cache_reuse_copies,
        "warm_prefix_reused": reusable_prefix is not None,
        "chunk_bytes": int(chunk_bytes),
        "transfer_seconds": transfer_seconds,
        "handoff_telemetry": handoff_telemetry,
        "import_seconds": import_seconds,
        "decode_seconds": decode_seconds,
        "drafter_telemetry": drafter_telemetry,
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
            mismatches = [
                _first_token_difference(local_reference["tokens"], item["tokens"])
                for item in generated
            ]
            first_mismatch = min(index for index in mismatches if index is not None)
            remote_hashes = sorted({item["token_sha256"] for item in generated})
            output["first_local_token_difference"] = first_mismatch
            output["local_token_sha256"] = local_reference["token_sha256"]
            output["remote_token_sha256"] = remote_hashes
            if strict_local_match:
                raise HandoffError(
                    "Spark-prefilled and Mac-prefilled replies differ at generated token "
                    f"{first_mismatch}; local_sha256={local_reference['token_sha256']}; "
                    f"remote_sha256={','.join(remote_hashes)}"
                )
    return output


def _run_workload(args: argparse.Namespace, model_dir: Path) -> dict[str, Any]:
    progressive_handoff = bool(getattr(args, "progressive_handoff", False))
    diagnose_cache_state = bool(getattr(args, "diagnose_cache_state", False))
    runtime, tokenizer = _load_runtime(
        model_dir,
        mtp_drafts=args.mtp_drafts,
        drafter=args.drafter,
        drafter_bits=args.drafter_bits,
    )
    prompt_ids = render_user_prompt(tokenizer, args.prompt)
    if len(prompt_ids) < 2:
        raise HandoffError("the rendered prompt must contain at least two tokens")

    local = None
    if args.compare_local or diagnose_cache_state:
        local = run_decode(
            runtime, tokenizer, prompt_ids,
            max_new_tokens=args.max_new_tokens,
            stream_id="local-reference",
            force_length=args.force_length,
            copy_drafts=args.copy_drafts,
            copy_min_match=args.copy_min_match,
        )

    if diagnose_cache_state:
        if local is None:
            raise HandoffError("cache-state diagnostic needs a Mac-local reference")
        return run_cache_state_diagnostic(
            runtime,
            prompt_ids,
            local,
            mailbox_name=args.mailbox,
            timeout=args.timeout,
            model_id=args.model_id,
            model_revision=args.model_revision,
            chunk_bytes=args.chunk_mib * 1024 * 1024,
            progressive_handoff=progressive_handoff,
            diagnose_kda_layer=getattr(args, "diagnose_kda_layer", None),
        )

    warmup_new_tokens = args.warmup_new_tokens or args.max_new_tokens
    reusable_prefix = None
    prefix_setup = None
    if args.reuse_warm_prefix:
        setup_started = time.perf_counter()
        reusable_prefix = prepare_reusable_prefix(
            runtime,
            prompt_ids,
            mailbox_name=args.mailbox,
            timeout=args.timeout,
            model_id=args.model_id,
            model_revision=args.model_revision,
            chunk_bytes=args.chunk_mib * 1024 * 1024,
            progressive_handoff=progressive_handoff,
        )
        prefix_setup = {
            "seconds": time.perf_counter() - setup_started,
            "transfer_seconds": reusable_prefix.transfer_seconds,
            "import_seconds": reusable_prefix.import_seconds,
            "cache_bytes": sum(item.nbytes for item in reusable_prefix.manifest.tensors),
            "cache_frame_count": len(reusable_prefix.frame_lengths),
            "cache_wire_transfers": 1,
            "transfer_id": reusable_prefix.manifest.transfer_id,
            "cache_protocol": reusable_prefix.manifest.protocol,
            "handoff_telemetry": reusable_prefix.handoff_telemetry,
        }
    try:
        warmups = [
            run_handoff_round(
                runtime,
                tokenizer,
                prompt_ids,
                parallel=args.parallel,
                max_new_tokens=warmup_new_tokens,
                mailbox_name=args.mailbox,
                timeout=args.timeout,
                model_id=args.model_id,
                model_revision=args.model_revision,
                local_reference=None,
                chunk_bytes=args.chunk_mib * 1024 * 1024,
                force_length=False,
                strict_local_match=not args.allow_local_drift,
                reusable_prefix=reusable_prefix,
                copy_drafts=args.copy_drafts,
                copy_min_match=args.copy_min_match,
                progressive_handoff=progressive_handoff,
            )
            for _ in range(args.warmups)
        ]
        measured = [
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
                force_length=args.force_length,
                strict_local_match=not args.allow_local_drift,
                reusable_prefix=reusable_prefix,
                copy_drafts=args.copy_drafts,
                copy_min_match=args.copy_min_match,
                progressive_handoff=progressive_handoff,
            )
            for _ in range(args.repeats)
        ]
    finally:
        if reusable_prefix is not None:
            reusable_prefix.cache.clear()
            reusable_prefix.final_hidden = None
            _reclaim_mlx_memory()
    if args.warmups == 0 and args.repeats == 1:
        return measured[0]
    if args.force_length and any(
        item["completion_tokens"] != args.parallel * args.max_new_tokens
        for item in measured
    ):
        raise HandoffError("a fixed-length measured round ended before the requested token count")
    measured_hashes = {
        generated["token_sha256"]
        for item in measured
        for generated in (
            item["batch"]["results"] if item.get("batch") is not None else [item["result"]]
        )
    }
    if len(measured_hashes) != 1:
        raise HandoffError("greedy measured rounds returned different output tokens")
    rates = [float(item["aggregate_pipeline_tokens_per_second"]) for item in measured]
    complete_times = [
        float(item["result"]["completion_from_origin_seconds"])
        for item in measured
    ]
    ttfts = [
        float(item["result"]["first_token_from_origin_seconds"])
        for item in measured
    ]
    decode_rates = [
        float(
            item["result"]["decode_tokens_per_second"]
            if "decode_tokens_per_second" in item["result"]
            else item["result"]["subsequent_tokens_per_second"]
        )
        for item in measured
    ]

    def summary(values: list[float]) -> dict[str, float]:
        return {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "event": "glm_mcdma_benchmark",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "prompt_tokens": len(prompt_ids),
        "parallel": args.parallel,
        "max_new_tokens": args.max_new_tokens,
        "force_length": bool(args.force_length),
        "copy_drafts": bool(args.copy_drafts),
        "copy_min_match": int(args.copy_min_match),
        "allow_local_drift": bool(args.allow_local_drift),
        "reuse_warm_prefix": bool(args.reuse_warm_prefix),
        "mtp_drafts": args.mtp_drafts,
        "drafter": args.drafter or None,
        "drafter_bits": args.drafter_bits if args.drafter else None,
        "chunk_mib": args.chunk_mib,
        "progressive_handoff": progressive_handoff,
        "warmup_count": args.warmups,
        "warmup_new_tokens": warmup_new_tokens,
        "repeat_count": args.repeats,
        "output_token_sha256": next(iter(measured_hashes)),
        "complete_request_seconds": summary(complete_times),
        "ttft_seconds": summary(ttfts),
        "decode_tokens_per_second": summary(decode_rates),
        "aggregate_pipeline_tokens_per_second": summary(rates),
        "local_reference": local,
        "prefix_setup": prefix_setup,
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
    parser.add_argument(
        "--diagnose-cache-state",
        action="store_true",
        help=(
            "compare one real Spark-prefilled cache with a Mac-prefilled cache, "
            "then stop at the first teacher-forced prediction split"
        ),
    )
    parser.add_argument(
        "--diagnose-kda-layer",
        type=int,
        help=(
            "diagnostic only: compare CUDA and Metal prompt intermediates for "
            "one KDA layer; requires --diagnose-cache-state"
        ),
    )
    parser.add_argument(
        "--allow-local-drift",
        action="store_true",
        help="record cross-backend token drift instead of stopping the timing run",
    )
    parser.add_argument("--force-length", action="store_true")
    parser.add_argument(
        "--copy-drafts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable TensorFold suffix-copy drafts during Mac decode",
    )
    parser.add_argument("--copy-min-match", type=int, default=4)
    parser.add_argument("--reuse-warm-prefix", action="store_true")
    parser.add_argument(
        "--progressive-handoff",
        action="store_true",
        help="stream final-chunk Spark cache state during prefill using protocol v4",
    )
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--warmup-new-tokens", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--mtp-drafts", type=int, default=3)
    parser.add_argument(
        "--drafter",
        default="",
        help="optional local DFlash2 helper directory; replaces MTP drafting",
    )
    parser.add_argument("--drafter-bits", type=int, choices=(0, 4, 8), default=4)
    parser.add_argument("--chunk-mib", type=int, default=DEFAULT_FRAME_BYTES // (1024 * 1024))
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.allow_local_drift and not args.compare_local:
        parser.error("--allow-local-drift requires --compare-local")
    if args.diagnose_kda_layer is not None:
        if not args.diagnose_cache_state:
            parser.error("--diagnose-kda-layer requires --diagnose-cache-state")
        if not 0 <= args.diagnose_kda_layer < 45:
            parser.error("--diagnose-kda-layer must be between 0 and 44")
        if args.progressive_handoff:
            parser.error("--diagnose-kda-layer requires the sealed non-progressive handoff")
    if not 1 <= args.parallel <= 8:
        parser.error("--parallel must be between 1 and 8")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be nonnegative and --repeats must be positive")
    if args.warmup_new_tokens is not None and args.warmup_new_tokens <= 0:
        parser.error("--warmup-new-tokens must be positive")
    if not 1 <= args.mtp_drafts <= 8:
        parser.error("--mtp-drafts must be between 1 and 8")
    if args.copy_min_match < 1:
        parser.error("--copy-min-match must be positive")
    if not 1 <= args.chunk_mib <= MAX_FRAME_BYTES // (1024 * 1024):
        parser.error(f"--chunk-mib must be between 1 and {MAX_FRAME_BYTES // (1024 * 1024)}")
    if args.diagnose_cache_state:
        if args.parallel != 1:
            parser.error("--diagnose-cache-state requires --parallel 1")
        if args.max_new_tokens < 2:
            parser.error("--diagnose-cache-state requires at least two generated tokens")
        if args.warmups != 0 or args.repeats != 1 or args.reuse_warm_prefix:
            parser.error(
                "--diagnose-cache-state requires --warmups 0, --repeats 1, "
                "and no --reuse-warm-prefix"
            )
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
