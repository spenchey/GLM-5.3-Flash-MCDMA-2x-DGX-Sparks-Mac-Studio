"""Two-Spark GLM prefill service for verified cache handoff to the Mac.

Both CUDA ranks run the complete portable 4-bit checkpoint.  Rank zero merges
head-sharded KDA state, verifies that latent attention state is identical on
both ranks, and exposes the resulting archive through the existing MCDMA
mailbox.  Decoding is intentionally absent from this process.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, field
import json
from pathlib import Path
import queue
import sys
import threading
import time
from types import ModuleType
from typing import Any, Sequence

import numpy as np

from experiments.three_machine.cache_handoff import (
    HandoffError,
    Request,
    TensorArchive,
    decode_request,
    merge_kda_conv,
    pack_ack,
    pack_error,
    pack_manifest,
    persist_archive,
    prompt_digest,
    reply_frame,
    require_replicated,
)
from experiments.three_machine.mailbox import ServiceMailbox
from experiments.three_machine.kda_intermediates import (
    KDA_INTERMEDIATE_STAGES,
    kda_intermediate_names,
    merge_head_shards,
    merge_kda_projection,
)
from experiments.three_machine.progressive_handoff import (
    ProgressiveJob,
    ProgressivePlan,
    ProgressiveRequest,
    ProgressiveTensorSpec,
    decode_progressive_request,
    is_progressive_request,
    pack_plan,
    pack_progressive_ack,
    pack_seal,
    reply_progressive_frame,
)


ACTION_PREPARE = 1
ACTION_SHUTDOWN = 2
ACTION_PROGRESSIVE_PREPARE = 3
CONTROL_WORDS = 3
DEFAULT_MODEL_ID = "Vontra/GLM-5.3-Flash-MLX-4bit-MTP"
DEFAULT_MODEL_REVISION = "76add2a341a1cd90ad0e86bb69839ea9c35827c6"


@dataclass
class _TimedPrefillChunk:
    rows: int
    start: Any
    end: Any | None = None
    layers: list[tuple[int, str, Any]] = field(default_factory=list)


class CudaLayerTimeline:
    """Record CUDA completion events without synchronizing between layers.

    This diagnostic wraps TensorFold only while one prompt prefill runs.  CUDA
    events stay on the existing stream, so the report shows when each layer's
    queued work actually completed without turning the layer loop into 45
    host/device round trips.
    """

    def __init__(self, torch_module: Any, decode_module: ModuleType, forward_module: ModuleType) -> None:
        self.torch = torch_module
        self.decode = decode_module
        self.forward = forward_module
        self._original_compute = decode_module.compute
        self._original_layer_forward = forward_module.layer_forward
        self._active: _TimedPrefillChunk | None = None
        self._chunks: list[_TimedPrefillChunk] = []

    def _event(self):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def __enter__(self) -> "CudaLayerTimeline":
        def timed_layer_forward(layer, *args, **kwargs):
            result = self._original_layer_forward(layer, *args, **kwargs)
            if self._active is not None:
                self._active.layers.append((int(layer.index), str(layer.kind), self._event()))
            return result

        def timed_compute(w, st, b, R, *args, **kwargs):
            if not getattr(b, "prefill", False):
                return self._original_compute(w, st, b, R, *args, **kwargs)
            chunk = _TimedPrefillChunk(rows=int(R), start=self._event())
            self._chunks.append(chunk)
            prior = self._active
            self._active = chunk
            try:
                return self._original_compute(w, st, b, R, *args, **kwargs)
            finally:
                chunk.end = self._event()
                self._active = prior

        self.forward.layer_forward = timed_layer_forward
        self.decode.compute = timed_compute
        return self

    def __exit__(self, *_args) -> None:
        self.decode.compute = self._original_compute
        self.forward.layer_forward = self._original_layer_forward

    def report(self, cache_bytes: dict[int, int] | None = None) -> list[dict[str, Any]]:
        """Resolve the recorded events after prefill and return JSON-safe timings."""

        self.torch.cuda.synchronize()
        payload = {} if cache_bytes is None else dict(cache_bytes)
        chunks: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(self._chunks):
            if chunk.end is None:
                raise RuntimeError("CUDA layer timeline ended before its compute event")
            previous = chunk.start
            layers = []
            for layer_index, kind, event in chunk.layers:
                layers.append({
                    "index": layer_index,
                    "kind": kind,
                    "ready_seconds": chunk.start.elapsed_time(event) / 1000.0,
                    "layer_seconds": previous.elapsed_time(event) / 1000.0,
                    "cache_bytes": int(payload.get(layer_index, 0)),
                })
                previous = event
            chunks.append({
                "chunk": chunk_index,
                "rows": chunk.rows,
                "compute_seconds": chunk.start.elapsed_time(chunk.end) / 1000.0,
                "post_layers_seconds": previous.elapsed_time(chunk.end) / 1000.0,
                "layers": layers,
            })
        return chunks


class CudaKDAIntermediateTrace:
    """Capture one CUDA KDA prompt layer without changing its computation."""

    def __init__(self, forward_module: ModuleType, layer_index: int) -> None:
        self.forward = forward_module
        self.layer_index = int(layer_index)
        self._original = forward_module.kda_block
        self._captures: list[dict[str, Any]] = []
        self.local_heads: int | None = None

    def __enter__(self) -> "CudaKDAIntermediateTrace":
        def traced(layer, w, state, buffers, rows, cut=None):
            result = self._original(layer, w, state, buffers, rows, cut)
            if int(layer.index) != self.layer_index:
                return result
            if not getattr(buffers, "prefill", False):
                raise HandoffError("KDA intermediate tracing observed a decode call")
            if cut is not None:
                raise HandoffError("KDA intermediate tracing does not support a split prompt chunk")
            cache_index = state.kda_index[int(layer.index)]
            current = state.rec[int(state.cur[cache_index]), cache_index]
            projection = buffers.kproj[0, :rows]
            scratch = buffers.kscratch
            self.local_heads = int(layer.kda.heads)
            self._captures.append({
                "input": buffers.normed[:rows].detach().clone(),
                "projection": projection.detach().clone(),
                "forget_projection": buffers.ka[:rows].detach().clone(),
                "key": scratch.k[:rows].detach().clone(),
                "value": scratch.v[:rows].detach().clone(),
                "decay": scratch.g[:rows].detach().clone(),
                "beta": scratch.b[:rows].detach().clone(),
                "output_gate_projection": buffers.kg[:rows].detach().clone(),
                "output": scratch.out[:rows].detach().clone(),
                "state": current.detach().clone(),
            })
            return result

        self.forward.kda_block = traced
        return self

    def __exit__(self, *_args) -> None:
        self.forward.kda_block = self._original

    def captured(self) -> tuple[dict[str, Any], int]:
        if len(self._captures) != 1 or self.local_heads is None:
            raise HandoffError(
                "KDA intermediate tracing requires exactly one target-layer prompt chunk"
            )
        capture = self._captures[0]
        missing = sorted(set(KDA_INTERMEDIATE_STAGES) - set(capture))
        if missing:
            raise HandoffError(f"KDA intermediate capture is incomplete: {missing}")
        return capture, self.local_heads


@dataclass
class _ProgressiveCopy:
    name: str
    mode: str
    logical_dtype: str
    send: Any
    gathered: Any
    host: Any | None
    done: Any


class CudaProgressiveCollector:
    """Queue final-chunk layer state on a CUDA stream and publish in order."""

    def __init__(
        self,
        exporter: "CudaPrefillExporter",
        plan: ProgressivePlan,
        job: ProgressiveJob | None,
    ) -> None:
        self.exporter = exporter
        self.plan = plan
        self.job = job
        self.torch = exporter.torch
        self.comm = exporter.comm
        self.rank = exporter.rank
        self.cached_tokens = plan.cached_tokens
        self.mtp_cached_tokens = plan.mtp_cached_tokens
        self.stream = self.torch.cuda.Stream()
        self._original_compute = None
        self._original_layer_forward = None
        self._active = False
        self._final_chunks = 0
        self._seen_layers: list[int] = []
        self._rank_tasks: list[_ProgressiveCopy] = []
        self._queue: queue.Queue[_ProgressiveCopy | None] = queue.Queue()
        self._failure: Exception | None = None
        self._publisher: threading.Thread | None = None
        self._finished = False
        if self.rank == 0:
            if job is None:
                raise HandoffError("rank zero progressive collection needs a live job")
            self._publisher = threading.Thread(
                target=self._publish,
                name=f"glm-progressive-{plan.job_id[:8]}",
                daemon=False,
            )
            self._publisher.start()

    def _dtype(self, value: Any) -> tuple[str, Any]:
        if value.dtype == self.torch.bfloat16:
            return "bfloat16", self.torch.uint16
        if value.dtype == self.torch.float32:
            return np.dtype(np.float32).str, self.torch.float32
        raise HandoffError(f"progressive cache tensor has unsupported CUDA dtype {value.dtype}")

    def _enqueue(self, name: str, value: Any, mode: str) -> None:
        spec = self.plan.tensor(name)
        logical_dtype, host_dtype = self._dtype(value)
        if spec.dtype != logical_dtype:
            raise HandoffError(f"progressive plan dtype changed for {name!r}")
        send = value.detach().contiguous()
        gathered = self.torch.empty(
            (self.comm.world,) + tuple(send.shape),
            dtype=send.dtype,
            device=send.device,
        )
        ready = self.torch.cuda.Event()
        ready.record()
        host = None
        with self.torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            self.comm.all_gather(send, gathered)
            if self.rank == 0:
                host = self.torch.empty(
                    tuple(gathered.shape),
                    dtype=host_dtype,
                    device="cpu",
                    pin_memory=True,
                )
                source = gathered.view(self.torch.uint16) if logical_dtype == "bfloat16" else gathered
                host.copy_(source, non_blocking=True)
            done = self.torch.cuda.Event()
            done.record()
        task = _ProgressiveCopy(name, mode, logical_dtype, send, gathered, host, done)
        if self.rank == 0:
            self._queue.put(task)
        else:
            self._rank_tasks.append(task)

    def _enqueue_layer(self, layer: Any, state: Any) -> None:
        layer_index = int(layer.index)
        if self._seen_layers and layer_index <= self._seen_layers[-1]:
            raise HandoffError("progressive layer callbacks are not in execution order")
        self._seen_layers.append(layer_index)
        if layer.kind == "kda":
            cache_index = state.kda_index[layer_index]
            self._enqueue(f"layers.{layer_index}.conv", state.conv[cache_index], "conv")
            current = state.rec[int(state.cur[cache_index]), cache_index]
            self._enqueue(f"layers.{layer_index}.ssm", current, "concat")
        elif layer.kind == "dsa":
            cache_index = state.dsa_index[layer_index]
            self._enqueue(
                f"layers.{layer_index}.keys",
                state.kc[cache_index][:self.cached_tokens],
                "replicated",
            )
        else:
            raise HandoffError(f"progressive handoff does not know GLM layer kind {layer.kind!r}")

    def __enter__(self) -> "CudaProgressiveCollector":
        from tensorfold.families.glm5_next.cuda import decode as decode_module
        from tensorfold.families.glm5_next.cuda import forward as forward_module

        self._original_compute = decode_module.compute
        self._original_layer_forward = forward_module.layer_forward

        def progressive_layer_forward(layer, w, state, buffers, rows, *args, **kwargs):
            result = self._original_layer_forward(layer, w, state, buffers, rows, *args, **kwargs)
            if self._active:
                self._enqueue_layer(layer, state)
            return result

        def progressive_compute(w, state, buffers, rows, *args, **kwargs):
            final = bool(
                getattr(buffers, "prefill", False)
                and int(state.pos) + int(rows) == self.cached_tokens
            )
            prior = self._active
            self._active = final
            if final:
                self._final_chunks += 1
            try:
                return self._original_compute(w, state, buffers, rows, *args, **kwargs)
            finally:
                self._active = prior

        forward_module.layer_forward = progressive_layer_forward
        decode_module.compute = progressive_compute
        return self

    def __exit__(self, *_args) -> None:
        from tensorfold.families.glm5_next.cuda import decode as decode_module
        from tensorfold.families.glm5_next.cuda import forward as forward_module

        if self._original_compute is not None:
            decode_module.compute = self._original_compute
        if self._original_layer_forward is not None:
            forward_module.layer_forward = self._original_layer_forward

    def _publish(self) -> None:
        assert self.job is not None
        while True:
            task = self._queue.get()
            if task is None:
                return
            if self._failure is not None:
                continue
            try:
                task.done.synchronize()
                if task.host is None:
                    raise HandoffError("rank-zero progressive task has no host result")
                gathered = task.host.numpy()
                ranks = [
                    np.ascontiguousarray(gathered[rank])
                    for rank in range(self.comm.world)
                ]
                if task.mode == "conv":
                    value = merge_kda_conv(ranks)
                elif task.mode == "concat":
                    value = np.ascontiguousarray(np.concatenate(ranks, axis=0))
                elif task.mode == "replicated":
                    value = require_replicated(ranks, name=task.name)
                else:
                    raise HandoffError(f"unknown progressive merge mode {task.mode!r}")
                spec = self.plan.tensor(task.name)
                if tuple(value.shape) != spec.shape or memoryview(value).cast("B").nbytes != spec.nbytes:
                    raise HandoffError(f"progressive CUDA geometry changed for {task.name!r}")
                self.job.publish_piece(task.name, 0, memoryview(value).cast("B"))
            except Exception as exc:
                self._failure = exc
                self.job.fail(f"progressive CUDA export failed: {exc}")

    def finish(self, final_hidden: Any, mtp_keys: Any, *, timeout: float) -> None:
        if self._finished:
            raise HandoffError("progressive collector was already finished")
        self._finished = True
        expected_layers = [int(layer.index) for layer in self.exporter.w.layers]
        if self._final_chunks != 1 or self._seen_layers != expected_layers:
            raise HandoffError("progressive collection did not observe one complete final prompt chunk")
        self._enqueue("prompt.final_hidden", final_hidden, "replicated")
        self._enqueue("mtp.keys", mtp_keys, "replicated")
        if self.rank == 0:
            self._queue.put(None)
            assert self._publisher is not None
            self._publisher.join(timeout)
            if self._publisher.is_alive():
                raise HandoffError("progressive CUDA publisher did not stop before its deadline")
        else:
            self.stream.synchronize()
        self._rank_tasks.clear()
        if self._failure is not None:
            raise HandoffError(f"progressive CUDA export failed: {self._failure}")

    def abort(self, reason: str, *, timeout: float) -> None:
        if self.rank == 0 and self._publisher is not None and self._publisher.is_alive():
            self._queue.put(None)
            self._publisher.join(timeout)
        try:
            self.stream.synchronize()
        finally:
            self._rank_tasks.clear()
            if self.job is not None:
                self.job.fail(reason or "progressive CUDA export aborted")


def _cache_bytes_by_layer(arrays: dict[str, np.ndarray]) -> dict[int, int]:
    """Return the immutable handoff payload owned by each model layer."""

    totals: dict[int, int] = {}
    for name, value in arrays.items():
        parts = name.split(".", 2)
        if len(parts) != 3 or parts[0] != "layers" or not parts[1].isdigit():
            continue
        layer_index = int(parts[1])
        totals[layer_index] = totals.get(layer_index, 0) + int(value.nbytes)
    return totals


def _control(comm, rank: int, values: tuple[int, int, int] | None = None) -> tuple[int, int, int]:
    import torch

    send = torch.zeros((CONTROL_WORDS,), dtype=torch.int64, device="cuda")
    if rank == 0:
        if values is None:
            raise ValueError("rank zero must supply control values")
        send.copy_(torch.tensor(values, dtype=torch.int64, device="cuda"))
    recv = torch.empty((comm.world * CONTROL_WORDS,), dtype=torch.int64, device="cuda")
    comm.all_gather(send, recv)
    torch.cuda.synchronize()
    return tuple(int(value) for value in recv[:CONTROL_WORDS].cpu().tolist())


def _token_ids(comm, rank: int, tokens: Sequence[int] | None, count: int) -> list[int]:
    import torch

    if rank == 0:
        if tokens is None or len(tokens) != count:
            raise HandoffError("rank-zero prompt length differs from its control word")
        send = torch.tensor(list(tokens), dtype=torch.int32, device="cuda")
    else:
        send = torch.zeros((count,), dtype=torch.int32, device="cuda")
    recv = torch.empty((comm.world * count,), dtype=torch.int32, device="cuda")
    comm.all_gather(send, recv)
    torch.cuda.synchronize()
    return [int(value) for value in recv[:count].cpu().tolist()]


class CudaPrefillExporter:
    """Own the full CUDA model and export a Mac-compatible prefix cache."""

    def __init__(
        self,
        model_dir: str,
        *,
        rank: int,
        master: str,
        port: int,
        model_id: str,
        model_revision: str,
        prefill_rows: int = 128,
        archive_dir: str | None = None,
        layer_timeline: bool = False,
        kda_trace_layer: int | None = None,
    ) -> None:
        import torch

        from tensorfold.cuda.comm import NCCL
        from tensorfold.families.glm5_next.cuda.decode import Engine
        from tensorfold.families.glm5_next.cuda.weights import load

        torch.cuda.set_device(0)
        self.torch = torch
        self.rank = int(rank)
        self.model_id = model_id
        self.model_revision = model_revision
        self.archive_dir = None if archive_dir is None else Path(archive_dir)
        self.layer_timeline = bool(layer_timeline)
        self.kda_trace_layer = (
            None if kda_trace_layer is None else int(kda_trace_layer)
        )
        self.comm = NCCL(self.rank, 2, master, port)
        self.comm.barrier()
        self.w = load(model_dir, rank=self.rank, mtp=True)
        self.w.comm = self.comm
        self.comm.ready("cache-handoff-weights")
        self.comm.barrier()
        if len(self.w.layers) != 45 or self.w.mtp is None:
            raise RuntimeError("the pinned portable GLM checkpoint did not load 45 layers and its MTP head")
        if self.kda_trace_layer is not None:
            matches = [
                layer for layer in self.w.layers
                if int(layer.index) == self.kda_trace_layer
            ]
            if len(matches) != 1 or matches[0].kind != "kda":
                raise RuntimeError(
                    f"layer {self.kda_trace_layer} is not one KDA layer in the loaded checkpoint"
                )
        self.capacity = int(self.w.cfg.dense_limit)
        # Handoff v2 transfers latent attention keys, but not the sparse
        # indexer keys/gates/pool.  Stay on the exact dense path until those
        # tensors are part of the protocol.
        self.handoff_limit = min(self.capacity, int(self.w.cfg.index_topk))
        self.engine = Engine(
            self.w,
            capacity=self.capacity,
            # This process never decodes or batches reply rows.  One row keeps
            # the Spark memory budget for the checkpoint and prefill buffers.
            max_rows=1,
            prefill_rows=int(prefill_rows),
            graphs=False,
            long_context=False,
        )
        self.comm.ready("cache-handoff-buffers")

    def _gather(self, value):
        send = value.detach().contiguous()
        recv = self.torch.empty((self.comm.world,) + tuple(send.shape), dtype=send.dtype, device=send.device)
        self.comm.all_gather(send, recv)
        self.torch.cuda.synchronize()
        return recv

    def _rank_arrays(self, gathered) -> tuple[list[np.ndarray], str | None]:
        if self.rank != 0:
            return [], None
        value = gathered.detach().cpu()
        logical = None
        if value.dtype == self.torch.bfloat16:
            value = value.view(self.torch.uint16)
            logical = "bfloat16"
        return [np.ascontiguousarray(value[rank].numpy()) for rank in range(self.comm.world)], logical

    def _collect_kda_trace(
        self,
        trace: CudaKDAIntermediateTrace,
    ) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        capture, local_heads = trace.captured()
        names = kda_intermediate_names(trace.layer_index)
        arrays: dict[str, np.ndarray] = {}
        dtypes: dict[str, str] = {}
        for stage in KDA_INTERMEDIATE_STAGES:
            rank_values, logical_dtype = self._rank_arrays(
                self._gather(capture[stage])
            )
            if self.rank != 0:
                continue
            name = names[stage]
            if stage == "input":
                value = require_replicated(rank_values, name=name)
            elif stage == "projection":
                value = merge_kda_projection(
                    rank_values,
                    local_heads=local_heads,
                )
            elif stage in {"forget_projection", "output_gate_projection", "output"}:
                value = merge_head_shards(rank_values, axis=-1, name=name)
            elif stage in {"key", "value", "decay", "beta"}:
                value = merge_head_shards(rank_values, axis=1, name=name)
            elif stage == "state":
                value = merge_head_shards(rank_values, axis=0, name=name)
            else:  # protected by KDA_INTERMEDIATE_STAGES
                raise HandoffError(f"unknown KDA intermediate stage {stage!r}")
            arrays[name] = value
            if logical_dtype is not None:
                dtypes[name] = logical_dtype
        return arrays, dtypes

    def _progressive_spec(
        self,
        ordinal: int,
        name: str,
        shape: Sequence[int],
        dtype: Any,
    ) -> ProgressiveTensorSpec:
        if dtype == self.torch.bfloat16:
            logical_dtype, itemsize = "bfloat16", 2
        elif dtype == self.torch.float32:
            logical_dtype, itemsize = np.dtype(np.float32).str, 4
        else:
            raise HandoffError(f"progressive cache tensor {name!r} has unsupported dtype {dtype}")
        geometry = tuple(int(size) for size in shape)
        return ProgressiveTensorSpec(
            ordinal=int(ordinal),
            name=name,
            dtype=logical_dtype,
            shape=geometry,
            nbytes=int(np.prod(geometry, dtype=np.int64)) * itemsize,
        )

    def progressive_plan(
        self,
        prompt: Sequence[int],
        *,
        job_id: str,
        generation: int,
    ) -> ProgressivePlan:
        """Describe final cache geometry without waiting for prefill contents."""

        tokens = [int(token) for token in prompt]
        if len(tokens) < 2:
            raise HandoffError("cache handoff needs at least two prompt tokens")
        if len(tokens) > self.handoff_limit:
            raise HandoffError(
                f"prompt has {len(tokens)} tokens; progressive cache handoff supports at most "
                f"{self.handoff_limit} before sparse index state is required"
            )
        state = self.engine.st
        specs: list[ProgressiveTensorSpec] = []

        def add(name: str, shape: Sequence[int], dtype: Any) -> None:
            specs.append(self._progressive_spec(len(specs), name, shape, dtype))

        for layer in self.w.layers:
            layer_index = int(layer.index)
            if layer.kind == "kda":
                cache_index = state.kda_index[layer_index]
                conv = state.conv[cache_index]
                conv_shape = list(conv.shape)
                conv_shape[-1] *= self.comm.world
                add(f"layers.{layer_index}.conv", conv_shape, conv.dtype)
                recurrent = state.rec[0, cache_index]
                recurrent_shape = list(recurrent.shape)
                recurrent_shape[0] *= self.comm.world
                add(f"layers.{layer_index}.ssm", recurrent_shape, recurrent.dtype)
            elif layer.kind == "dsa":
                cache_index = state.dsa_index[layer_index]
                keys = state.kc[cache_index]
                add(
                    f"layers.{layer_index}.keys",
                    (len(tokens), *tuple(keys.shape[1:])),
                    keys.dtype,
                )
            else:
                raise HandoffError(f"progressive handoff does not know GLM layer kind {layer.kind!r}")
        hidden = self.engine.pbuf.fnormed
        add("prompt.final_hidden", (1, *tuple(hidden.shape[1:])), hidden.dtype)
        add(
            "mtp.keys",
            (len(tokens) - 1, *tuple(state.mtp_kc.shape[1:])),
            state.mtp_kc.dtype,
        )
        return ProgressivePlan.build(
            job_id=job_id,
            generation=generation,
            model_id=self.model_id,
            model_revision=self.model_revision,
            prompt_sha256=prompt_digest(tokens),
            cached_tokens=len(tokens),
            mtp_cached_tokens=len(tokens) - 1,
            tensors=specs,
        )

    def prepare_progressive(
        self,
        prompt: Sequence[int],
        job: ProgressiveJob | None,
        *,
        timeout: float,
    ) -> Any:
        """Prefill while final-chunk cache tensors stream into ``job``."""

        if self.kda_trace_layer is not None:
            raise HandoffError(
                "KDA intermediate tracing requires the sealed non-progressive diagnostic path"
            )

        from tensorfold.families.glm5_next.cuda import decode as decode_module

        tokens = [int(token) for token in prompt]
        plan = (
            self.progressive_plan(tokens, job_id="00" * 16, generation=1)
            if job is None
            else job.plan
        )
        expected = self.progressive_plan(
            tokens,
            job_id=plan.job_id,
            generation=plan.generation,
        )
        if plan != expected:
            raise HandoffError("progressive job geometry differs from the loaded CUDA model")
        self.engine.reset()
        collector = CudaProgressiveCollector(self, plan, job)
        started = time.perf_counter()
        try:
            with collector:
                spark_first_token = int(
                    decode_module.prefill(self.engine, tokens, None, mtp=True)
                )
            state = self.engine.st
            if self.engine.last_hidden is None:
                raise RuntimeError("CUDA prefill did not retain its final hidden row")
            if (
                state.pos != plan.cached_tokens
                or state.mtp_len != plan.mtp_cached_tokens
                or state.mtp_drafted
            ):
                raise RuntimeError(
                    f"CUDA cache stopped at main={state.pos}, mtp={state.mtp_len}, "
                    f"drafted={state.mtp_drafted}; expected {plan.cached_tokens}, "
                    f"{plan.mtp_cached_tokens}, 0"
                )
            collector.finish(
                self.engine.last_hidden,
                state.mtp_kc[:plan.mtp_cached_tokens],
                timeout=timeout,
            )
            seal = None if job is None else job.seal(
                spark_first_token=spark_first_token,
                generation=plan.generation,
            )
        except Exception as exc:
            collector.abort(str(exc), timeout=timeout)
            raise
        print(json.dumps({
            "event": "glm_progressive_prefill",
            "rank": self.rank,
            "cached_tokens": plan.cached_tokens,
            "mtp_cached_tokens": plan.mtp_cached_tokens,
            "spark_first_token": spark_first_token,
            "cache_bytes": plan.total_nbytes,
            "job_id": plan.job_id if self.rank == 0 else None,
            "plan_id": plan.plan_id if self.rank == 0 else None,
            "transfer_id": None if seal is None else seal.transfer_id,
            "seconds": time.perf_counter() - started,
        }, sort_keys=True), flush=True)
        return seal

    def prepare(self, prompt: Sequence[int]) -> TensorArchive | None:
        """Prefill the full prompt and export the state before reply decode."""

        from tensorfold.families.glm5_next.cuda import decode as decode_module
        from tensorfold.families.glm5_next.cuda import forward as forward_module

        prefill = decode_module.prefill

        tokens = [int(token) for token in prompt]
        if len(tokens) < 2:
            raise HandoffError("cache handoff needs at least two prompt tokens")
        cached_tokens = len(tokens)
        mtp_cached_tokens = cached_tokens - 1
        if len(tokens) > self.handoff_limit:
            raise HandoffError(
                f"prompt has {len(tokens)} tokens; cache handoff v2 supports at most "
                f"{self.handoff_limit} before sparse index state is required"
            )
        # A released transfer must not contaminate the next independent
        # request.  Both ranks reset before entering any collective work.
        reset_started = time.perf_counter()
        self.engine.reset()
        reset_seconds = time.perf_counter() - reset_started
        started = time.perf_counter()
        timeline = None
        kda_trace = None
        with ExitStack() as stack:
            if self.layer_timeline:
                timeline = stack.enter_context(
                    CudaLayerTimeline(self.torch, decode_module, forward_module)
                )
            if self.kda_trace_layer is not None:
                kda_trace = stack.enter_context(
                    CudaKDAIntermediateTrace(
                        forward_module,
                        self.kda_trace_layer,
                    )
                )
            spark_first_token = int(prefill(self.engine, tokens, None, mtp=True))
        prefill_seconds = time.perf_counter() - started
        if self.engine.last_hidden is None:
            raise RuntimeError("CUDA prefill did not retain its final hidden row")
        state = self.engine.st
        if (
            state.pos != cached_tokens
            or state.mtp_len != mtp_cached_tokens
            or state.mtp_drafted
        ):
            raise RuntimeError(
                f"CUDA cache stopped at main={state.pos}, mtp={state.mtp_len}, drafted={state.mtp_drafted}; "
                f"expected {cached_tokens}, {mtp_cached_tokens}, 0"
            )

        arrays: dict[str, np.ndarray] = {}
        dtypes: dict[str, str] = {}
        diagnostic_started = time.perf_counter()
        if kda_trace is not None:
            diagnostic_arrays, diagnostic_dtypes = self._collect_kda_trace(kda_trace)
            arrays.update(diagnostic_arrays)
            dtypes.update(diagnostic_dtypes)
        diagnostic_collect_seconds = time.perf_counter() - diagnostic_started
        kda_started = time.perf_counter()
        for layer_index, cache_index in sorted(state.kda_index.items()):
            conv_ranks, conv_dtype = self._rank_arrays(self._gather(state.conv[cache_index]))
            current = state.rec[int(state.cur[cache_index]), cache_index]
            rec_ranks, _ = self._rank_arrays(self._gather(current))
            if self.rank == 0:
                conv_name = f"layers.{layer_index}.conv"
                arrays[conv_name] = merge_kda_conv(conv_ranks)
                arrays[f"layers.{layer_index}.ssm"] = np.ascontiguousarray(
                    np.concatenate(rec_ranks, axis=0)
                )
                if conv_dtype is not None:
                    dtypes[conv_name] = conv_dtype
        kda_collect_seconds = time.perf_counter() - kda_started

        dsa_started = time.perf_counter()
        for layer_index, cache_index in sorted(state.dsa_index.items()):
            rank_values, logical_dtype = self._rank_arrays(
                self._gather(state.kc[cache_index][:cached_tokens])
            )
            if self.rank == 0:
                name = f"layers.{layer_index}.keys"
                arrays[name] = require_replicated(rank_values, name=name)
                if logical_dtype is not None:
                    dtypes[name] = logical_dtype
        dsa_collect_seconds = time.perf_counter() - dsa_started

        hidden_started = time.perf_counter()
        hidden_ranks, hidden_dtype = self._rank_arrays(
            self._gather(self.engine.last_hidden)
        )
        if self.rank == 0:
            arrays["prompt.final_hidden"] = require_replicated(
                hidden_ranks, name="prompt.final_hidden"
            )
            if hidden_dtype is not None:
                dtypes["prompt.final_hidden"] = hidden_dtype
        hidden_collect_seconds = time.perf_counter() - hidden_started

        mtp_started = time.perf_counter()
        rank_values, logical_dtype = self._rank_arrays(
            self._gather(state.mtp_kc[:mtp_cached_tokens])
        )
        mtp_collect_seconds = time.perf_counter() - mtp_started
        phase_seconds = {
            "reset": reset_seconds,
            "prefill": prefill_seconds,
            "diagnostic_collect": diagnostic_collect_seconds,
            "kda_collect": kda_collect_seconds,
            "dsa_collect": dsa_collect_seconds,
            "hidden_collect": hidden_collect_seconds,
            "mtp_collect": mtp_collect_seconds,
        }
        if self.rank != 0:
            layer_timeline = None if timeline is None else timeline.report()
            print(json.dumps({
                "event": "glm_cache_prefill",
                "rank": self.rank,
                "cached_tokens": cached_tokens,
                "mtp_cached_tokens": mtp_cached_tokens,
                "spark_first_token": spark_first_token,
                "seconds": time.perf_counter() - started,
                "phase_seconds": phase_seconds,
                "layer_timeline": layer_timeline,
                "kda_trace_layer": self.kda_trace_layer,
            }, sort_keys=True), flush=True)
            return None
        arrays["mtp.keys"] = require_replicated(rank_values, name="mtp.keys")
        if logical_dtype is not None:
            dtypes["mtp.keys"] = logical_dtype
        archive_started = time.perf_counter()
        archive = TensorArchive(
            arrays,
            model_id=self.model_id,
            model_revision=self.model_revision,
            prompt_sha256=prompt_digest(tokens),
            cached_tokens=cached_tokens,
            mtp_cached_tokens=mtp_cached_tokens,
            spark_first_token=spark_first_token,
            dtype_names=dtypes,
        )
        phase_seconds["archive_build"] = time.perf_counter() - archive_started
        layer_timeline = None if timeline is None else timeline.report(_cache_bytes_by_layer(arrays))
        bundle_path = None
        if self.archive_dir is not None:
            bundle_path = str(persist_archive(archive, self.archive_dir))
        print(json.dumps({
            "event": "glm_cache_prefill",
            "rank": self.rank,
            "cached_tokens": cached_tokens,
            "mtp_cached_tokens": mtp_cached_tokens,
            "spark_first_token": spark_first_token,
            "tensor_count": len(archive.manifest.tensors),
            "cache_bytes": sum(item.nbytes for item in archive.manifest.tensors),
            "transfer_id": archive.manifest.transfer_id,
            "seconds": time.perf_counter() - started,
            "phase_seconds": phase_seconds,
            "layer_timeline": layer_timeline,
            "kda_trace_layer": self.kda_trace_layer,
            "bundle_path": bundle_path,
        }, sort_keys=True), flush=True)
        return archive


class ProgressiveProducer:
    """Own one rank-zero CUDA producer without touching the service mailbox."""

    def __init__(
        self,
        exporter: CudaPrefillExporter,
        tokens: Sequence[int],
        job: ProgressiveJob,
        *,
        timeout: float,
    ) -> None:
        self.exporter = exporter
        self.tokens = tuple(int(token) for token in tokens)
        self.job = job
        self.timeout = float(timeout)
        self.error: Exception | None = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"glm-progressive-producer-{job.plan.job_id[:8]}",
            daemon=False,
        )

    def _run(self) -> None:
        try:
            self.exporter.prepare_progressive(
                self.tokens,
                self.job,
                timeout=self.timeout,
            )
        except Exception as exc:
            self.error = exc
            self.job.fail(f"progressive producer failed: {exc}")

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float) -> None:
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise HandoffError("progressive producer did not stop before its deadline")


def rank_zero(exporter: CudaPrefillExporter, mailbox_name: str, socket_path: str, timeout: float) -> dict:
    archive: TensorArchive | None = None
    progressive: ProgressiveProducer | None = None
    prepared = 0
    with ServiceMailbox(mailbox_name, socket_path=socket_path) as mailbox:
        while True:
            incoming = mailbox.next_request(timeout)
            if incoming is None:
                continue
            sequence, raw = incoming
            dispatched = False
            try:
                progressive_protocol = is_progressive_request(raw)
                request = (
                    decode_progressive_request(raw)
                    if progressive_protocol
                    else decode_request(raw)
                )
                if progressive_protocol:
                    assert isinstance(request, ProgressiveRequest)
                    mailbox.bind_generation(request.generation)
                    if request.operation == "prepare":
                        if archive is not None or progressive is not None:
                            raise HandoffError("another cache transfer is still live")
                        if request.model_id != exporter.model_id or request.model_revision != exporter.model_revision:
                            raise HandoffError("Mac and Sparks selected different model identities")
                        if request.prompt_sha256 != prompt_digest(request.tokens):
                            raise HandoffError("progressive prepare prompt digest changed in transit")
                        plan = exporter.progressive_plan(
                            request.tokens,
                            job_id=request.job_id,
                            generation=request.generation,
                        )
                        job = ProgressiveJob(plan, generation=request.generation)
                        dispatched = True
                        _control(
                            exporter.comm,
                            0,
                            (ACTION_PROGRESSIVE_PREPARE, len(request.tokens), 0),
                        )
                        tokens = _token_ids(
                            exporter.comm, 0, request.tokens, len(request.tokens)
                        )
                        dispatched = False
                        progressive = ProgressiveProducer(
                            exporter, tokens, job, timeout=timeout
                        )
                        progressive.start()
                        prepared += 1
                        mailbox.reply(sequence, pack_plan(plan))
                        continue
                    if progressive is None:
                        raise HandoffError("requested progressive cache job is not live")
                    job = progressive.job
                    if (
                        request.job_id != job.plan.job_id
                        or request.plan_id != job.plan.plan_id
                        or request.generation != job.plan.generation
                    ):
                        raise HandoffError("progressive request belongs to another cache job")
                    if request.operation == "frame":
                        reply_progressive_frame(
                            mailbox,
                            sequence,
                            job,
                            request.offset,
                            request.length,
                            timeout=timeout,
                            generation=request.generation,
                        )
                    elif request.operation == "seal":
                        seal = job.wait_seal(
                            timeout=timeout,
                            generation=request.generation,
                        )
                        mailbox.reply(sequence, pack_seal(seal))
                    elif request.operation == "release":
                        if request.transfer_id is not None:
                            seal = job.wait_seal(
                                timeout=timeout,
                                generation=request.generation,
                            )
                            if request.transfer_id != seal.transfer_id:
                                raise HandoffError("released progressive transfer identity is not live")
                        job.release(generation=request.generation)
                        progressive.join(timeout)
                        acknowledgement = pack_progressive_ack(
                            request, request.transfer_id
                        )
                        progressive = None
                        mailbox.reply(sequence, acknowledgement)
                    else:
                        raise HandoffError(
                            f"unsupported progressive cache operation {request.operation!r}"
                        )
                    continue

                assert isinstance(request, Request)
                if request.operation == "prepare":
                    if archive is not None or progressive is not None:
                        raise HandoffError("another cache transfer is still live")
                    if request.model_id != exporter.model_id or request.model_revision != exporter.model_revision:
                        raise HandoffError("Mac and Sparks selected different model identities")
                    if request.prompt_sha256 != prompt_digest(request.tokens):
                        raise HandoffError("prepare request prompt digest changed in transit")
                    dispatched = True
                    _control(exporter.comm, 0, (ACTION_PREPARE, len(request.tokens), 0))
                    tokens = _token_ids(exporter.comm, 0, request.tokens, len(request.tokens))
                    archive = exporter.prepare(tokens)
                    if archive is None:
                        raise RuntimeError("rank zero did not build a cache archive")
                    prepared += 1
                    mailbox.reply(sequence, pack_manifest(archive.manifest))
                elif request.operation == "chunk":
                    if archive is None or request.transfer_id != archive.manifest.transfer_id:
                        raise HandoffError("requested cache transfer is not live")
                    mailbox.reply(sequence, archive.chunk(request.tensor or "", request.offset, request.length))
                elif request.operation == "frame":
                    if archive is None or request.transfer_id != archive.manifest.transfer_id:
                        raise HandoffError("requested cache transfer is not live")
                    reply_frame(mailbox, sequence, archive, request.offset, request.length)
                elif request.operation == "release":
                    if archive is None or request.transfer_id != archive.manifest.transfer_id:
                        raise HandoffError("released cache transfer is not live")
                    transfer_id = archive.manifest.transfer_id
                    archive = None
                    mailbox.reply(sequence, pack_ack("release", transfer_id))
                elif request.operation == "shutdown":
                    if progressive is not None:
                        progressive.job.release(
                            generation=progressive.job.plan.generation
                        )
                        progressive.join(timeout)
                        progressive = None
                    archive = None
                    dispatched = True
                    _control(exporter.comm, 0, (ACTION_SHUTDOWN, 0, 0))
                    reply = pack_ack("shutdown")
                    mailbox.reply(sequence, reply)
                    mailbox.wait_reply_accepted(sequence, len(reply), min(timeout, 30.0))
                    return {"rank": 0, "prepared": prepared}
                else:  # decode_request currently refuses this; keep the service fail-closed if extended.
                    raise HandoffError(f"unsupported cache operation {request.operation!r}")
            except Exception as exc:
                print(json.dumps({"event": "glm_cache_error", "rank": 0,
                                  "error_type": type(exc).__name__, "message": str(exc)},
                                 sort_keys=True), file=sys.stderr, flush=True)
                if dispatched:
                    raise
                mailbox.reply(sequence, pack_error(exc))


def rank_one(exporter: CudaPrefillExporter) -> dict:
    prepared = 0
    while True:
        action, count, _ = _control(exporter.comm, 1)
        if action == ACTION_PREPARE:
            tokens = _token_ids(exporter.comm, 1, None, count)
            exporter.prepare(tokens)
            prepared += 1
        elif action == ACTION_PROGRESSIVE_PREPARE:
            tokens = _token_ids(exporter.comm, 1, None, count)
            exporter.prepare_progressive(tokens, None, timeout=600.0)
            prepared += 1
        elif action == ACTION_SHUTDOWN:
            return {"rank": 1, "prepared": prepared}
        else:
            raise RuntimeError(f"unknown rank-zero cache action {action}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--port", type=int, default=29641)
    parser.add_argument("--mailbox", default="p06cfr")
    parser.add_argument("--socket", default="/tmp/mcdma-rpcd.p06cfr.sock")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--prefill-rows", type=int, default=128)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--archive-dir")
    parser.add_argument("--layer-timeline", action="store_true")
    parser.add_argument(
        "--kda-trace-layer",
        type=int,
        help="diagnostic only: export aligned CUDA intermediates for one KDA prompt layer",
    )
    args = parser.parse_args()
    if args.kda_trace_layer is not None and not 0 <= args.kda_trace_layer < 45:
        parser.error("--kda-trace-layer must be between 0 and 44")
    exporter = CudaPrefillExporter(
        args.model,
        rank=args.rank,
        master=args.master,
        port=args.port,
        model_id=args.model_id,
        model_revision=args.model_revision,
        prefill_rows=args.prefill_rows,
        archive_dir=args.archive_dir if args.rank == 0 else None,
        layer_timeline=args.layer_timeline,
        kda_trace_layer=args.kda_trace_layer,
    )
    result = (rank_zero(exporter, args.mailbox, args.socket, args.timeout)
              if args.rank == 0 else rank_one(exporter))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
