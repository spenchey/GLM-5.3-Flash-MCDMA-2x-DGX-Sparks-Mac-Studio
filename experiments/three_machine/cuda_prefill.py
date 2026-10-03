"""Two-Spark GLM prefill service for verified cache handoff to the Mac.

Both CUDA ranks run the complete portable 4-bit checkpoint.  Rank zero merges
head-sharded KDA state, verifies that latent attention state is identical on
both ranks, and exposes the resulting archive through the existing MCDMA
mailbox.  Decoding is intentionally absent from this process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Sequence

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


ACTION_PREPARE = 1
ACTION_SHUTDOWN = 2
CONTROL_WORDS = 3
DEFAULT_MODEL_ID = "Vontra/GLM-5.3-Flash-MLX-4bit-MTP"
DEFAULT_MODEL_REVISION = "76add2a341a1cd90ad0e86bb69839ea9c35827c6"


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
        self.comm = NCCL(self.rank, 2, master, port)
        self.comm.barrier()
        self.w = load(model_dir, rank=self.rank, mtp=True)
        self.w.comm = self.comm
        self.comm.ready("cache-handoff-weights")
        self.comm.barrier()
        if len(self.w.layers) != 45 or self.w.mtp is None:
            raise RuntimeError("the pinned portable GLM checkpoint did not load 45 layers and its MTP head")
        self.capacity = int(self.w.cfg.dense_limit)
        # Handoff v1 transfers latent attention keys, but not the sparse
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

    def prepare(self, prompt: Sequence[int]) -> TensorArchive | None:
        """Prefill all but the last token and include that pair in MTP state."""

        from tensorfold.families.glm5_next.cuda.decode import absorb, prefill

        tokens = [int(token) for token in prompt]
        if len(tokens) < 2:
            raise HandoffError("cache handoff needs at least two prompt tokens")
        cached_tokens = len(tokens) - 1
        if len(tokens) > self.handoff_limit:
            raise HandoffError(
                f"prompt has {len(tokens)} tokens; cache handoff v1 supports at most "
                f"{self.handoff_limit} before sparse index state is required"
            )
        # A released transfer must not contaminate the next independent
        # request.  Both ranks reset before entering any collective work.
        self.engine.reset()
        started = time.perf_counter()
        prefill(self.engine, tokens[:-1], None, mtp=True)
        if self.engine.last_hidden is None:
            raise RuntimeError("CUDA prefill did not retain its final hidden row")
        absorb(self.engine, self.engine.last_hidden, [tokens[-1]])
        state = self.engine.st
        if state.pos != cached_tokens or state.mtp_len != cached_tokens or state.mtp_drafted:
            raise RuntimeError(
                f"CUDA cache stopped at main={state.pos}, mtp={state.mtp_len}, drafted={state.mtp_drafted}; "
                f"expected {cached_tokens}, {cached_tokens}, 0"
            )

        arrays: dict[str, np.ndarray] = {}
        dtypes: dict[str, str] = {}
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

        for layer_index, cache_index in sorted(state.dsa_index.items()):
            rank_values, logical_dtype = self._rank_arrays(
                self._gather(state.kc[cache_index][:cached_tokens])
            )
            if self.rank == 0:
                name = f"layers.{layer_index}.keys"
                arrays[name] = require_replicated(rank_values, name=name)
                if logical_dtype is not None:
                    dtypes[name] = logical_dtype

        rank_values, logical_dtype = self._rank_arrays(
            self._gather(state.mtp_kc[:cached_tokens])
        )
        if self.rank != 0:
            print(json.dumps({
                "event": "glm_cache_prefill",
                "rank": self.rank,
                "cached_tokens": cached_tokens,
                "seconds": time.perf_counter() - started,
            }, sort_keys=True), flush=True)
            return None
        arrays["mtp.keys"] = require_replicated(rank_values, name="mtp.keys")
        if logical_dtype is not None:
            dtypes["mtp.keys"] = logical_dtype
        archive = TensorArchive(
            arrays,
            model_id=self.model_id,
            model_revision=self.model_revision,
            prompt_sha256=prompt_digest(tokens),
            cached_tokens=cached_tokens,
            dtype_names=dtypes,
        )
        bundle_path = None
        if self.archive_dir is not None:
            bundle_path = str(persist_archive(archive, self.archive_dir))
        print(json.dumps({
            "event": "glm_cache_prefill",
            "rank": self.rank,
            "cached_tokens": cached_tokens,
            "tensor_count": len(archive.manifest.tensors),
            "cache_bytes": sum(item.nbytes for item in archive.manifest.tensors),
            "transfer_id": archive.manifest.transfer_id,
            "seconds": time.perf_counter() - started,
            "bundle_path": bundle_path,
        }, sort_keys=True), flush=True)
        return archive


def rank_zero(exporter: CudaPrefillExporter, mailbox_name: str, socket_path: str, timeout: float) -> dict:
    archive: TensorArchive | None = None
    prepared = 0
    with ServiceMailbox(mailbox_name, socket_path=socket_path) as mailbox:
        while True:
            incoming = mailbox.next_request(timeout)
            if incoming is None:
                continue
            sequence, raw = incoming
            dispatched = False
            try:
                request = decode_request(raw)
                if request.operation == "prepare":
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
    args = parser.parse_args()
    exporter = CudaPrefillExporter(
        args.model,
        rank=args.rank,
        master=args.master,
        port=args.port,
        model_id=args.model_id,
        model_revision=args.model_revision,
        prefill_rows=args.prefill_rows,
        archive_dir=args.archive_dir if args.rank == 0 else None,
    )
    result = (rank_zero(exporter, args.mailbox, args.socket, args.timeout)
              if args.rank == 0 else rank_one(exporter))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
