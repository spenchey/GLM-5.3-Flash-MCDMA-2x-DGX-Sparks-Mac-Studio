"""Continue GLM-5.3 after Metal layer 0 on both TensorFold CUDA ranks.

Rank 0 owns the MCDMA mailbox.  Each accepted activation is mirrored to rank 1
over TensorFold's existing NCCL fabric, both ranks execute layers 1..44, and
the normal sharded head selects one identical greedy token.  Verified window
state advances before MTP drafts its successor; the separate Mac ACK makes
that tentative advance durable at the cross-machine session boundary.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass

import torch

from experiments.three_machine import protocol
from experiments.three_machine.mailbox import ServiceMailbox
from experiments.three_machine.transaction import Session

ACTION_ACTIVATE = 1
ACTION_ACK = 2
ACTION_RESET = 3
ACTION_SHUTDOWN = 4
ACTION_NOOP = 5
ACTION_DUPLICATE = 6
ACTION_WINDOW_ACTIVATE = 7
CONTROL_WORDS = 6
DENSE_CONTEXT_CAPACITY = 2051


@dataclass(frozen=True)
class WindowRun:
    """One verified decode result and the uncommitted drafts for its successor."""

    accepted: list[int]
    drafts: list[int]
    keep: int


@dataclass(frozen=True)
class PendingCUDA:
    """One Spark window awaiting the Mac's matching session ACK."""

    buffer: object
    rows: int
    keep: int
    state_advanced: bool


def _tensorfold_commit(weights, state, buffer, rows: int, keep: int) -> None:
    """Keep the TensorFold import patchable for ordering regression tests."""

    from tensorfold.families.glm5_next.cuda.forward import commit

    commit(weights, state, buffer, rows, keep)


def _tensorfold_draft(engine, hidden: torch.Tensor, accepted: list[int]) -> list[int]:
    """Draft only after TensorFold's verified state position has advanced."""

    from tensorfold.families.glm5_next.cuda.decode import draft

    return draft(
        engine,
        hidden,
        accepted,
        engine.state.pos + 1,
        engine.draft_count,
        None,
        engine.mtp_confidence,
    )


def _process_rss_bytes() -> int:
    """Return this Python rank's current resident set, not docker-init's."""

    with open("/proc/self/status", encoding="utf-8") as handle:
        for raw in handle:
            if raw.startswith("VmRSS:"):
                return int(raw.split()[1]) * 1024
    raise RuntimeError("/proc/self/status has no VmRSS entry")


def _rank_memory() -> dict[str, int]:
    """Evidence for both CUDA accounting and unified host memory growth."""

    return {
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
        "process_rss_bytes": _process_rss_bytes(),
    }


def _control(
    comm,
    rank: int,
    values: tuple[int, int, int, int, int, int] | None = None,
) -> tuple[int, int, int, int, int, int]:
    send = torch.zeros((CONTROL_WORDS,), dtype=torch.int64, device="cuda")
    if rank == 0:
        if values is None:
            raise ValueError("rank 0 must supply control values")
        send.copy_(torch.tensor(values, dtype=torch.int64, device="cuda"))
    recv = torch.empty((comm.world * CONTROL_WORDS,), dtype=torch.int64, device="cuda")
    comm.all_gather(send, recv)
    torch.cuda.synchronize()
    return tuple(int(x) for x in recv[:CONTROL_WORDS].cpu().tolist())


def _activation(comm, rank: int, payload: bytes | None, rows: int) -> torch.Tensor:
    values = rows * protocol.WIDTH
    if rank == 0:
        if payload is None or len(payload) != values * 2:
            raise protocol.ProtocolError("rank 0 activation bytes do not match the declared rows")
        host = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(rows, protocol.WIDTH)
        send = host.to("cuda")
    else:
        send = torch.zeros((rows, protocol.WIDTH), dtype=torch.bfloat16, device="cuda")
    recv = torch.empty((comm.world * values,), dtype=torch.bfloat16, device="cuda")
    comm.all_gather(send.reshape(-1), recv)
    torch.cuda.synchronize()
    return recv[:values].reshape(rows, protocol.WIDTH)


def _token_ids(comm, rank: int, tokens: list[int] | None, rows: int) -> list[int]:
    """Mirror the authenticated token ids beside their activation rows."""

    if rank == 0:
        if tokens is None or len(tokens) != rows:
            raise protocol.ProtocolError("rank 0 token ids do not match the declared rows")
        send = torch.tensor(tokens, dtype=torch.int32, device="cuda")
    else:
        send = torch.zeros((rows,), dtype=torch.int32, device="cuda")
    recv = torch.empty((comm.world * rows,), dtype=torch.int32, device="cuda")
    comm.all_gather(send, recv)
    torch.cuda.synchronize()
    return [int(value) for value in recv[:rows].cpu().tolist()]


class PartialCUDA:
    """TensorFold CUDA state whose layer list begins at original layer 1."""

    def __init__(self, model_dir: str, rank: int, master: str, port: int, capacity: int, max_rows: int,
                 drafts: int = 3, mtp_confidence: float = 0.35) -> None:
        from tensorfold.cuda.comm import NCCL
        from tensorfold.families.glm5_next import config as metal_config
        from tensorfold.families.glm5_next.cuda.forward import Buffers, State
        from tensorfold.families.glm5_next.cuda.weights import load

        self.rank = rank
        self.comm = NCCL(rank, 2, master, port)
        if not 1 <= drafts <= 8:
            raise ValueError("MTP draft count must be in 1..8")
        if not 0.0 <= mtp_confidence <= 1.0:
            raise ValueError("MTP confidence must be in 0..1")
        full = load(model_dir, rank=rank, mtp=True)
        full.comm = self.comm
        self.comm.ready("three-machine-weights")
        if len(full.layers) != 45 or full.layers[0].index != 0:
            raise RuntimeError("expected the pinned 45-layer GLM-5.3 checkpoint")
        self.w = copy.copy(full)
        self.w.layers = list(full.layers[1:])
        self.w.meta = dict(full.meta, layers=[layer.index for layer in self.w.layers], metal_layers=[0])
        self.w.comm = self.comm
        if self.w.mtp is None:
            raise RuntimeError("the pinned GLM checkpoint did not load its MTP head")
        self.context_capacity = int(self.w.cfg.dense_limit)
        if capacity != self.context_capacity:
            raise RuntimeError(
                f"requested CUDA capacity {capacity} differs from the model's "
                f"supported dense capacity {self.context_capacity}"
            )
        self.state = State(self.w, capacity, max_rows)
        self.st = self.state
        self.decode = Buffers(self.w, max_rows, capacity)
        self.prefill = Buffers(self.w, max_rows, capacity, prefill=True)
        self.mbuf = Buffers(self.w, min(max_rows, 8), capacity)
        self.draft_n = self.w.head.n
        self.decode_rows = int(metal_config.DECODE_ROWS)
        self.draft_count = int(drafts)
        self.mtp_confidence = float(mtp_confidence)
        self.pending: PendingCUDA | None = None
        self.expected_window: list[int] | None = None
        self.prompt_last_hidden: torch.Tensor | None = None
        self.phase_timing = os.environ.get("THREE_MACHINE_PHASE_TIMING", "0") == "1"
        self.last_run_metrics: dict[str, float] = {}
        self.comm.ready("three-machine-buffers")

    def run(self, activation: torch.Tensor, rows: int) -> int:
        from tensorfold.families.glm5_next.cuda import glue
        from tensorfold.families.glm5_next.cuda.decode import sample_rows
        from tensorfold.families.glm5_next.cuda.forward import check_room, chunks_for, layer_forward, mm

        if self.pending is not None:
            raise protocol.ProtocolError("CUDA already has an uncommitted activation")
        b = self.prefill if rows > self.decode_rows else self.decode
        check_room(self.w, self.state, rows)
        if self.phase_timing:
            torch.cuda.synchronize()
        gpu_began = time.perf_counter()
        began = gpu_began
        b.x[:rows].copy_(activation)
        if self.phase_timing:
            torch.cuda.synchronize()
        copy_s = time.perf_counter() - began
        nch = chunks_for(self.state, rows)
        began = time.perf_counter()
        for layer in self.w.layers:
            layer_forward(layer, self.w, self.state, b, rows, nch=nch, host_pos=self.state.pos)
        if self.phase_timing:
            torch.cuda.synchronize()
        layers_s = time.perf_counter() - began
        began = time.perf_counter()
        glue.stream_mean(b.x[:rows], b.hidden[:rows])
        glue.rmsnorm(b.hidden[:rows], self.w.norm, self.w.cfg.eps, b.fnormed[:rows], b.fxs[:rows])
        logits = mm(b, b.fnormed[rows - 1:rows], self.w.head, b.fxs[rows - 1:rows], b.logits[:1])
        token = sample_rows(self.w, logits, [self.state.pos + rows - 1], None)[0]
        if self.phase_timing:
            torch.cuda.synchronize()
        head_s = time.perf_counter() - began
        began = time.perf_counter()
        agreed = torch.empty((self.comm.world,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(torch.tensor([token], dtype=torch.int32, device="cuda"), agreed)
        torch.cuda.synchronize()
        agreement_s = time.perf_counter() - began
        gpu_total_s = time.perf_counter() - gpu_began
        if len(set(int(x) for x in agreed.cpu().tolist())) != 1:
            raise RuntimeError("the two CUDA ranks selected different tokens")
        self.pending = PendingCUDA(b, rows, rows, False)
        self.last_run_metrics = {"gpu_total_s": gpu_total_s}
        if self.phase_timing:
            self.last_run_metrics.update({
                "buffer_copy_s": copy_s,
                "layers_s": layers_s,
                "head_s": head_s,
                "agreement_s": agreement_s,
            })
        return token

    def sample(self, logits: torch.Tensor, positions, sampling, *, draft: bool = False,
               probs: list[float] | None = None) -> list[int]:
        from tensorfold.families.glm5_next.cuda.decode import sample_rows

        return sample_rows(self.w, logits, positions, sampling, None, probs)

    def mtp(self, next_tokens, hidden: torch.Tensor) -> torch.Tensor:
        from tensorfold.families.glm5_next.cuda.decode import mtp_compute, mtp_stage
        from tensorfold.families.glm5_next.cuda.attention import CHUNK

        n = mtp_stage(self.w, self.state, self.mbuf, next_tokens, hidden)
        return mtp_compute(
            self.w,
            self.state,
            self.mbuf,
            n,
            nch=-(-(self.state.mtp_len + n) // CHUNK),
            host_pos=self.state.mtp_len,
        )

    def draft_hidden(self, _row: int) -> torch.Tensor:
        return self.mbuf.fnormed[0:1]

    def _prompt_absorb(self, hidden: torch.Tensor, next_tokens: list[int]) -> None:
        """Advance the prompt-side MTP cache using TensorFold's prefill arithmetic."""

        if not next_tokens:
            return
        from tensorfold.families.glm5_next.cuda.decode import mtp_forward

        mtp_forward(self.w, self.state, self.prefill, next_tokens, hidden)
        self.state.set_mtp_len(self.state.mtp_len + len(next_tokens))

    def _advance_window(self, buffer, rows: int, keep: int) -> None:
        """Tentatively advance verified Spark state before producing new drafts."""

        _tensorfold_commit(self.w, self.state, buffer, rows, keep)
        self.pending = PendingCUDA(buffer, rows, keep, True)

    def run_window(self, activation: torch.Tensor, tokens: list[int], *, prompt: bool, final: bool) -> WindowRun:
        """Verify one token-bound Metal window and prepare the next MTP draft chain."""

        from tensorfold.families.glm5_next.cuda import glue
        from tensorfold.families.glm5_next.cuda.forward import check_room, chunks_for, layer_forward, mm

        rows = len(tokens)
        if self.pending is not None:
            raise protocol.ProtocolError("CUDA already has an uncommitted activation")
        if final and not prompt:
            raise protocol.ProtocolError("only a prompt window may be final")
        if not prompt:
            if self.expected_window is None:
                raise protocol.ProtocolError("decode arrived before the final prompt window")
            if tokens != self.expected_window[:rows]:
                raise protocol.ProtocolError("decode tokens differ from the MTP window issued by CUDA")
        b = self.prefill if prompt else self.decode
        check_room(self.w, self.state, rows)
        if self.phase_timing:
            torch.cuda.synchronize()
        gpu_began = time.perf_counter()
        began = gpu_began
        b.x[:rows].copy_(activation)
        if self.phase_timing:
            torch.cuda.synchronize()
        copy_s = time.perf_counter() - began
        nch = chunks_for(self.state, rows)
        began = time.perf_counter()
        for layer in self.w.layers:
            layer_forward(layer, self.w, self.state, b, rows, nch=nch, host_pos=self.state.pos)
        if self.phase_timing:
            torch.cuda.synchronize()
        layers_s = time.perf_counter() - began
        began = time.perf_counter()
        glue.stream_mean(b.x[:rows], b.hidden[:rows])
        glue.rmsnorm(b.hidden[:rows], self.w.norm, self.w.cfg.eps, b.fnormed[:rows], b.fxs[:rows])
        head_rows = b.fnormed[rows - 1:rows] if prompt else b.fnormed[:rows]
        head_fxs = b.fxs[rows - 1:rows] if prompt else b.fxs[:rows]
        logits = mm(b, head_rows, self.w.head, head_fxs, b.logits[:1] if prompt else b.logits[:rows])
        positions = [self.state.pos + rows] if prompt else [self.state.pos + 1 + index for index in range(rows)]
        sampled = self.sample(logits, positions, None)
        if self.phase_timing:
            torch.cuda.synchronize()
        head_s = time.perf_counter() - began
        began = time.perf_counter()
        if prompt:
            current_hidden = b.fnormed[:rows].clone()
            absorb_hidden: list[torch.Tensor] = []
            absorb_tokens: list[int] = []
            if self.prompt_last_hidden is not None:
                absorb_hidden.append(self.prompt_last_hidden)
                absorb_tokens.append(tokens[0])
            if rows > 1:
                absorb_hidden.append(current_hidden[:-1])
                absorb_tokens.extend(tokens[1:])
            if absorb_hidden:
                self._prompt_absorb(torch.cat(absorb_hidden, dim=0), absorb_tokens)
            self.prompt_last_hidden = current_hidden[-1:].clone()
            accepted = [sampled[0]]
            keep = rows
            drafts = []
            self._advance_window(b, rows, keep)
            if final:
                drafts = _tensorfold_draft(self, self.prompt_last_hidden, accepted)
                self.expected_window = [accepted[-1], *drafts]
        else:
            keep = 1
            eos = {int(value) for value in self.w.cfg.eos}
            for index, candidate in enumerate(tokens[1:]):
                if sampled[index] != candidate or sampled[index] in eos:
                    break
                keep += 1
            accepted = sampled[:keep]
            drafts = []
            self._advance_window(b, rows, keep)
            if accepted[-1] not in eos:
                drafts = _tensorfold_draft(self, b.fnormed[:keep], accepted)
            self.expected_window = [accepted[-1], *drafts]
        draft_s = time.perf_counter() - began
        self.last_run_metrics = {
            "gpu_total_s": time.perf_counter() - gpu_began,
            "verified_drafts": 0 if prompt else rows - 1,
            "accepted_drafts": 0 if prompt else keep - 1,
            "next_drafts": len(drafts),
        }
        if self.phase_timing:
            self.last_run_metrics.update({
                "buffer_copy_s": copy_s,
                "layers_s": layers_s,
                "head_s": head_s,
                "draft_s": draft_s,
            })
        return WindowRun(accepted, drafts, keep)

    def commit(self, keep: int | None = None) -> None:
        if self.pending is None:
            raise protocol.ProtocolError("CUDA has no pending activation")
        pending = self.pending
        keep = pending.keep if keep is None else int(keep)
        if keep != pending.keep:
            raise protocol.ProtocolError("CUDA commit differs from the verified keep count")
        if not pending.state_advanced:
            _tensorfold_commit(self.w, self.state, pending.buffer, pending.rows, keep)
        self.pending = None

    def reset(self) -> None:
        self.state.reset()
        self.pending = None
        self.expected_window = None
        self.prompt_last_hidden = None


def _error(session: Session | None, message: str, timeout: float) -> bytes:
    return protocol.pack(protocol.error(request_id=(session.request_id if session else 0) or 0,
                                        step=session.step if session else 0,
                                        committed=session.committed if session else 0,
                                        generation=session.generation if session else 0,
                                        message=message[:1000], timeout_s=timeout))


def validate_activation_capacity(frame: protocol.Frame, capacity: int) -> None:
    """Reject an over-capacity activation before either Spark enters a collective."""

    if frame.committed + frame.rows > capacity:
        raise protocol.ProtocolError(
            f"activation needs {frame.committed + frame.rows} positions; "
            f"this deployment supports {capacity}"
        )


def rank_zero(engine: PartialCUDA, name: str, socket_path: str, timeout: float) -> dict:
    completed = 0
    with ServiceMailbox(name, socket_path=socket_path) as mailbox:
        session: Session | None = None
        while True:
            request = mailbox.next_request(timeout)
            if request is None:
                continue
            sequence, raw = request
            action_dispatched = False
            try:
                request_started = time.perf_counter()
                began = time.perf_counter()
                frame = protocol.unpack(raw, now_ns=time.time_ns())
                mailbox.bind_generation(frame.generation)
                unpack_s = time.perf_counter() - began
                if session is None:
                    session = Session(frame.generation)
                if frame.kind in (protocol.Kind.ACTIVATION, protocol.Kind.WINDOW_ACTIVATION):
                    cached = session.classify_activation(frame)
                    if cached is not None:
                        action_dispatched = True
                        _control(engine.comm, 0, (ACTION_DUPLICATE, 0, 0, 0, 0, frame.request_id))
                        mailbox.reply(sequence, protocol.pack(cached))
                        continue
                    validate_activation_capacity(frame, engine.context_capacity)
                    action_dispatched = True
                    began = time.perf_counter()
                    action = (ACTION_WINDOW_ACTIVATE
                              if frame.kind is protocol.Kind.WINDOW_ACTIVATION else ACTION_ACTIVATE)
                    _control(
                        engine.comm,
                        0,
                        (action, frame.rows, int(frame.flags), frame.step, frame.committed, frame.request_id),
                    )
                    control_s = time.perf_counter() - began
                    began = time.perf_counter()
                    if frame.kind is protocol.Kind.WINDOW_ACTIVATION:
                        tokens, payload = protocol.window_activation_values(frame)
                    else:
                        tokens, payload = None, frame.payload
                    activation = _activation(engine.comm, 0, payload, frame.rows)
                    mirrored_tokens = (_token_ids(engine.comm, 0, tokens, frame.rows)
                                       if tokens is not None else None)
                    activation_collective_s = time.perf_counter() - began
                    if mirrored_tokens is not None:
                        result = engine.run_window(
                            activation,
                            mirrored_tokens,
                            prompt=bool(frame.flags & protocol.Flags.PROMPT),
                            final=bool(frame.flags & protocol.Flags.FINAL),
                        )
                        began = time.perf_counter()
                        reply = session.propose_window(
                            frame,
                            result.accepted,
                            result.drafts,
                            keep=result.keep,
                            timeout_s=timeout,
                        )
                    else:
                        selected = engine.run(activation, frame.rows)
                        began = time.perf_counter()
                        reply = session.propose(frame, [selected], timeout_s=timeout)
                    session_propose_s = time.perf_counter() - began
                    began = time.perf_counter()
                    mailbox.reply(sequence, protocol.pack(reply))
                    reply_s = time.perf_counter() - began
                    spark_service_s = time.perf_counter() - request_started
                    print(json.dumps({
                        "event": "three_machine_step",
                        "request_id": frame.request_id,
                        "step": frame.step,
                        "committed": frame.committed,
                        "rows": frame.rows,
                        "prompt": bool(frame.flags & protocol.Flags.PROMPT),
                        "timing_mode": "synchronized" if engine.phase_timing else "host-enqueue",
                        "unpack_s": unpack_s,
                        "control_collective_s": control_s,
                        "activation_collective_s": activation_collective_s,
                        **engine.last_run_metrics,
                        "session_propose_s": session_propose_s,
                        "reply_s": reply_s,
                        "spark_service_s": spark_service_s,
                    }, sort_keys=True, separators=(",", ":")), flush=True)
                elif frame.kind is protocol.Kind.ACK:
                    pending = session.validate_ack(frame)
                    action_dispatched = True
                    _control(
                        engine.comm,
                        0,
                        (ACTION_ACK, pending.rows, pending.keep, frame.step, frame.committed, frame.request_id),
                    )
                    engine.commit(pending.keep)
                    session.commit(pending)
                    completed += 1
                    print(json.dumps({
                        "event": "three_machine_rank0_commit",
                        "rank": 0,
                        "request_id": frame.request_id,
                        "step": frame.step,
                        "committed": frame.committed,
                        "committed_windows": completed,
                        **_rank_memory(),
                    }, sort_keys=True, separators=(",", ":")), flush=True)
                    answer = protocol.control(kind=protocol.Kind.ACK, request_id=frame.request_id,
                                              step=frame.step, committed=frame.committed,
                                              generation=session.generation, timeout_s=timeout)
                    mailbox.reply(sequence, protocol.pack(answer))
                elif frame.kind is protocol.Kind.RESET:
                    session.validate_reset(frame)
                    action_dispatched = True
                    _control(engine.comm, 0, (ACTION_RESET, 0, 0, 0, 0, frame.request_id))
                    engine.reset()
                    session.reset()
                    mailbox.reply(sequence, protocol.pack(protocol.control(
                        kind=protocol.Kind.ACK, request_id=frame.request_id, step=0, committed=0,
                        generation=session.generation, timeout_s=timeout)))
                elif frame.kind is protocol.Kind.SHUTDOWN:
                    session.validate_shutdown(frame)
                    action_dispatched = True
                    _control(engine.comm, 0, (ACTION_SHUTDOWN, 0, 0, 0, 0, frame.request_id))
                    mailbox.reply(sequence, protocol.pack(protocol.control(
                        kind=protocol.Kind.ACK, request_id=frame.request_id, step=frame.step,
                        committed=frame.committed, generation=session.generation, timeout_s=timeout)))
                    return {"rank": 0, "committed_windows": completed, "committed_tokens": session.committed}
                else:
                    raise protocol.ProtocolError(f"unsupported request kind {frame.kind.name}")
            except Exception as exc:  # keep rank 1 in lockstep only for errors found before a collective action
                if action_dispatched:
                    raise
                print(json.dumps({
                    "event": "three_machine_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "generation": session.generation if session else None,
                }, sort_keys=True, separators=(",", ":")), file=sys.stderr, flush=True)
                _control(engine.comm, 0, (ACTION_NOOP, 0, 0, 0, 0, 0))
                mailbox.reply(sequence, _error(session, f"{type(exc).__name__}: {exc}", timeout))


def rank_one(engine: PartialCUDA) -> dict:
    completed = 0
    while True:
        action, rows, auxiliary, _step, _committed, request_id = _control(engine.comm, 1)
        if action == ACTION_ACTIVATE:
            activation = _activation(engine.comm, 1, None, rows)
            engine.run(activation, rows)
        elif action == ACTION_WINDOW_ACTIVATE:
            activation = _activation(engine.comm, 1, None, rows)
            tokens = _token_ids(engine.comm, 1, None, rows)
            flags = protocol.Flags(auxiliary)
            engine.run_window(
                activation,
                tokens,
                prompt=bool(flags & protocol.Flags.PROMPT),
                final=bool(flags & protocol.Flags.FINAL),
            )
        elif action == ACTION_ACK:
            engine.commit(auxiliary)
            completed += 1
            print(json.dumps({
                "event": "three_machine_rank1_commit",
                "rank": 1,
                "request_id": request_id,
                "step": _step,
                "committed": _committed,
                "committed_windows": completed,
                **_rank_memory(),
            }, sort_keys=True, separators=(",", ":")), flush=True)
        elif action == ACTION_RESET:
            engine.reset()
        elif action == ACTION_SHUTDOWN:
            return {"rank": 1, "committed_windows": completed, "committed_tokens": engine.state.pos}
        elif action not in (ACTION_NOOP, ACTION_DUPLICATE):
            raise RuntimeError(f"unknown rank-0 action {action}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--master", default="192.0.2.1")
    parser.add_argument("--port", type=int, default=29631)
    parser.add_argument("--mailbox", default="tfglm53")
    parser.add_argument("--socket", default="/tmp/mcdma-rpcd.tfglm53.sock")
    parser.add_argument("--capacity", type=int, default=DENSE_CONTEXT_CAPACITY)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--drafts", type=int, default=3)
    parser.add_argument("--mtp-confidence", type=float, default=0.35)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    engine = PartialCUDA(
        args.model,
        args.rank,
        args.master,
        args.port,
        args.capacity,
        args.max_rows,
        args.drafts,
        args.mtp_confidence,
    )
    result = (rank_zero(engine, args.mailbox, args.socket, args.timeout) if args.rank == 0 else rank_one(engine))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
