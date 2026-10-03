"""Exact full GLM main model on two Sparks; Mac supplies only MTP proposals.

Rank 0 owns the MCDMA mailbox.  Every prompt or verification token window is
mirrored to rank 1, both ranks run TensorFold's normal complete CUDA model, and
TensorFold's sharded sampler chooses the exact token.  Rank 0 returns those
tokens plus the final-normalized hidden rows consumed by the Mac MTP head.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass

import torch

from experiments.three_machine import protocol
from experiments.three_machine.cuda_partial import _control, _rank_memory, _token_ids
from experiments.three_machine.mailbox import ServiceMailbox
from experiments.three_machine.transaction import Session


ACTION_TOKEN_WINDOW = 9
ACTION_ACK = 2
ACTION_RESET = 3
ACTION_SHUTDOWN = 4
ACTION_NOOP = 5
ACTION_DUPLICATE = 6
DENSE_CONTEXT_CAPACITY = 2051


@dataclass(frozen=True)
class MainRun:
    accepted: list[int]
    hidden: bytes
    keep: int


class MainCUDA:
    """TensorFold's complete two-rank main model with MTP deliberately absent."""

    def __init__(self, model_dir: str, rank: int, master: str, port: int, capacity: int,
                 prefill_rows: int = 64) -> None:
        from tensorfold.families.glm5_next.cuda.engine import GlmEngine

        self.rank = rank
        # serial_only leaves the MTP layer out of Spark memory.  Speculation is
        # supplied by the Mac, while verification remains TensorFold-native.
        self.owner = GlmEngine(
            model_dir,
            rank=rank,
            master=master,
            port=port,
            context=capacity,
            context_explicit=True,
            serial_only=True,
            prefill_rows=prefill_rows,
            vision=False,
            parallel=1,
        )
        self.e = self.owner.e
        self.comm = self.owner.comm
        self.context_capacity = int(self.owner.limit)
        if capacity > self.context_capacity:
            raise RuntimeError(
                f"requested CUDA capacity {capacity} exceeds the admitted {self.context_capacity}"
            )
        if prefill_rows != self.e.prefill_rows:
            raise RuntimeError("TensorFold did not retain the requested prompt chunk size")
        self.awaiting_ack: tuple[int, int] | None = None
        self.expected_first: int | None = None
        self.last_run_metrics: dict[str, float | int] = {}
        # GlmEngine's calibration is collective and RoCE switches out of its
        # startup-safe mode before this wrapper begins.  Do not let the faster
        # rank enter a runtime collective until both constructors are fully
        # finished, or it can consume the peer's final calibration payload.
        self.comm.ready("three-machine-runtime")
        self.comm.barrier()

    def ring(self) -> None:
        """Wake rank 1 through TensorFold's store before a runtime collective."""

        self.owner._ring()

    def await_action(self) -> None:
        """Keep rank 1 out of RoCE while no Mac request is waiting."""

        self.owner._await_bell()

    @staticmethod
    def _hidden_bytes(rows: torch.Tensor) -> bytes:
        data = rows.detach().contiguous().to("cpu")
        if data.dtype != torch.bfloat16:
            raise RuntimeError("TensorFold final-normalized hidden rows are not BF16")
        return data.view(torch.uint16).numpy().tobytes(order="C")

    def process(self, tokens: list[int], *, prompt: bool, final: bool) -> MainRun:
        from tensorfold.families.glm5_next.cuda.decode import prefill_chunk
        from tensorfold.families.glm5_next.cuda.forward import commit

        if self.awaiting_ack is not None:
            raise protocol.ProtocolError("the previous Spark token window is still awaiting acknowledgement")
        if final and not prompt:
            raise protocol.ProtocolError("only the final prompt window may carry FINAL")
        rows = len(tokens)
        if not 1 <= rows <= protocol.WINDOW_MAX_ROWS:
            raise protocol.ProtocolError("a Spark token window must contain 1..64 rows")
        torch.cuda.synchronize()
        started = time.perf_counter()
        if prompt:
            if self.expected_first is not None:
                raise protocol.ProtocolError("prompt rows arrived after decoding began")
            if rows > self.e.prefill_rows:
                raise protocol.ProtocolError("prompt window exceeds TensorFold's prefill buffer")
            start = self.e.st.pos
            logits = prefill_chunk(
                self.e,
                tokens,
                0,
                rows,
                drafter=None,
                feed=None,
                mtp=False,
                head=final,
            )
            hidden_rows = self.e.pbuf.fnormed[:rows]
            accepted: list[int] = []
            if final:
                if logits is None:
                    raise RuntimeError("final prompt did not produce logits")
                accepted = self.e.sample(logits[:1], [start + rows], None)
                self.e.follow(accepted)
                self.expected_first = accepted[0]
            keep = rows
        else:
            if self.expected_first is None:
                raise protocol.ProtocolError("decode arrived before the final prompt")
            if tokens[0] != self.expected_first:
                raise protocol.ProtocolError("decode begins with a token other than the last exact Spark sample")
            if rows > self.e.rows:
                raise protocol.ProtocolError("decode window exceeds TensorFold's exact verification width")
            logits = self.e.forward(self.e.verify_window(tokens))
            sampled = self.e.sample(
                logits[:rows],
                [self.e.st.pos + 1 + index for index in range(rows)],
                None,
            )
            keep = 1
            eos = {int(value) for value in self.e.w.cfg.eos}
            for index, candidate in enumerate(tokens[1:]):
                if sampled[index] != candidate or sampled[index] in eos:
                    break
                keep += 1
            accepted = sampled[:keep]
            hidden_rows = self.e.main_hidden(slice(0, keep))
            commit(self.e.w, self.e.st, self.e.buf, rows, keep)
            self.e.follow(accepted)
            self.expected_first = accepted[-1]
        hidden = self._hidden_bytes(hidden_rows) if self.rank == 0 else b""
        torch.cuda.synchronize()
        self.awaiting_ack = (rows, keep)
        self.last_run_metrics = {
            "gpu_total_s": time.perf_counter() - started,
            "verified_drafts": 0 if prompt else rows - 1,
            "accepted_drafts": 0 if prompt else keep - 1,
        }
        return MainRun(accepted, hidden, keep)

    def ack(self, rows: int, keep: int) -> None:
        if self.awaiting_ack != (rows, keep):
            raise protocol.ProtocolError("Spark acknowledgement differs from the completed token window")
        self.awaiting_ack = None

    def reset(self) -> None:
        self.e.reset()
        self.e.last_hidden = None
        self.e.head = None
        self.e.window = None
        self.awaiting_ack = None
        self.expected_first = None


def _error(session: Session | None, message: str, timeout: float) -> bytes:
    return protocol.pack(protocol.error(
        request_id=(session.request_id if session else 0) or 0,
        step=session.step if session else 0,
        committed=session.committed if session else 0,
        generation=session.generation if session else 0,
        message=message[:1000],
        timeout_s=timeout,
    ))


def validate_token_capacity(frame: protocol.Frame, capacity: int) -> None:
    if frame.committed + frame.rows > capacity:
        raise protocol.ProtocolError(
            f"token window needs {frame.committed + frame.rows} positions; this deployment supports {capacity}"
        )


def rank_zero(engine: MainCUDA, name: str, socket_path: str, timeout: float) -> dict:
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
                frame = protocol.unpack(raw, now_ns=time.time_ns())
                mailbox.bind_generation(frame.generation)
                if session is None:
                    session = Session(frame.generation)
                if frame.kind is protocol.Kind.TOKEN_WINDOW:
                    cached = session.classify_activation(frame)
                    if cached is not None:
                        action_dispatched = True
                        engine.ring()
                        _control(engine.comm, 0, (ACTION_DUPLICATE, 0, 0, 0, 0, frame.request_id))
                        mailbox.reply(sequence, protocol.pack(cached))
                        continue
                    validate_token_capacity(frame, engine.context_capacity)
                    tokens = protocol.token_window_values(frame)
                    action_dispatched = True
                    engine.ring()
                    _control(engine.comm, 0, (
                        ACTION_TOKEN_WINDOW,
                        frame.rows,
                        int(frame.flags),
                        frame.step,
                        frame.committed,
                        frame.request_id,
                    ))
                    mirrored = _token_ids(engine.comm, 0, tokens, frame.rows)
                    result = engine.process(
                        mirrored,
                        prompt=bool(frame.flags & protocol.Flags.PROMPT),
                        final=bool(frame.flags & protocol.Flags.FINAL),
                    )
                    reply = session.propose_hidden(
                        frame,
                        result.accepted,
                        result.hidden,
                        keep=result.keep,
                        timeout_s=timeout,
                    )
                    mailbox.reply(sequence, protocol.pack(reply))
                    print(json.dumps({
                        "event": "three_machine_main_step",
                        "request_id": frame.request_id,
                        "step": frame.step,
                        "committed": frame.committed,
                        "rows": frame.rows,
                        "keep": result.keep,
                        "prompt": bool(frame.flags & protocol.Flags.PROMPT),
                        "spark_service_s": time.perf_counter() - request_started,
                        **engine.last_run_metrics,
                    }, sort_keys=True, separators=(",", ":")), flush=True)
                elif frame.kind is protocol.Kind.ACK:
                    pending = session.validate_ack(frame)
                    action_dispatched = True
                    engine.ring()
                    _control(engine.comm, 0, (
                        ACTION_ACK,
                        pending.rows,
                        pending.keep,
                        frame.step,
                        frame.committed,
                        frame.request_id,
                    ))
                    engine.ack(pending.rows, pending.keep)
                    session.commit(pending)
                    completed += 1
                    mailbox.reply(sequence, protocol.pack(protocol.control(
                        kind=protocol.Kind.ACK,
                        request_id=frame.request_id,
                        step=frame.step,
                        committed=frame.committed,
                        generation=session.generation,
                        timeout_s=timeout,
                    )))
                    print(json.dumps({
                        "event": "three_machine_main_commit",
                        "rank": 0,
                        "request_id": frame.request_id,
                        "step": frame.step,
                        "committed": frame.committed,
                        "committed_windows": completed,
                        **_rank_memory(),
                    }, sort_keys=True, separators=(",", ":")), flush=True)
                elif frame.kind is protocol.Kind.RESET:
                    session.validate_reset(frame)
                    action_dispatched = True
                    engine.ring()
                    _control(engine.comm, 0, (ACTION_RESET, 0, 0, 0, 0, frame.request_id))
                    engine.reset()
                    session.reset()
                    mailbox.reply(sequence, protocol.pack(protocol.control(
                        kind=protocol.Kind.ACK,
                        request_id=frame.request_id,
                        step=0,
                        committed=0,
                        generation=session.generation,
                        timeout_s=timeout,
                    )))
                elif frame.kind is protocol.Kind.SHUTDOWN:
                    session.validate_shutdown(frame)
                    action_dispatched = True
                    engine.ring()
                    _control(engine.comm, 0, (ACTION_SHUTDOWN, 0, 0, 0, 0, frame.request_id))
                    mailbox.reply(sequence, protocol.pack(protocol.control(
                        kind=protocol.Kind.ACK,
                        request_id=frame.request_id,
                        step=frame.step,
                        committed=frame.committed,
                        generation=session.generation,
                        timeout_s=timeout,
                    )))
                    return {"rank": 0, "committed_windows": completed, "committed_tokens": session.committed}
                else:
                    raise protocol.ProtocolError(f"unsupported request kind {frame.kind.name}")
            except Exception as exc:
                if action_dispatched:
                    raise
                print(json.dumps({
                    "event": "three_machine_main_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "generation": session.generation if session else None,
                }, sort_keys=True, separators=(",", ":")), file=sys.stderr, flush=True)
                engine.ring()
                _control(engine.comm, 0, (ACTION_NOOP, 0, 0, 0, 0, 0))
                mailbox.reply(sequence, _error(session, f"{type(exc).__name__}: {exc}", timeout))


def rank_one(engine: MainCUDA) -> dict:
    completed = 0
    while True:
        engine.await_action()
        action, rows, auxiliary, _step, _committed, request_id = _control(engine.comm, 1)
        if action == ACTION_TOKEN_WINDOW:
            tokens = _token_ids(engine.comm, 1, None, rows)
            flags = protocol.Flags(auxiliary)
            engine.process(
                tokens,
                prompt=bool(flags & protocol.Flags.PROMPT),
                final=bool(flags & protocol.Flags.FINAL),
            )
        elif action == ACTION_ACK:
            engine.ack(rows, auxiliary)
            completed += 1
            print(json.dumps({
                "event": "three_machine_main_commit",
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
            return {"rank": 1, "committed_windows": completed, "committed_tokens": engine.e.st.pos}
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
    parser.add_argument("--prefill-rows", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    engine = MainCUDA(args.model, args.rank, args.master, args.port, args.capacity, args.prefill_rows)
    result = rank_zero(engine, args.mailbox, args.socket, args.timeout) if args.rank == 0 else rank_one(engine)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
