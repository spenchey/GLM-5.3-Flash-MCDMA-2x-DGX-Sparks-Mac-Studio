"""Acknowledged position state for one Mac-stage/Spark-stage inference stream.

The accepted position advances only after the Mac acknowledges a token. Cache
writes happen eagerly during forward, so a process/link fault still requires a
full reset or recovery. A repeated activation before acknowledgement receives
the cached token and is not computed twice.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, replace

from experiments.three_machine import protocol


@dataclass(frozen=True)
class Pending:
    kind: protocol.Kind
    request_id: int
    step: int
    base_committed: int
    proposed_committed: int
    rows: int
    keep: int
    digest: bytes
    flags: protocol.Flags
    reply: protocol.Frame


class Session:
    """Validate activation order and hold one uncommitted reply."""

    def __init__(self, generation: int) -> None:
        if generation <= 0:
            raise protocol.ProtocolError("link generation must be positive")
        self.generation = generation
        self.request_id: int | None = None
        self.step = 0
        self.committed = 0
        self.pending: Pending | None = None
        # The mailbox has one synchronous slot, so only a small bounded replay
        # window is needed. Retiring completed request IDs prevents an old
        # activation copied near RESET from becoming a new stream at position 0.
        self._retired: deque[int] = deque(maxlen=64)

    def _identity(self, frame: protocol.Frame) -> None:
        if frame.generation != self.generation:
            raise protocol.ProtocolError("frame belongs to a different MCDMA link generation")
        if self.request_id is not None and frame.request_id != self.request_id:
            raise protocol.ProtocolError("another request cannot replace the active inference stream")

    def classify_activation(self, frame: protocol.Frame) -> protocol.Frame | None:
        """Return a cached reply for an exact retry, or None when compute is required."""

        if frame.kind not in (
            protocol.Kind.ACTIVATION,
            protocol.Kind.WINDOW_ACTIVATION,
            protocol.Kind.TOKEN_WINDOW,
        ):
            raise protocol.ProtocolError("expected an activation frame")
        self._identity(frame)
        if frame.request_id in self._retired:
            raise protocol.ProtocolError("activation belongs to a reset inference stream")
        digest = hashlib.sha256(frame.payload).digest()
        if self.pending is not None:
            p = self.pending
            if (frame.kind, frame.request_id, frame.step, frame.committed, frame.rows, digest, frame.flags) == (
                    p.kind, p.request_id, p.step, p.base_committed, p.rows, p.digest, p.flags):
                # A retry can carry a later valid deadline. Keep the cached token
                # identity and bytes, but do not return an already-expired frame.
                return replace(p.reply, deadline_ns=max(p.reply.deadline_ns, frame.deadline_ns))
            raise protocol.ProtocolError("a different activation arrived before the pending token was acknowledged")
        if frame.step != self.step or frame.committed != self.committed:
            raise protocol.ProtocolError("activation step or committed position is out of order")
        if self.request_id is None:
            self.request_id = frame.request_id
        return None

    def propose(self, frame: protocol.Frame, tokens: list[int], *, timeout_s: float = 30.0) -> protocol.Frame:
        if self.pending is not None:
            raise protocol.ProtocolError("a token is already pending acknowledgement")
        proposed = self.committed + frame.rows
        reply = protocol.token(request_id=frame.request_id, step=frame.step, committed=proposed,
                               generation=self.generation, tokens=tokens, timeout_s=timeout_s)
        self.pending = Pending(frame.kind, frame.request_id, frame.step, self.committed, proposed, frame.rows,
                               frame.rows,
                               hashlib.sha256(frame.payload).digest(), frame.flags, reply)
        return reply

    def propose_window(self, frame: protocol.Frame, accepted: list[int], drafts: list[int], *, keep: int,
                       timeout_s: float = 30.0) -> protocol.Frame:
        """Propose a partially kept verification window without advancing state before ACK."""

        if frame.kind is not protocol.Kind.WINDOW_ACTIVATION:
            raise protocol.ProtocolError("window proposal requires a window activation")
        if self.pending is not None:
            raise protocol.ProtocolError("a window result is already pending acknowledgement")
        if not 1 <= keep <= frame.rows:
            raise protocol.ProtocolError("window keep must be within the activation rows")
        prompt = bool(frame.flags & protocol.Flags.PROMPT)
        if prompt and (keep != frame.rows or len(accepted) != 1):
            raise protocol.ProtocolError("prompt window must keep every row and return one sampled token")
        if not prompt and len(accepted) != keep:
            raise protocol.ProtocolError("decode window must return one accepted token per kept row")
        proposed = self.committed + keep
        reply = protocol.window_result(
            request_id=frame.request_id,
            step=frame.step,
            committed=proposed,
            generation=self.generation,
            accepted=accepted,
            drafts=drafts,
            timeout_s=timeout_s,
        )
        self.pending = Pending(frame.kind, frame.request_id, frame.step, self.committed, proposed, frame.rows, keep,
                               hashlib.sha256(frame.payload).digest(), frame.flags, reply)
        return reply

    def propose_hidden(self, frame: protocol.Frame, accepted: list[int], hidden: bytes, *, keep: int,
                       timeout_s: float = 30.0) -> protocol.Frame:
        """Propose exact main-model tokens plus the hidden rows the Mac drafter consumes."""

        if frame.kind is not protocol.Kind.TOKEN_WINDOW:
            raise protocol.ProtocolError("hidden proposal requires a token window")
        if self.pending is not None:
            raise protocol.ProtocolError("a verified hidden result is already pending acknowledgement")
        if not 1 <= keep <= frame.rows:
            raise protocol.ProtocolError("verified hidden keep must be within the token-window rows")
        prompt = bool(frame.flags & protocol.Flags.PROMPT)
        final = bool(frame.flags & protocol.Flags.FINAL)
        if prompt:
            expected = 1 if final else 0
            if keep != frame.rows or len(accepted) != expected:
                raise protocol.ProtocolError(
                    "prompt token window must keep every row and sample only at the final prompt"
                )
            hidden_rows = frame.rows
        else:
            if len(accepted) != keep:
                raise protocol.ProtocolError("decode token window must return one accepted token per kept row")
            hidden_rows = keep
        proposed = self.committed + keep
        reply = protocol.verified_hidden(
            request_id=frame.request_id,
            step=frame.step,
            committed=proposed,
            generation=self.generation,
            accepted=accepted,
            hidden=hidden,
            rows=hidden_rows,
            prompt=prompt,
            final=final,
            timeout_s=timeout_s,
        )
        self.pending = Pending(
            frame.kind,
            frame.request_id,
            frame.step,
            self.committed,
            proposed,
            frame.rows,
            keep,
            hashlib.sha256(frame.payload).digest(),
            frame.flags,
            reply,
        )
        return reply

    def validate_reset(self, frame: protocol.Frame) -> None:
        """Require a reset to name the exact active generation and base position."""

        if frame.kind is not protocol.Kind.RESET:
            raise protocol.ProtocolError("expected a reset frame")
        if frame.generation != self.generation:
            raise protocol.ProtocolError("reset belongs to a different MCDMA link generation")
        if self.request_id is not None and frame.request_id != self.request_id:
            raise protocol.ProtocolError("reset belongs to another inference stream")
        if (frame.step, frame.committed) != (self.step, self.committed):
            raise protocol.ProtocolError("reset position differs from the active inference stream")

    def validate_shutdown(self, frame: protocol.Frame) -> None:
        """Require shutdown only at the exact clean session position."""

        if frame.kind is not protocol.Kind.SHUTDOWN:
            raise protocol.ProtocolError("expected a shutdown frame")
        if self.pending is not None:
            raise protocol.ProtocolError("cannot shut down with an unacknowledged token")
        if frame.generation != self.generation:
            raise protocol.ProtocolError("shutdown belongs to a different MCDMA link generation")
        if self.request_id is not None and frame.request_id != self.request_id:
            raise protocol.ProtocolError("shutdown belongs to another inference stream")
        if (frame.step, frame.committed) != (self.step, self.committed):
            raise protocol.ProtocolError("shutdown position differs from the active inference stream")

    def validate_ack(self, frame: protocol.Frame) -> Pending:
        if frame.kind is not protocol.Kind.ACK or self.pending is None:
            raise protocol.ProtocolError("there is no pending token to acknowledge")
        self._identity(frame)
        p = self.pending
        if (frame.request_id, frame.step, frame.committed) != (p.request_id, p.step, p.proposed_committed):
            raise protocol.ProtocolError("acknowledgement does not match the pending token")
        return p

    def commit(self, pending: Pending) -> None:
        if self.pending != pending:
            raise protocol.ProtocolError("pending transaction changed before commit")
        self.committed = pending.proposed_committed
        self.step += 1
        self.pending = None

    def reset(self) -> None:
        if self.request_id is not None and self.request_id not in self._retired:
            self._retired.append(self.request_id)
        self.request_id = None
        self.step = 0
        self.committed = 0
        self.pending = None
