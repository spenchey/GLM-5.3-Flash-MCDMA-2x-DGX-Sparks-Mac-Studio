"""Fail-closed wire frames for the Mac-to-Spark GLM stage boundary.

The same bytes are used by the reference TCP path and the MCDMA mailbox path.
The first accepted boundary is four BF16 GLM residual streams flattened to
``[rows, 16384]``.  A frame is self-identifying and checksum protected; no
transport is allowed to infer shape or state from payload length alone.
"""

from __future__ import annotations

import enum
import hashlib
import struct
import time
from dataclasses import dataclass

MAGIC = b"TFMCDMA1"
# Version 2 authenticates the complete header plus payload. Version 1 covered
# only payload bytes and is intentionally rejected rather than silently mixed.
VERSION = 2
WIDTH = 16_384
BYTES_PER_VALUE = 2
ROW_BYTES = WIDTH * BYTES_PER_VALUE
MAX_ROWS = 64
ACTIVATION_MAX_PAYLOAD = MAX_ROWS * ROW_BYTES
# A token-bearing activation or verified-hidden reply carries one uint32 token
# beside every BF16 row.  The MCDMA mailbox reserves 4 MiB per direction, so a
# complete 64-row window fits with its authenticated token identities.
WINDOW_ROW_BYTES = ROW_BYTES + 4
MAX_PAYLOAD = MAX_ROWS * WINDOW_ROW_BYTES
WINDOW_MAX_ROWS = MAX_ROWS


class ProtocolError(ValueError):
    """A frame is malformed, unsupported, stale, or internally inconsistent."""


class Kind(enum.IntEnum):
    ACTIVATION = 1
    TOKEN = 2
    RESET = 3
    ERROR = 4
    SHUTDOWN = 5
    ACK = 6
    WINDOW_ACTIVATION = 7
    WINDOW_RESULT = 8
    TOKEN_WINDOW = 9
    VERIFIED_HIDDEN = 10


class DType(enum.IntEnum):
    NONE = 0
    BF16 = 1
    UINT32 = 2
    UTF8 = 3


class Flags(enum.IntFlag):
    NONE = 0
    PROMPT = 1
    DECODE = 2
    FINAL = 4


# magic, version, kind, flags, request, step, committed, generation,
# rows, width, dtype, reserved, payload length, absolute deadline, sha256
_HEADER = struct.Struct(">8sHHIQQQQIIHHIQ32s")
HEADER_BYTES = _HEADER.size


@dataclass(frozen=True)
class Frame:
    kind: Kind
    request_id: int
    step: int
    committed: int
    generation: int
    rows: int
    width: int
    dtype: DType
    payload: bytes
    deadline_ns: int
    flags: Flags = Flags.NONE


def _unsigned(name: str, value: int, bits: int = 64) -> None:
    if type(value) is not int or not 0 <= value < 1 << bits:
        raise ProtocolError(f"{name} must be an unsigned {bits}-bit integer")


def _validate(frame: Frame, *, now_ns: int | None = None) -> None:
    for name in ("request_id", "step", "committed", "generation", "deadline_ns"):
        _unsigned(name, getattr(frame, name))
    _unsigned("rows", frame.rows, 32)
    _unsigned("width", frame.width, 32)
    try:
        kind, dtype, flags = Kind(frame.kind), DType(frame.dtype), Flags(frame.flags)
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc
    payload = bytes(frame.payload)
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError("payload exceeds the fixed authenticated transport window")
    if not frame.deadline_ns:
        raise ProtocolError("frame deadline must be nonzero")
    if (now_ns if now_ns is not None else time.time_ns()) > frame.deadline_ns:
        raise ProtocolError("frame deadline has expired")
    if kind is Kind.ACTIVATION:
        if not 1 <= frame.rows <= MAX_ROWS or frame.width != WIDTH or dtype is not DType.BF16:
            raise ProtocolError("activation must be BF16 [1..64, 16384]")
        if len(payload) != frame.rows * ROW_BYTES:
            raise ProtocolError("activation byte length does not match its rows")
        if int(flags) & ~int(Flags.PROMPT | Flags.DECODE):
            raise ProtocolError("activation contains an unsupported flag")
        if bool(flags & Flags.PROMPT) == bool(flags & Flags.DECODE):
            raise ProtocolError("activation must name exactly one of prompt or decode")
    elif kind is Kind.WINDOW_ACTIVATION:
        if not 1 <= frame.rows <= WINDOW_MAX_ROWS or frame.width != WIDTH or dtype is not DType.BF16:
            raise ProtocolError("window activation must be BF16 [1..63, 16384] with bound token ids")
        if len(payload) != frame.rows * WINDOW_ROW_BYTES:
            raise ProtocolError("window activation byte length does not match its rows")
        if int(flags) & ~int(Flags.PROMPT | Flags.DECODE | Flags.FINAL):
            raise ProtocolError("window activation contains an unsupported flag")
        if bool(flags & Flags.PROMPT) == bool(flags & Flags.DECODE):
            raise ProtocolError("window activation must name exactly one of prompt or decode")
        if flags & Flags.FINAL and not flags & Flags.PROMPT:
            raise ProtocolError("only the final prompt window may carry FINAL")
    elif kind is Kind.TOKEN_WINDOW:
        if not 1 <= frame.rows <= WINDOW_MAX_ROWS or frame.width != 1 or dtype is not DType.UINT32:
            raise ProtocolError("token window must be UINT32 [1..64, 1]")
        if len(payload) != frame.rows * 4:
            raise ProtocolError("token window byte length does not match its rows")
        if int(flags) & ~int(Flags.PROMPT | Flags.DECODE | Flags.FINAL):
            raise ProtocolError("token window contains an unsupported flag")
        if bool(flags & Flags.PROMPT) == bool(flags & Flags.DECODE):
            raise ProtocolError("token window must name exactly one of prompt or decode")
        if flags & Flags.FINAL and not flags & Flags.PROMPT:
            raise ProtocolError("only the final prompt window may carry FINAL")
    elif kind is Kind.VERIFIED_HIDDEN:
        if not 1 <= frame.rows <= WINDOW_MAX_ROWS or not 0 <= frame.width <= frame.rows:
            raise ProtocolError("verified hidden must declare BF16 rows and accepted-token count")
        if dtype is not DType.BF16 or len(payload) != frame.width * 4 + frame.rows * ROW_BYTES:
            raise ProtocolError("verified hidden payload does not match its token and BF16 rows")
        if int(flags) & ~int(Flags.PROMPT | Flags.DECODE | Flags.FINAL):
            raise ProtocolError("verified hidden contains an unsupported flag")
        if bool(flags & Flags.PROMPT) == bool(flags & Flags.DECODE):
            raise ProtocolError("verified hidden must name exactly one of prompt or decode")
        if flags & Flags.FINAL and not flags & Flags.PROMPT:
            raise ProtocolError("only the final prompt result may carry FINAL")
    elif kind is Kind.TOKEN:
        if not 1 <= frame.rows <= MAX_ROWS or frame.width != 1 or dtype is not DType.UINT32:
            raise ProtocolError("token frame must be UINT32 [1..64, 1]")
        if len(payload) != frame.rows * 4 or flags:
            raise ProtocolError("token frame shape or flags are invalid")
    elif kind is Kind.WINDOW_RESULT:
        if not 1 <= frame.rows <= MAX_ROWS or not 1 <= frame.width <= frame.rows or dtype is not DType.UINT32:
            raise ProtocolError("window result must declare accepted and drafted uint32 tokens")
        if len(payload) != frame.rows * 4 or flags:
            raise ProtocolError("window result shape or flags are invalid")
    elif kind is Kind.ERROR:
        if frame.rows or frame.width or dtype is not DType.UTF8 or flags:
            raise ProtocolError("error frame metadata is invalid")
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("error payload is not UTF-8") from exc
    else:
        if frame.rows or frame.width or dtype is not DType.NONE or payload or flags:
            raise ProtocolError(f"{kind.name.lower()} frame must not carry a payload")


def pack(frame: Frame, *, now_ns: int | None = None) -> bytes:
    """Return one exact frame after validating every declared field."""

    _validate(frame, now_ns=now_ns)
    payload = bytes(frame.payload)
    zero_digest = bytes(32)
    header_without_digest = _HEADER.pack(
        MAGIC,
        VERSION,
        int(frame.kind),
        int(frame.flags),
        frame.request_id,
        frame.step,
        frame.committed,
        frame.generation,
        frame.rows,
        frame.width,
        int(frame.dtype),
        0,
        len(payload),
        frame.deadline_ns,
        zero_digest,
    )
    digest = hashlib.sha256(header_without_digest + payload).digest()
    header = header_without_digest[:-32] + digest
    return header + payload


def unpack(data: bytes | bytearray | memoryview, *, now_ns: int | None = None) -> Frame:
    """Parse exactly one frame. Truncation, trailing bytes, and corruption fail."""

    raw = bytes(data)
    if len(raw) < HEADER_BYTES:
        raise ProtocolError("truncated frame header")
    (magic, version, kind, flags, request_id, step, committed, generation, rows, width, dtype, reserved,
     length, deadline_ns, digest) = _HEADER.unpack_from(raw)
    if magic != MAGIC or version != VERSION or reserved:
        raise ProtocolError("unsupported frame header")
    if length > MAX_PAYLOAD or len(raw) != HEADER_BYTES + length:
        raise ProtocolError("frame length is truncated, oversized, or has trailing bytes")
    payload = raw[HEADER_BYTES:]
    header_without_digest = _HEADER.pack(
        magic,
        version,
        kind,
        flags,
        request_id,
        step,
        committed,
        generation,
        rows,
        width,
        dtype,
        reserved,
        length,
        deadline_ns,
        bytes(32),
    )
    if hashlib.sha256(header_without_digest + payload).digest() != digest:
        raise ProtocolError("frame checksum mismatch")
    try:
        frame = Frame(Kind(kind), request_id, step, committed, generation, rows, width, DType(dtype), payload,
                      deadline_ns, Flags(flags))
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc
    _validate(frame, now_ns=now_ns)
    return frame


def activation(*, request_id: int, step: int, committed: int, generation: int, rows: int, payload: bytes,
               prompt: bool, timeout_s: float = 30.0) -> Frame:
    if timeout_s <= 0:
        raise ProtocolError("timeout must be positive")
    return Frame(Kind.ACTIVATION, request_id, step, committed, generation, rows, WIDTH, DType.BF16, payload,
                 time.time_ns() + int(timeout_s * 1e9), Flags.PROMPT if prompt else Flags.DECODE)


def window_activation(*, request_id: int, step: int, committed: int, generation: int,
                      tokens: list[int], payload: bytes, prompt: bool, final: bool = False,
                      timeout_s: float = 30.0) -> Frame:
    """Bind every Metal activation row to the token id that produced it."""

    if timeout_s <= 0:
        raise ProtocolError("timeout must be positive")
    if not tokens or len(tokens) > WINDOW_MAX_ROWS:
        raise ProtocolError(f"window tokens must contain 1..{WINDOW_MAX_ROWS} values")
    if any(type(item) is not int or not 0 <= item < 1 << 32 for item in tokens):
        raise ProtocolError("window tokens must be uint32 values")
    if len(payload) != len(tokens) * ROW_BYTES:
        raise ProtocolError("window activation rows do not match the token count")
    token_bytes = struct.pack(f">{len(tokens)}I", *tokens)
    flags = Flags.PROMPT if prompt else Flags.DECODE
    if final:
        flags |= Flags.FINAL
    return Frame(Kind.WINDOW_ACTIVATION, request_id, step, committed, generation, len(tokens), WIDTH,
                 DType.BF16, token_bytes + bytes(payload), time.time_ns() + int(timeout_s * 1e9), flags)


def window_activation_values(frame: Frame) -> tuple[list[int], bytes]:
    """Return the authenticated input token ids and their Metal activation rows."""

    _validate(frame)
    if frame.kind is not Kind.WINDOW_ACTIVATION:
        raise ProtocolError("expected a window activation frame")
    token_bytes = frame.rows * 4
    tokens = list(struct.unpack(f">{frame.rows}I", frame.payload[:token_bytes]))
    return tokens, frame.payload[token_bytes:]


def token_window(*, request_id: int, step: int, committed: int, generation: int,
                 tokens: list[int], prompt: bool, final: bool = False,
                 timeout_s: float = 30.0) -> Frame:
    """Send only token ids to the complete Spark main model."""

    if timeout_s <= 0:
        raise ProtocolError("timeout must be positive")
    if not tokens or len(tokens) > WINDOW_MAX_ROWS:
        raise ProtocolError(f"token window must contain 1..{WINDOW_MAX_ROWS} values")
    if any(type(item) is not int or not 0 <= item < 1 << 32 for item in tokens):
        raise ProtocolError("token window values must be uint32")
    flags = Flags.PROMPT if prompt else Flags.DECODE
    if final:
        flags |= Flags.FINAL
    payload = struct.pack(f">{len(tokens)}I", *tokens)
    return Frame(Kind.TOKEN_WINDOW, request_id, step, committed, generation, len(tokens), 1,
                 DType.UINT32, payload, time.time_ns() + int(timeout_s * 1e9), flags)


def token_window_values(frame: Frame) -> list[int]:
    """Return authenticated token ids for a complete-main-model window."""

    _validate(frame)
    if frame.kind is not Kind.TOKEN_WINDOW:
        raise ProtocolError("expected a token window frame")
    return list(struct.unpack(f">{frame.rows}I", frame.payload))


def verified_hidden(*, request_id: int, step: int, committed: int, generation: int,
                    accepted: list[int], hidden: bytes, rows: int, prompt: bool,
                    final: bool = False, timeout_s: float = 30.0) -> Frame:
    """Return exact main-model choices and their final-normalized hidden rows."""

    if timeout_s <= 0:
        raise ProtocolError("timeout must be positive")
    if not 1 <= rows <= WINDOW_MAX_ROWS or len(hidden) != rows * ROW_BYTES:
        raise ProtocolError("verified hidden rows do not match the BF16 payload")
    if len(accepted) > rows or any(type(item) is not int or not 0 <= item < 1 << 32 for item in accepted):
        raise ProtocolError("accepted values must be bounded uint32 tokens")
    flags = Flags.PROMPT if prompt else Flags.DECODE
    if final:
        flags |= Flags.FINAL
    token_bytes = struct.pack(f">{len(accepted)}I", *accepted) if accepted else b""
    return Frame(Kind.VERIFIED_HIDDEN, request_id, step, committed, generation, rows, len(accepted),
                 DType.BF16, token_bytes + bytes(hidden), time.time_ns() + int(timeout_s * 1e9), flags)


def verified_hidden_values(frame: Frame) -> tuple[list[int], bytes]:
    """Split a verified main-model reply into accepted ids and BF16 hidden rows."""

    _validate(frame)
    if frame.kind is not Kind.VERIFIED_HIDDEN:
        raise ProtocolError("expected a verified hidden frame")
    token_bytes = frame.width * 4
    accepted = (list(struct.unpack(f">{frame.width}I", frame.payload[:token_bytes]))
                if frame.width else [])
    return accepted, frame.payload[token_bytes:]


def checked_verified_hidden(
    frame: Frame,
    *,
    request_id: int,
    step: int,
    generation: int,
    base_committed: int,
    input_rows: int,
    prompt: bool,
    final: bool,
) -> tuple[list[int], bytes, int]:
    """Validate a complete-main-model reply and return accepted ids, rows and keep."""

    accepted, hidden = verified_hidden_values(frame)
    if (frame.request_id, frame.step, frame.generation) != (request_id, step, generation):
        raise ProtocolError("verified hidden identity differs from the pending token window")
    expected_flags = Flags.PROMPT if prompt else Flags.DECODE
    if final:
        expected_flags |= Flags.FINAL
    if frame.flags != expected_flags:
        raise ProtocolError("verified hidden mode differs from the pending token window")
    keep = frame.committed - base_committed
    if not 1 <= keep <= input_rows:
        raise ProtocolError("verified hidden committed position is outside the pending token window")
    if prompt:
        expected = 1 if final else 0
        if keep != input_rows or len(accepted) != expected or frame.rows != input_rows:
            raise ProtocolError("prompt hidden result has an invalid keep, sample, or row count")
    elif keep != len(accepted) or frame.rows != keep:
        raise ProtocolError("decode hidden result must return one hidden row per accepted token")
    return accepted, hidden, keep


def token(*, request_id: int, step: int, committed: int, generation: int, tokens: list[int],
          timeout_s: float = 30.0) -> Frame:
    if not tokens or any(type(item) is not int or not 0 <= item < 1 << 32 for item in tokens):
        raise ProtocolError("tokens must be nonempty uint32 values")
    payload = struct.pack(f">{len(tokens)}I", *tokens)
    return Frame(Kind.TOKEN, request_id, step, committed, generation, len(tokens), 1, DType.UINT32, payload,
                 time.time_ns() + int(timeout_s * 1e9))


def token_values(frame: Frame) -> list[int]:
    _validate(frame)
    if frame.kind is not Kind.TOKEN:
        raise ProtocolError("expected a token frame")
    return list(struct.unpack(f">{frame.rows}I", frame.payload))


def window_result(*, request_id: int, step: int, committed: int, generation: int,
                  accepted: list[int], drafts: list[int], timeout_s: float = 30.0) -> Frame:
    """Return verified output tokens followed by uncommitted candidates for the next window."""

    values = [*accepted, *drafts]
    if not accepted:
        raise ProtocolError("window result must accept at least one output token")
    if len(values) > MAX_ROWS or any(type(item) is not int or not 0 <= item < 1 << 32 for item in values):
        raise ProtocolError("window result values must contain at most 64 uint32 tokens")
    if timeout_s <= 0:
        raise ProtocolError("timeout must be positive")
    payload = struct.pack(f">{len(values)}I", *values)
    return Frame(Kind.WINDOW_RESULT, request_id, step, committed, generation, len(values), len(accepted),
                 DType.UINT32, payload, time.time_ns() + int(timeout_s * 1e9))


def window_result_values(frame: Frame) -> tuple[list[int], list[int]]:
    """Split a window reply into committed output and next-round draft tokens."""

    _validate(frame)
    if frame.kind is not Kind.WINDOW_RESULT:
        raise ProtocolError("expected a window result frame")
    values = list(struct.unpack(f">{frame.rows}I", frame.payload))
    return values[:frame.width], values[frame.width:]


def checked_window_result(
    frame: Frame,
    *,
    request_id: int,
    step: int,
    generation: int,
    base_committed: int,
    input_rows: int,
    prompt: bool,
) -> tuple[list[int], list[int], int]:
    """Validate reply identity and return accepted tokens, drafts, and input rows kept."""

    accepted, drafts = window_result_values(frame)
    if (frame.request_id, frame.step, frame.generation) != (request_id, step, generation):
        raise ProtocolError("window result identity differs from the pending activation")
    keep = frame.committed - base_committed
    if not 1 <= keep <= input_rows:
        raise ProtocolError("window result committed position is outside the pending activation")
    if prompt:
        if keep != input_rows or len(accepted) != 1:
            raise ProtocolError("prompt window must keep every input row and return one sampled token")
    elif keep != len(accepted):
        raise ProtocolError("decode window must keep one input row per accepted output token")
    if len(accepted) + len(drafts) > MAX_ROWS:
        raise ProtocolError("window result exceeds the supported token window")
    return accepted, drafts, keep


def checked_token_values(
    frame: Frame,
    *,
    request_id: int,
    step: int,
    committed: int,
    generation: int,
    count: int = 1,
) -> list[int]:
    """Return tokens only when the reply belongs to the exact request position."""

    values = token_values(frame)
    if (frame.request_id, frame.step, frame.committed, frame.generation) != (
        request_id,
        step,
        committed,
        generation,
    ):
        raise ProtocolError("token reply identity differs from the pending activation")
    if len(values) != count:
        raise ProtocolError(f"token reply must contain exactly {count} value(s)")
    return values


def control(*, kind: Kind, request_id: int, step: int, committed: int, generation: int,
            timeout_s: float = 30.0) -> Frame:
    """Build a payload-free transaction control frame."""

    if kind not in (Kind.RESET, Kind.SHUTDOWN, Kind.ACK):
        raise ProtocolError("control kind must be RESET, SHUTDOWN, or ACK")
    if timeout_s <= 0:
        raise ProtocolError("timeout must be positive")
    return Frame(kind, request_id, step, committed, generation, 0, 0, DType.NONE, b"",
                 time.time_ns() + int(timeout_s * 1e9))


def error(*, request_id: int, step: int, committed: int, generation: int, message: str,
          timeout_s: float = 30.0) -> Frame:
    """Build a bounded UTF-8 error reply without leaking an exception traceback."""

    if timeout_s <= 0:
        raise ProtocolError("timeout must be positive")
    payload = str(message).encode("utf-8")
    if not payload:
        raise ProtocolError("error message must not be empty")
    return Frame(Kind.ERROR, request_id, step, committed, generation, 0, 0, DType.UTF8, payload,
                 time.time_ns() + int(timeout_s * 1e9))
