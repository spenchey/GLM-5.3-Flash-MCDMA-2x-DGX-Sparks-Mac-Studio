"""Python bindings for ABI-1 ``mcdma-rpcd`` request/reply mailboxes.

The daemon mailbox ABI is separate from the authenticated TensorFold frame
protocol carried inside each request and reply.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import secrets
import select
import socket
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, TypeVar

CTRL = 4096
READY_WORD = 64
GENERATION_WORD = 72
STAGED_WORD = 128
SIZES = 256
DEFAULT_HALF = 4 << 20
ABI = 1
_U32 = 0xFFFFFFFF
_ReplyResult = TypeVar("_ReplyResult")


class MailboxError(RuntimeError):
    """The daemon mailbox is unavailable, stale, disconnected, or malformed."""


def _next_sequence(sequence: int) -> int:
    """Burn one published sequence and advance monotonically with wraparound."""

    return (sequence % _U32) + 1


def _initial_sequence(*seen: int, generation: int) -> int:
    """Choose an unpredictable nonzero sequence away from visible old words."""

    used = {value & _U32 for value in seen if value & _U32}
    folded = (generation ^ (generation >> 32)) & _U32
    mixed_visible = 0
    for value in used:
        mixed_visible = ((mixed_visible << 7) | (mixed_visible >> 25)) & _U32
        mixed_visible ^= value
    candidate = (secrets.randbits(32) ^ folded ^ mixed_visible) & _U32 or 1
    while candidate in used:
        candidate = _next_sequence(candidate)
    return candidate


def load_helper(path: str | None = None) -> Any:
    candidates = [path] if path else [
        os.environ.get("MCDMA_RPC_LIBRARY", ""),
        "/usr/local/lib/libmcdma-rpc.so",
        "/usr/lib/libmcdma-rpc.so",
        "/usr/local/lib/libmcdma-rpc.dylib",
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        library = ctypes.CDLL(candidate)
        library.mcdma_rpc_abi.restype = ctypes.c_uint32
        if library.mcdma_rpc_abi() != ABI:
            raise MailboxError(f"{candidate} speaks ABI {library.mcdma_rpc_abi()}, not {ABI}")
        library.mcdma_rpc_wait_word.restype = ctypes.c_uint64
        library.mcdma_rpc_wait_word.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
        ]
        library.mcdma_rpc_store_word.restype = None
        library.mcdma_rpc_store_word.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        return library
    raise MailboxError("libmcdma-rpc is not installed; set MCDMA_RPC_LIBRARY")


def _shm_open(name: str) -> int:
    if sys.platform != "darwin":
        return os.open(f"/dev/shm/mcdma-rpc.{name}", os.O_RDWR)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.shm_open.restype = ctypes.c_int
    libc.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
    fd = libc.shm_open(f"/mcdma-rpc.{name}".encode(), os.O_RDWR, 0)
    if fd < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), f"/mcdma-rpc.{name}")
    return fd


def _read_line(control: socket.socket, limit: int = 256) -> bytes:
    received = b""
    while not received.endswith(b"\n") and len(received) < limit:
        chunk = control.recv(1)
        if not chunk:
            break
        received += chunk
    return received.strip()


class _Mapped:
    def __init__(self, name: str, *, mailbox_path: str | None = None, helper: Any = None) -> None:
        self.name = name
        self._helper = helper or load_helper()
        descriptor = os.open(mailbox_path, os.O_RDWR) if mailbox_path else _shm_open(name)
        try:
            self._map = mmap.mmap(descriptor, os.fstat(descriptor).st_size)
        finally:
            os.close(descriptor)
        self.buffer = memoryview(self._map)
        self._base = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
        self.request_bytes = self._load(SIZES) or DEFAULT_HALF
        self.reply_bytes = self._load(SIZES + 8) or DEFAULT_HALF
        if (self.request_bytes < DEFAULT_HALF or self.reply_bytes < DEFAULT_HALF
                or self.request_bytes + self.reply_bytes != len(self.buffer)):
            self.close()
            raise MailboxError("invalid mailbox half sizes")

    def _load(self, offset: int) -> int:
        return int.from_bytes(self.buffer[offset:offset + 8], "little")

    def _store(self, offset: int, value: int) -> None:
        self._helper.mcdma_rpc_store_word(self._base + offset, value)

    def _wait(self, offset: int, seq: int, equal: bool, timeout_s: float) -> int:
        if timeout_s <= 0:
            raise MailboxError("timeout must be positive")
        return int(self._helper.mcdma_rpc_wait_word(
            self._base + offset, seq, int(equal), 100_000, int(timeout_s * 1e9)
        ))

    def close(self) -> None:
        with suppress(AttributeError, BufferError):
            self.buffer.release()
        with suppress(AttributeError, BufferError):
            self._map.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def link_generation(name: str, *, mailbox_path: str | None = None) -> int:
    """Read the live Mac-side link generation without publishing a request."""

    # Mapping and reading the two control words does not need the native wait
    # helper. Keeping this probe read-only makes it safe for API health checks.
    with _Mapped(name, mailbox_path=mailbox_path, helper=object()) as mapped:
        ready = mapped._load(READY_WORD)
        generation = mapped._load(GENERATION_WORD)
    if ready != 1 or generation <= 0:
        raise MailboxError("MCDMA link is not ready")
    return generation


class ClientMailbox(_Mapped):
    """The Mac/connect end: publish one request, wait for its direct reply."""

    def __init__(self, name: str, *, mailbox_path: str | None = None, helper: Any = None,
                 ready_timeout_s: float = 10.0) -> None:
        super().__init__(name, mailbox_path=mailbox_path, helper=helper)
        deadline = time.monotonic() + ready_timeout_s
        while self._load(READY_WORD) != 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.generation = self._load(GENERATION_WORD)
        if self._load(READY_WORD) != 1 or not self.generation:
            self.close()
            raise MailboxError("MCDMA link did not become ready")
        # Avoid both the last staged request and the last delivered reply.  A
        # stale done word must never satisfy the first call of a new client.
        self._seq = _initial_sequence(
            self._load(0) >> 32,
            self._load(self.request_bytes + READY_WORD) >> 32,
            generation=self.generation,
        )

    @property
    def max_request(self) -> int:
        return self.request_bytes - CTRL

    @property
    def max_reply(self) -> int:
        return self.reply_bytes - CTRL

    def call_consume(
        self,
        payload: bytes | bytearray | memoryview,
        consume: Callable[[memoryview], _ReplyResult],
        timeout_s: float = 30.0,
    ) -> _ReplyResult:
        """Consume one stable reply directly from the mapped reply area.

        The view is valid only for the duration of ``consume``.  Keeping the
        control-word and generation checks around that callback preserves the
        same torn-reply protection as :meth:`call` without first copying a
        large reply into an intermediate ``bytes`` object.
        """

        raw = bytes(payload)
        if not raw or len(raw) > self.max_request:
            raise MailboxError("request is empty or larger than the request half")
        if timeout_s <= 0:
            raise MailboxError("timeout must be positive")
        if self._load(READY_WORD) != 1 or self._load(GENERATION_WORD) != self.generation:
            raise MailboxError("MCDMA link generation changed before the request")
        seq = self._seq
        # Once a request is published its sequence is burned, including on a
        # timeout or malformed reply.  Reusing it can make a late old reply look
        # like the answer to a different payload.
        self._seq = _next_sequence(seq)
        self.buffer[CTRL:CTRL + len(raw)] = raw
        self._store(0, (seq << 32) | len(raw))
        done = self._wait(self.request_bytes + READY_WORD, seq, True, timeout_s)
        if not done:
            raise MailboxError(f"reply timed out for sequence {seq}")
        done_seq, length = done >> 32, done & _U32
        if done_seq != seq:
            raise MailboxError("reply sequence differs from the pending request")
        if not length or length > self.reply_bytes - CTRL:
            raise MailboxError("reply is empty or exceeds the reply half")
        if self._load(READY_WORD) != 1 or self._load(GENERATION_WORD) != self.generation:
            raise MailboxError("MCDMA link generation changed during the request")
        reply = self.buffer[
            self.request_bytes + CTRL:self.request_bytes + CTRL + length
        ].toreadonly()
        try:
            result = consume(reply)
            if self._load(self.request_bytes + READY_WORD) != done:
                raise MailboxError("reply changed while it was copied")
            if self._load(READY_WORD) != 1 or self._load(GENERATION_WORD) != self.generation:
                raise MailboxError("MCDMA link generation changed while the reply was copied")
            return result
        finally:
            reply.release()

    def call(self, payload: bytes | bytearray | memoryview, timeout_s: float = 30.0) -> bytes:
        return self.call_consume(payload, bytes, timeout_s)


class ServiceMailbox(_Mapped):
    """The Linux/listen end: register one service and answer requests."""

    def __init__(self, name: str, *, socket_path: str | None = None, mailbox_path: str | None = None,
                 helper: Any = None) -> None:
        super().__init__(name, mailbox_path=mailbox_path, helper=helper)
        self._control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        path = socket_path or f"/tmp/mcdma-rpcd.{name}.sock"
        try:
            self._control.settimeout(5.0)
            self._control.connect(path)
            self._control.sendall(b"MODE poll\n")
            answer = _read_line(self._control)
            self._control.settimeout(None)
        except OSError as exc:
            self.close()
            raise MailboxError(f"cannot register with mcdma-rpcd at {path}: {exc}") from exc
        if answer != b"OK":
            self.close()
            raise MailboxError(f"mcdma-rpcd refused the service: {answer.decode(errors='replace')}")
        # The Linux/listen mapping does not receive the Mac connector's local
        # ready/generation control words.  The authenticated application frame
        # supplies the generation; the control socket supplies link liveness.
        self.generation: int | None = None
        # A request left before this service registered is deliberately stale.
        # Processing it could mutate model state after its client gave up.
        self._last = self._load(0) >> 32

    @property
    def max_reply(self) -> int:
        return self.reply_bytes - CTRL

    @property
    def alive(self) -> bool:
        try:
            readable, _, _ = select.select([self._control], [], [], 0)
        except (OSError, ValueError):
            return False
        return not readable

    def _assert_link(self) -> None:
        if not self.alive:
            raise MailboxError("mcdma-rpcd control connection closed")

    def bind_generation(self, generation: int) -> None:
        """Bind the authenticated generation once and reject cross-link replay."""

        if generation <= 0:
            raise MailboxError("frame generation must be positive")
        if self.generation is None:
            self.generation = generation
        elif self.generation != generation:
            raise MailboxError("frame belongs to a different MCDMA generation")

    def next_request(self, timeout_s: float) -> tuple[int, bytes] | None:
        if timeout_s <= 0:
            raise MailboxError("timeout must be positive")
        deadline = time.monotonic() + timeout_s
        staged = 0
        while not staged:
            self._assert_link()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            # Recheck the daemon control socket and generation at least once a
            # second instead of sleeping through a long inference timeout.
            staged = self._wait(0, self._last, False, min(remaining, 1.0))
        while True:
            self._assert_link()
            seq, length = staged >> 32, staged & _U32
            # Consume even a malformed word so one corrupt request cannot wedge
            # the service forever. It is rejected without reading or replying.
            if not length or length > self.request_bytes - CTRL:
                self._last = seq
                raise MailboxError("request is empty or exceeds the request half")
            request = bytes(self.buffer[CTRL:CTRL + length])
            self._assert_link()
            observed = self._load(0)
            if observed == staged:
                self._last = seq
                return seq, request
            # The client changed the control word while this service copied.
            # Discard the bytes and consume only a later stable word.  This
            # reread detects a concurrent rewrite; the authenticated full-frame
            # digest is what proves payload integrity.
            self._last = seq
            staged = 0
            while not staged:
                self._assert_link()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                staged = self._wait(0, self._last, False, min(remaining, 1.0))

    def reply(self, seq: int, payload: bytes | bytearray | memoryview) -> None:
        raw = bytes(payload)
        if not raw or len(raw) > self.max_reply:
            raise MailboxError("reply is empty or exceeds the reply half")
        self.reply_area()[:len(raw)] = raw
        self.publish(seq, len(raw))

    def reply_area(self) -> memoryview:
        """Return the reply payload area for zero-copy frame construction."""

        self._assert_link()
        start = self.request_bytes + CTRL
        return self.buffer[start:start + self.max_reply]

    def publish(self, seq: int, length: int) -> None:
        """Publish bytes already written to ``reply_area``."""

        if not 0 < length <= self.max_reply:
            raise MailboxError("reply is empty or exceeds the reply half")
        self._assert_link()
        self._store(self.request_bytes + STAGED_WORD, ((seq & _U32) << 32) | length)
        self._assert_link()

    def wait_reply_accepted(self, seq: int, length: int, timeout_s: float) -> None:
        """Keep a terminating service attached until rpcd accepts its reply."""

        if not 0 < length <= self.max_reply:
            raise MailboxError("reply acknowledgement length is invalid")
        self._assert_link()
        accepted = self._wait(self.request_bytes, seq, True, timeout_s)
        if not accepted:
            raise MailboxError(f"reply acknowledgement timed out for sequence {seq}")
        accepted_seq, accepted_length = accepted >> 32, accepted & _U32
        if accepted_seq != seq or accepted_length != length:
            raise MailboxError("reply acknowledgement differs from the staged reply")
        self._assert_link()

    def close(self) -> None:
        with suppress(AttributeError, OSError):
            self._control.close()
        super().close()
