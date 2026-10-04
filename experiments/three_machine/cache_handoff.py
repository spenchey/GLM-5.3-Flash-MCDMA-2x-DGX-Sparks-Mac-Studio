"""Verified GLM prompt-cache handoff from two CUDA ranks to one MLX host.

The low-level MCDMA mailbox is deliberately unaware of model tensors.  This
module adds a versioned, fail-closed envelope above it.  It has no Torch or MLX
imports so its ordering and corruption checks can be tested on every host.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = 2
CHUNK_MAGIC = b"TFCH"
FRAME_MAGIC = b"TFFR"
MANIFEST_MAGIC = b"TFMF"
ACK_MAGIC = b"TFAK"
ERROR_MAGIC = b"TFER"
MAX_HEADER_BYTES = 64 * 1024
# GLM's recurrent state is exactly 4 MiB per KDA layer. Requesting a 4 MiB
# chunk lets the client use the largest payload that fits the live mailbox; a
# proven 4 MiB reply half carries each tensor in two calls rather than four.
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
MAX_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_FRAME_BYTES = 64 * 1024 * 1024
# MCDMA supports reply halves through 256 MiB. Keep the conservative 64 MiB
# default, but permit a complete 148 MiB GLM cache to cross in one call when
# both peers reserve a 256 MiB reply half.
MAX_FRAME_BYTES = 256 * 1024 * 1024
_HEADER_LENGTH = struct.Struct(">I")
BUNDLE_SCHEMA = 1
BUNDLE_MANIFEST = "manifest.json"
BUNDLE_INDEX = "bundle.json"


class HandoffError(RuntimeError):
    """A cache cannot be trusted or cannot be represented by this protocol."""


@dataclass(frozen=True)
class Request:
    operation: str
    transfer_id: str | None = None
    tokens: tuple[int, ...] = ()
    prompt_sha256: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    tensor: str | None = None
    offset: int = 0
    length: int = DEFAULT_CHUNK_BYTES


def encode_request(request: Request) -> bytes:
    if request.operation not in {"prepare", "chunk", "frame", "release", "shutdown"}:
        raise HandoffError(f"unsupported cache operation {request.operation!r}")
    value: dict[str, object] = {"protocol": PROTOCOL_VERSION, "operation": request.operation}
    if request.operation == "prepare":
        if len(request.tokens) < 2 or any(not 0 <= int(token) <= 0xFFFFFFFF for token in request.tokens):
            raise HandoffError("cache preparation needs at least two uint32 token IDs")
        digest = prompt_digest(request.tokens)
        if request.prompt_sha256 not in (None, digest):
            raise HandoffError("prepare request prompt digest does not match its tokens")
        if not request.model_id or not request.model_revision:
            raise HandoffError("prepare request model identity is incomplete")
        value.update(tokens=list(request.tokens), prompt_sha256=digest,
                     model_id=request.model_id, model_revision=request.model_revision)
    elif request.operation == "chunk":
        if not request.transfer_id or not request.tensor or request.offset < 0:
            raise HandoffError("chunk request identity or range is incomplete")
        if not 0 < request.length <= MAX_CHUNK_BYTES:
            raise HandoffError("chunk request exceeds the safe mailbox payload")
        value.update(transfer_id=request.transfer_id, tensor=request.tensor,
                     offset=int(request.offset), length=int(request.length))
    elif request.operation == "frame":
        if not request.transfer_id or request.tensor is not None or request.offset < 0:
            raise HandoffError("frame request identity or range is incomplete")
        if not 0 < request.length <= MAX_FRAME_BYTES:
            raise HandoffError("frame request exceeds the safe mailbox payload")
        value.update(transfer_id=request.transfer_id, offset=int(request.offset),
                     length=int(request.length))
    elif request.operation == "release":
        if not request.transfer_id:
            raise HandoffError("release request has no transfer identity")
        value["transfer_id"] = request.transfer_id
    return _canonical_json(value)


def decode_request(raw: bytes) -> Request:
    try:
        value = json.loads(raw.decode("ascii"))
        if int(value.pop("protocol")) != PROTOCOL_VERSION:
            raise HandoffError("cache request protocol is unsupported")
        operation = str(value.pop("operation"))
        if operation == "prepare":
            tokens = tuple(int(token) for token in value.pop("tokens"))
            request = Request(operation=operation, tokens=tokens, **value)
        elif operation == "chunk":
            request = Request(operation=operation, tensor=value.pop("tensor"), **value)
        else:
            request = Request(operation=operation, **value)
        # Re-encoding is the single validation path for both local and remote requests.
        encode_request(request)
        return request
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HandoffError(f"invalid cache request: {exc}") from exc


def pack_manifest(manifest: "HandoffManifest") -> bytes:
    return MANIFEST_MAGIC + manifest.to_bytes()


def unpack_manifest(raw: bytes) -> "HandoffManifest":
    if not raw.startswith(MANIFEST_MAGIC):
        raise HandoffError("cache response is not a manifest")
    return HandoffManifest.from_bytes(raw[len(MANIFEST_MAGIC):])


def pack_ack(operation: str, transfer_id: str | None = None) -> bytes:
    return ACK_MAGIC + _canonical_json({"protocol": PROTOCOL_VERSION, "operation": operation,
                                        "transfer_id": transfer_id})


def unpack_ack(raw: bytes, operation: str) -> str | None:
    if raw.startswith(ERROR_MAGIC):
        raise HandoffError(raw[len(ERROR_MAGIC):].decode("utf-8", errors="replace"))
    if not raw.startswith(ACK_MAGIC):
        raise HandoffError("cache response is not an acknowledgement")
    try:
        value = json.loads(raw[len(ACK_MAGIC):].decode("ascii"))
        if int(value["protocol"]) != PROTOCOL_VERSION or value["operation"] != operation:
            raise HandoffError("cache acknowledgement does not match the request")
        return None if value.get("transfer_id") is None else str(value["transfer_id"])
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HandoffError(f"invalid cache acknowledgement: {exc}") from exc


def pack_error(exc: BaseException) -> bytes:
    message = f"{type(exc).__name__}: {exc}".encode("utf-8", errors="replace")
    return ERROR_MAGIC + message[:4096]


def raise_if_error(raw: bytes | bytearray | memoryview) -> None:
    view = memoryview(raw).cast("B")
    if bytes(view[:len(ERROR_MAGIC)]) == ERROR_MAGIC:
        raise HandoffError(bytes(view[len(ERROR_MAGIC):]).decode("utf-8", errors="replace"))


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def prompt_digest(tokens: Sequence[int]) -> str:
    """Stable hash of signed-independent uint32 token IDs."""

    ids = np.asarray(tokens, dtype=">u4")
    return hashlib.sha256(ids.tobytes()).hexdigest()


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    sha256: str

    @classmethod
    def from_array(cls, name: str, value: np.ndarray, *, dtype: str | None = None) -> "TensorSpec":
        array = np.ascontiguousarray(value)
        raw = memoryview(array).cast("B")
        return cls(
            name=name,
            dtype=dtype or array.dtype.str,
            shape=tuple(int(n) for n in array.shape),
            nbytes=raw.nbytes,
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TensorSpec":
        try:
            return cls(
                name=str(value["name"]),
                dtype=str(value["dtype"]),
                shape=tuple(int(n) for n in value["shape"]),  # type: ignore[arg-type]
                nbytes=int(value["nbytes"]),
                sha256=str(value["sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HandoffError(f"invalid tensor specification: {exc}") from exc

    def validate(self) -> None:
        if not self.name or any(n < 0 for n in self.shape) or self.nbytes < 0:
            raise HandoffError(f"invalid tensor specification for {self.name!r}")
        if len(self.sha256) != 64:
            raise HandoffError(f"invalid tensor digest for {self.name!r}")


@dataclass(frozen=True)
class HandoffManifest:
    protocol: int
    transfer_id: str
    model_id: str
    model_revision: str
    prompt_sha256: str
    cached_tokens: int
    mtp_cached_tokens: int
    spark_first_token: int | None
    world_size: int
    long_context: bool
    tensors: tuple[TensorSpec, ...]

    @classmethod
    def build(
        cls,
        *,
        model_id: str,
        model_revision: str,
        prompt_sha256: str,
        cached_tokens: int,
        tensors: Iterable[TensorSpec],
        mtp_cached_tokens: int | None = None,
        spark_first_token: int | None = None,
        world_size: int = 2,
        long_context: bool = False,
    ) -> "HandoffManifest":
        ordered = tuple(sorted(tensors, key=lambda item: item.name))
        mtp_tokens = int(cached_tokens if mtp_cached_tokens is None else mtp_cached_tokens)
        body = {
            "protocol": PROTOCOL_VERSION,
            "model_id": model_id,
            "model_revision": model_revision,
            "prompt_sha256": prompt_sha256,
            "cached_tokens": int(cached_tokens),
            "mtp_cached_tokens": mtp_tokens,
            "spark_first_token": (
                None if spark_first_token is None else int(spark_first_token)
            ),
            "world_size": int(world_size),
            "long_context": bool(long_context),
            "tensors": [asdict(item) for item in ordered],
        }
        transfer_id = hashlib.sha256(_canonical_json(body)).hexdigest()
        manifest = cls(
            protocol=PROTOCOL_VERSION,
            transfer_id=transfer_id,
            model_id=model_id,
            model_revision=model_revision,
            prompt_sha256=prompt_sha256,
            cached_tokens=int(cached_tokens),
            mtp_cached_tokens=mtp_tokens,
            spark_first_token=(
                None if spark_first_token is None else int(spark_first_token)
            ),
            world_size=int(world_size),
            long_context=bool(long_context),
            tensors=ordered,
        )
        manifest.validate()
        return manifest

    def _identity_body(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_sha256": self.prompt_sha256,
            "cached_tokens": self.cached_tokens,
            "mtp_cached_tokens": self.mtp_cached_tokens,
            "spark_first_token": self.spark_first_token,
            "world_size": self.world_size,
            "long_context": self.long_context,
            "tensors": [asdict(item) for item in self.tensors],
        }

    def validate(self) -> None:
        if self.protocol != PROTOCOL_VERSION:
            raise HandoffError(f"unsupported cache protocol {self.protocol}")
        if not self.model_id or not self.model_revision:
            raise HandoffError("model identity is incomplete")
        if len(self.prompt_sha256) != 64:
            raise HandoffError("prompt digest is invalid")
        if self.cached_tokens <= 0 or self.world_size != 2:
            raise HandoffError("cache position or Spark world size is invalid")
        if not 0 <= self.mtp_cached_tokens <= self.cached_tokens:
            raise HandoffError("MTP cache position is invalid")
        if self.spark_first_token is not None and not 0 <= self.spark_first_token <= 0xFFFFFFFF:
            raise HandoffError("Spark first token is not uint32")
        if self.long_context:
            raise HandoffError("long-context index state is not supported by cache handoff v2")
        names = [item.name for item in self.tensors]
        if not names or len(names) != len(set(names)) or names != sorted(names):
            raise HandoffError("tensor names must be nonempty, unique, and sorted")
        for item in self.tensors:
            item.validate()
        want = hashlib.sha256(_canonical_json(self._identity_body())).hexdigest()
        if self.transfer_id != want:
            raise HandoffError("manifest identity digest does not match its contents")

    def require_identity(
        self,
        *,
        model_id: str,
        model_revision: str,
        prompt_sha256: str,
        cached_tokens: int,
        mtp_cached_tokens: int | None = None,
    ) -> None:
        self.validate()
        expected_mtp = int(cached_tokens if mtp_cached_tokens is None else mtp_cached_tokens)
        actual = (
            self.model_id, self.model_revision, self.prompt_sha256,
            self.cached_tokens, self.mtp_cached_tokens,
        )
        expected = (
            model_id, model_revision, prompt_sha256,
            int(cached_tokens), expected_mtp,
        )
        if actual != expected:
            raise HandoffError("cache model, prompt, or position differs from the decode request")

    def to_bytes(self) -> bytes:
        self.validate()
        return _canonical_json({**self._identity_body(), "transfer_id": self.transfer_id})

    @classmethod
    def from_bytes(cls, raw: bytes) -> "HandoffManifest":
        try:
            value = json.loads(raw.decode("ascii"))
            tensors = tuple(TensorSpec.from_dict(item) for item in value.pop("tensors"))
            manifest = cls(tensors=tensors, **value)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise HandoffError(f"invalid cache manifest: {exc}") from exc
        manifest.validate()
        return manifest

    def tensor(self, name: str) -> TensorSpec:
        for item in self.tensors:
            if item.name == name:
                return item
        raise HandoffError(f"manifest has no tensor {name!r}")


class TensorArchive:
    """Immutable contiguous tensor bytes and their signed manifest."""

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        model_id: str,
        model_revision: str,
        prompt_sha256: str,
        cached_tokens: int,
        dtype_names: Mapping[str, str] | None = None,
        mtp_cached_tokens: int | None = None,
        spark_first_token: int | None = None,
    ) -> None:
        if not arrays:
            raise HandoffError("a handoff must contain at least one tensor")
        # Keep one immutable contiguous snapshot.  The previous implementation
        # retained a full bytes copy and made another temporary bytes copy for
        # every tensor digest, which copied a typical 148 MiB GLM cache twice.
        self._arrays = {
            name: np.array(value, copy=True, order="C")
            for name, value in arrays.items()
        }
        for value in self._arrays.values():
            value.flags.writeable = False
        self._raw = {
            name: memoryview(value).cast("B").toreadonly()
            for name, value in self._arrays.items()
        }
        aliases = {} if dtype_names is None else dict(dtype_names)
        specs = [
            TensorSpec.from_array(name, value, dtype=aliases.get(name))
            for name, value in self._arrays.items()
        ]
        self.manifest = HandoffManifest.build(
            model_id=model_id,
            model_revision=model_revision,
            prompt_sha256=prompt_sha256,
            cached_tokens=cached_tokens,
            mtp_cached_tokens=mtp_cached_tokens,
            spark_first_token=spark_first_token,
            tensors=specs,
        )

    def bytes_for(self, name: str) -> memoryview:
        self.manifest.tensor(name)
        return self._raw[name]

    def chunk(self, name: str, offset: int, length: int = DEFAULT_CHUNK_BYTES) -> bytes:
        spec = self.manifest.tensor(name)
        if offset < 0 or length <= 0 or offset >= spec.nbytes:
            raise HandoffError("chunk range is outside the tensor")
        raw = self.bytes_for(name)[offset:offset + length]
        return pack_chunk(self.manifest.transfer_id, spec, offset, raw)

    def frame(self, offset: int, length: int = DEFAULT_FRAME_BYTES) -> bytes:
        return pack_frame(self, offset, length)

    def write_frame(self, target: memoryview, offset: int,
                    length: int = DEFAULT_FRAME_BYTES) -> int:
        return write_frame(self, target, offset, length)


def persist_archive(archive: TensorArchive, root: Path) -> Path:
    """Atomically persist one verified TensorFold cache for GPU-free replay."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        raise HandoffError("cache bundle root must be a real directory")
    # A prompt can legitimately produce a different on-disk shape when the
    # handoff protocol changes.  Keep each protocol in a separate immutable
    # bundle instead of trying to parse or overwrite a prior protocol's cache.
    target = root / f"v{PROTOCOL_VERSION}-{archive.manifest.prompt_sha256}"
    incoming = Path(tempfile.mkdtemp(prefix=".incoming-", dir=root))
    try:
        manifest_bytes = archive.manifest.to_bytes()
        (incoming / BUNDLE_MANIFEST).write_bytes(manifest_bytes)
        files: list[dict[str, object]] = []
        for index, spec in enumerate(archive.manifest.tensors):
            filename = f"{index:03d}.bin"
            raw = archive.bytes_for(spec.name)
            if len(raw) != spec.nbytes or hashlib.sha256(raw).hexdigest() != spec.sha256:
                raise HandoffError(f"cache tensor {spec.name!r} changed before persistence")
            (incoming / filename).write_bytes(raw)
            files.append({"name": spec.name, "file": filename})
        index_value = {
            "schema": BUNDLE_SCHEMA,
            "transfer_id": archive.manifest.transfer_id,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "files": files,
        }
        (incoming / BUNDLE_INDEX).write_bytes(_canonical_json(index_value))
        if target.exists():
            existing = StoredTensorArchive(target)
            try:
                if existing.manifest != archive.manifest:
                    raise HandoffError("an existing cache bundle has a different manifest")
            finally:
                existing.close()
            shutil.rmtree(incoming)
            return target
        os.replace(incoming, target)
        return target
    except BaseException:
        if incoming.exists():
            shutil.rmtree(incoming)
        raise


class StoredTensorArchive:
    """Read-only on-disk cache archive served without loading the CUDA model."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_dir() or self.path.is_symlink():
            raise HandoffError("cache bundle must be a real directory")
        manifest_path = self.path / BUNDLE_MANIFEST
        index_path = self.path / BUNDLE_INDEX
        if manifest_path.is_symlink() or index_path.is_symlink():
            raise HandoffError("cache bundle metadata must not be symlinks")
        manifest_bytes = manifest_path.read_bytes()
        self.manifest = HandoffManifest.from_bytes(manifest_bytes)
        try:
            value = json.loads(index_path.read_text(encoding="ascii"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HandoffError(f"cache bundle index is invalid: {exc}") from exc
        expected = {"schema", "transfer_id", "manifest_sha256", "files"}
        if not isinstance(value, dict) or set(value) != expected:
            raise HandoffError("cache bundle index has unexpected fields")
        if (
            value["schema"] != BUNDLE_SCHEMA
            or value["transfer_id"] != self.manifest.transfer_id
            or value["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()
        ):
            raise HandoffError("cache bundle identity does not match its manifest")
        files = value["files"]
        if not isinstance(files, list) or len(files) != len(self.manifest.tensors):
            raise HandoffError("cache bundle file index is incomplete")
        self._raw: dict[str, bytes] = {}
        for entry, spec in zip(files, self.manifest.tensors, strict=True):
            if not isinstance(entry, dict) or set(entry) != {"name", "file"}:
                raise HandoffError("cache bundle file entry is invalid")
            expected_file = f"{len(self._raw):03d}.bin"
            if entry != {"name": spec.name, "file": expected_file}:
                raise HandoffError("cache bundle tensor ordering changed")
            tensor_path = self.path / expected_file
            if not tensor_path.is_file() or tensor_path.is_symlink():
                raise HandoffError(f"cache bundle tensor {spec.name!r} is missing")
            raw = tensor_path.read_bytes()
            if len(raw) != spec.nbytes or hashlib.sha256(raw).hexdigest() != spec.sha256:
                raise HandoffError(f"cache bundle tensor {spec.name!r} failed verification")
            self._raw[spec.name] = raw

    def close(self) -> None:
        self._raw.clear()

    def bytes_for(self, name: str) -> bytes:
        self.manifest.tensor(name)
        return self._raw[name]

    def chunk(self, name: str, offset: int, length: int = DEFAULT_CHUNK_BYTES) -> bytes:
        spec = self.manifest.tensor(name)
        if offset < 0 or length <= 0 or offset >= spec.nbytes:
            raise HandoffError("chunk range is outside the tensor")
        raw = self.bytes_for(name)[offset:offset + length]
        return pack_chunk(self.manifest.transfer_id, spec, offset, raw)

    def frame(self, offset: int, length: int = DEFAULT_FRAME_BYTES) -> bytes:
        return pack_frame(self, offset, length)

    def write_frame(self, target: memoryview, offset: int,
                    length: int = DEFAULT_FRAME_BYTES) -> int:
        return write_frame(self, target, offset, length)


@dataclass(frozen=True)
class FrameSegment:
    name: str
    offset: int
    length: int


@dataclass(frozen=True)
class CacheFrame:
    transfer_id: str
    offset: int
    total: int
    segments: tuple[FrameSegment, ...]
    payload: memoryview


def archive_nbytes(manifest: HandoffManifest) -> int:
    return sum(item.nbytes for item in manifest.tensors)


def frame_ranges(total: int, length: int = DEFAULT_FRAME_BYTES) -> tuple[tuple[int, int], ...]:
    """Return bounded archive ranges without allocating tensor storage."""

    if total <= 0 or not 0 < length <= MAX_FRAME_BYTES:
        raise HandoffError("cache frame plan is outside the safe payload range")
    return tuple(
        (offset, min(length, total - offset))
        for offset in range(0, total, length)
    )


def _frame_segments(manifest: HandoffManifest, offset: int,
                    length: int) -> tuple[FrameSegment, ...]:
    total = archive_nbytes(manifest)
    if offset < 0 or offset >= total or length <= 0 or length > MAX_FRAME_BYTES:
        raise HandoffError("cache frame range is outside the archive")
    remaining = min(length, total - offset)
    cursor = 0
    segments: list[FrameSegment] = []
    for spec in manifest.tensors:
        end = cursor + spec.nbytes
        if offset < end and remaining:
            tensor_offset = max(0, offset - cursor)
            count = min(remaining, spec.nbytes - tensor_offset)
            segments.append(FrameSegment(spec.name, tensor_offset, count))
            offset += count
            remaining -= count
        cursor = end
        if not remaining:
            break
    if remaining or not segments:
        raise HandoffError("cache frame range could not be mapped to tensors")
    return tuple(segments)


def _frame_header(archive: object, offset: int, length: int) -> tuple[bytes, tuple[FrameSegment, ...]]:
    manifest = archive.manifest  # type: ignore[attr-defined]
    segments = _frame_segments(manifest, offset, length)
    digest = hashlib.sha256()
    for segment in segments:
        raw = archive.bytes_for(segment.name)  # type: ignore[attr-defined]
        digest.update(memoryview(raw)[segment.offset:segment.offset + segment.length])
    header = _canonical_json({
        "protocol": PROTOCOL_VERSION,
        "transfer_id": manifest.transfer_id,
        "offset": int(offset),
        "total": archive_nbytes(manifest),
        "payload_sha256": digest.hexdigest(),
        "segments": [asdict(segment) for segment in segments],
    })
    if len(header) > MAX_HEADER_BYTES:
        raise HandoffError("cache frame header is too large")
    return header, segments


def write_frame(archive: object, target: memoryview, offset: int,
                length: int = DEFAULT_FRAME_BYTES) -> int:
    """Write one authenticated multi-tensor frame directly into reply storage."""

    header, segments = _frame_header(archive, offset, length)
    payload_bytes = sum(segment.length for segment in segments)
    total_bytes = len(FRAME_MAGIC) + _HEADER_LENGTH.size + len(header) + payload_bytes
    area = memoryview(target)
    if area.readonly or area.nbytes < total_bytes:
        raise HandoffError("cache frame does not fit the reply area")
    cursor = 0
    area[cursor:cursor + len(FRAME_MAGIC)] = FRAME_MAGIC
    cursor += len(FRAME_MAGIC)
    area[cursor:cursor + _HEADER_LENGTH.size] = _HEADER_LENGTH.pack(len(header))
    cursor += _HEADER_LENGTH.size
    area[cursor:cursor + len(header)] = header
    cursor += len(header)
    for segment in segments:
        raw = archive.bytes_for(segment.name)  # type: ignore[attr-defined]
        part = memoryview(raw)[segment.offset:segment.offset + segment.length]
        area[cursor:cursor + segment.length] = part
        cursor += segment.length
    return cursor


def pack_frame(archive: object, offset: int,
               length: int = DEFAULT_FRAME_BYTES) -> bytes:
    header, segments = _frame_header(archive, offset, length)
    size = len(FRAME_MAGIC) + _HEADER_LENGTH.size + len(header)
    size += sum(segment.length for segment in segments)
    raw = bytearray(size)
    written = write_frame(archive, memoryview(raw), offset, length)
    if written != size:
        raise HandoffError("cache frame size changed while it was packed")
    return bytes(raw)


def unpack_frame(raw: bytes | bytearray | memoryview) -> CacheFrame:
    view = memoryview(raw).cast("B")
    prefix = len(FRAME_MAGIC) + _HEADER_LENGTH.size
    if view.nbytes <= prefix or bytes(view[:len(FRAME_MAGIC)]) != FRAME_MAGIC:
        raise HandoffError("cache frame magic is invalid")
    header_len = _HEADER_LENGTH.unpack_from(view, len(FRAME_MAGIC))[0]
    if not 0 < header_len <= MAX_HEADER_BYTES or view.nbytes <= prefix + header_len:
        raise HandoffError("cache frame header length is invalid")
    try:
        header = json.loads(bytes(view[prefix:prefix + header_len]).decode("ascii"))
        if not isinstance(header, dict) or set(header) != {
            "protocol", "transfer_id", "offset", "total", "payload_sha256", "segments"
        }:
            raise HandoffError("cache frame header has unexpected fields")
        if int(header["protocol"]) != PROTOCOL_VERSION:
            raise HandoffError("cache frame protocol is unsupported")
        payload = view[prefix + header_len:].toreadonly()
        if not payload or len(payload) > MAX_FRAME_BYTES:
            raise HandoffError("cache frame payload size is invalid")
        if hashlib.sha256(payload).hexdigest() != str(header["payload_sha256"]):
            raise HandoffError("cache frame digest mismatch")
        segment_values = header["segments"]
        if not isinstance(segment_values, list) or not segment_values:
            raise HandoffError("cache frame has no tensor segments")
        segments = tuple(FrameSegment(
            name=str(item["name"]), offset=int(item["offset"]), length=int(item["length"])
        ) for item in segment_values)
        if any(segment.offset < 0 or segment.length <= 0 for segment in segments):
            raise HandoffError("cache frame tensor segment is invalid")
        if sum(segment.length for segment in segments) != len(payload):
            raise HandoffError("cache frame segments do not cover its payload")
        return CacheFrame(
            transfer_id=str(header["transfer_id"]),
            offset=int(header["offset"]),
            total=int(header["total"]),
            segments=segments,
            payload=payload,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HandoffError(f"invalid cache frame: {exc}") from exc


def reply_frame(mailbox: object, sequence: int, archive: object,
                offset: int, length: int) -> None:
    """Use an in-place reply area when available, or a compatible bytes reply."""

    reply_area = getattr(mailbox, "reply_area", None)
    publish = getattr(mailbox, "publish", None)
    if callable(reply_area) and callable(publish):
        written = archive.write_frame(reply_area(), offset, length)  # type: ignore[attr-defined]
        publish(sequence, written)
        return
    mailbox.reply(sequence, archive.frame(offset, length))  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Chunk:
    transfer_id: str
    name: str
    offset: int
    total: int
    payload: bytes


def pack_chunk(transfer_id: str, spec: TensorSpec, offset: int, payload: bytes) -> bytes:
    if not payload or offset < 0 or offset + len(payload) > spec.nbytes:
        raise HandoffError("invalid cache chunk range")
    header = _canonical_json({
        "protocol": PROTOCOL_VERSION,
        "transfer_id": transfer_id,
        "name": spec.name,
        "offset": int(offset),
        "total": spec.nbytes,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    })
    if len(header) > MAX_HEADER_BYTES:
        raise HandoffError("cache chunk header is too large")
    return CHUNK_MAGIC + _HEADER_LENGTH.pack(len(header)) + header + payload


def unpack_chunk(raw: bytes) -> Chunk:
    prefix = len(CHUNK_MAGIC) + _HEADER_LENGTH.size
    if len(raw) <= prefix or raw[: len(CHUNK_MAGIC)] != CHUNK_MAGIC:
        raise HandoffError("cache chunk magic is invalid")
    header_len = _HEADER_LENGTH.unpack_from(raw, len(CHUNK_MAGIC))[0]
    if not 0 < header_len <= MAX_HEADER_BYTES or len(raw) <= prefix + header_len:
        raise HandoffError("cache chunk header length is invalid")
    try:
        header = json.loads(raw[prefix:prefix + header_len].decode("ascii"))
        if int(header["protocol"]) != PROTOCOL_VERSION:
            raise HandoffError("cache chunk protocol is unsupported")
        payload = raw[prefix + header_len:]
        if hashlib.sha256(payload).hexdigest() != str(header["payload_sha256"]):
            raise HandoffError("cache chunk digest mismatch")
        return Chunk(
            transfer_id=str(header["transfer_id"]),
            name=str(header["name"]),
            offset=int(header["offset"]),
            total=int(header["total"]),
            payload=payload,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HandoffError(f"invalid cache chunk: {exc}") from exc


class ChunkAssembler:
    """Sequential assembler that verifies every chunk and the final tensor."""

    def __init__(self, manifest: HandoffManifest) -> None:
        manifest.validate()
        self.manifest = manifest
        self._data = {item.name: bytearray(item.nbytes) for item in manifest.tensors}
        self._written = {item.name: 0 for item in manifest.tensors}
        self._verified: set[str] = set()
        self._frame_offset = 0

    def accept(self, raw: bytes) -> bool:
        chunk = unpack_chunk(raw)
        if chunk.transfer_id != self.manifest.transfer_id:
            raise HandoffError("chunk belongs to a different handoff")
        spec = self.manifest.tensor(chunk.name)
        data = self._data[chunk.name]
        written = self._written[chunk.name]
        if chunk.total != spec.nbytes or chunk.offset != written:
            raise HandoffError("chunk size or order differs from the manifest")
        if written + len(chunk.payload) > spec.nbytes:
            raise HandoffError("chunk exceeds the tensor size")
        data[written:written + len(chunk.payload)] = chunk.payload
        self._written[chunk.name] = written + len(chunk.payload)
        self._verified.discard(chunk.name)
        return self._written[chunk.name] == spec.nbytes

    def accept_frame(self, raw: bytes | bytearray | memoryview) -> bool:
        """Verify and append one sequential frame spanning any tensor boundaries."""

        frame = unpack_frame(raw)
        total = archive_nbytes(self.manifest)
        if frame.transfer_id != self.manifest.transfer_id:
            raise HandoffError("frame belongs to a different handoff")
        if frame.total != total or frame.offset != self._frame_offset:
            raise HandoffError("frame size or order differs from the manifest")
        expected = _frame_segments(self.manifest, frame.offset, len(frame.payload))
        if frame.segments != expected:
            raise HandoffError("frame tensor map differs from the manifest")
        cursor = 0
        for segment in frame.segments:
            data = self._data[segment.name]
            written = self._written[segment.name]
            if segment.offset != written:
                raise HandoffError("frame tensor order differs from the manifest")
            end = cursor + segment.length
            data[written:written + segment.length] = frame.payload[cursor:end]
            self._written[segment.name] = written + segment.length
            self._verified.discard(segment.name)
            cursor = end
        self._frame_offset += len(frame.payload)
        return self._frame_offset == total

    def verify_complete(self) -> None:
        """Require every tensor to be complete and match its manifest digest."""

        for spec in self.manifest.tensors:
            self.buffer_for(spec.name)

    def buffer_for(self, name: str) -> memoryview:
        """Return verified tensor storage without allocating another full copy."""

        spec = self.manifest.tensor(name)
        raw = memoryview(self._data[name]).toreadonly()
        if self._written[name] != spec.nbytes:
            raise HandoffError(f"tensor {name!r} is incomplete")
        if name not in self._verified:
            if hashlib.sha256(raw).hexdigest() != spec.sha256:
                raise HandoffError(f"tensor {name!r} digest mismatch")
            self._verified.add(name)
        return raw

    def bytes_for(self, name: str) -> bytes:
        """Compatibility copy for callers that explicitly require bytes."""

        return self.buffer_for(name).tobytes()

    def array(self, name: str) -> np.ndarray:
        spec = self.manifest.tensor(name)
        try:
            dtype = np.dtype(spec.dtype)
        except TypeError as exc:
            raise HandoffError(f"tensor {name!r} uses a host-specific dtype {spec.dtype!r}") from exc
        array = np.frombuffer(self.buffer_for(name), dtype=dtype)
        try:
            return array.reshape(spec.shape)
        except ValueError as exc:
            raise HandoffError(f"tensor {name!r} byte count does not match its shape") from exc


def select_current_recurrent(rec: np.ndarray, parity: Sequence[int]) -> np.ndarray:
    """Select each KDA layer's committed recurrent buffer from one CUDA rank."""

    value = np.asarray(rec)
    if value.ndim != 5 or value.shape[0] != 2 or len(parity) != value.shape[1]:
        raise HandoffError("rank recurrent state has an unexpected shape or parity list")
    if any(int(bit) not in (0, 1) for bit in parity):
        raise HandoffError("rank recurrent parity must contain only zero or one")
    return np.ascontiguousarray(np.stack([value[int(bit), layer] for layer, bit in enumerate(parity)]))


def merge_kda_recurrent(
    rank_states: Sequence[np.ndarray], rank_parities: Sequence[Sequence[int]]
) -> np.ndarray:
    """Join head-sharded current KDA states as [layers, global_heads, 128, 128]."""

    if len(rank_states) != 2 or len(rank_parities) != 2:
        raise HandoffError("exactly two Spark recurrent shards are required")
    selected = [select_current_recurrent(state, parity) for state, parity in zip(rank_states, rank_parities)]
    base = selected[0]
    if any(value.shape[0] != base.shape[0] or value.shape[2:] != base.shape[2:] for value in selected[1:]):
        raise HandoffError("Spark recurrent shards have incompatible shapes")
    return np.ascontiguousarray(np.concatenate(selected, axis=1))


def merge_kda_conv(rank_states: Sequence[np.ndarray], *, head_dim: int = 128) -> np.ndarray:
    """Join conv shards without interleaving q/k/v groups incorrectly.

    CUDA stores each rank as ``[q-local, k-local, v-local]``.  A plain concat
    would produce ``q0,k0,v0,q1,k1,v1``; MLX needs
    ``q0,q1,k0,k1,v0,v1``.
    """

    if len(rank_states) != 2 or head_dim <= 0:
        raise HandoffError("exactly two Spark convolution shards are required")
    values = [np.asarray(value) for value in rank_states]
    base = values[0]
    if base.ndim not in (2, 3):
        raise HandoffError("rank convolution state must have two or three dimensions")
    reshaped = []
    for value in values:
        if value.shape[:-1] != base.shape[:-1] or value.shape[-1] % (3 * head_dim):
            raise HandoffError("Spark convolution shards have incompatible shapes")
        local_heads = value.shape[-1] // (3 * head_dim)
        reshaped.append(value.reshape(*value.shape[:-1], 3, local_heads, head_dim))
    merged = np.concatenate(reshaped, axis=-2)
    return np.ascontiguousarray(merged.reshape(*base.shape[:-1], -1))


def require_replicated(rank_states: Sequence[np.ndarray], *, name: str) -> np.ndarray:
    """Return one copy only when both ranks hold byte-identical replicated state."""

    if len(rank_states) != 2:
        raise HandoffError(f"exactly two copies of {name} are required")
    first, second = (np.ascontiguousarray(value) for value in rank_states)
    if first.shape != second.shape or first.dtype != second.dtype or first.tobytes() != second.tobytes():
        raise HandoffError(f"replicated tensor {name!r} differs between Sparks")
    return first.copy()
