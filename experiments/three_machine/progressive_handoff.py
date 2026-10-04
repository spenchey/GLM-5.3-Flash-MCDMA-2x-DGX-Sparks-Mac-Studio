"""Fail-closed protocol for transferring a prompt cache while it is produced.

Protocol 2 publishes a complete, hashed cache manifest after Spark prefill has
finished.  Protocol 3 separates the immutable plan from the final seal: the
Mac may receive ordered bytes as layers become final, but no staged tensor is
visible to decode until the complete cache and its final hashes are verified.
Protocol 4 authenticates each tensor segment while it is copied so the final
seal can reuse those proofs instead of rereading the complete cache.

This module has no Torch, MLX, or MCDMA imports.  The host-only state machine
can therefore be tested before it is connected to CUDA callbacks or a live
mailbox.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import secrets
import struct
import threading
import time
from typing import Callable, Iterable, Mapping

import numpy as np

from experiments.three_machine.cache_handoff import HandoffError, MAX_FRAME_BYTES


PROGRESSIVE_PROTOCOL_VERSION = 4
PLAN_MAGIC = b"TFPL"
SEAL_MAGIC = b"TFSL"
PROGRESSIVE_FRAME_MAGIC = b"TFPF"
PROGRESSIVE_REQUEST_MAGIC = b"TFPQ"
PROGRESSIVE_ACK_MAGIC = b"TFPA"
MAX_HEADER_BYTES = 64 * 1024
DEFAULT_PROGRESSIVE_FRAME_BYTES = 32 * 1024 * 1024
_HEADER_LENGTH = struct.Struct(">I")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and value == value.lower() and all(char in "0123456789abcdef" for char in value)


def new_job_id() -> str:
    """Return a per-attempt nonce; it is deliberately not the final transfer ID."""

    return secrets.token_hex(16)


@dataclass(frozen=True)
class ProgressiveTensorSpec:
    """Tensor geometry in the exact order that bytes become transferable."""

    ordinal: int
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int

    @classmethod
    def from_array(
        cls, ordinal: int, name: str, value: np.ndarray, *, dtype: str | None = None
    ) -> "ProgressiveTensorSpec":
        array = np.ascontiguousarray(value)
        return cls(
            ordinal=int(ordinal),
            name=name,
            dtype=dtype or array.dtype.str,
            shape=tuple(int(item) for item in array.shape),
            nbytes=memoryview(array).cast("B").nbytes,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProgressiveTensorSpec":
        try:
            return cls(
                ordinal=int(value["ordinal"]),
                name=str(value["name"]),
                dtype=str(value["dtype"]),
                shape=tuple(int(item) for item in value["shape"]),  # type: ignore[arg-type]
                nbytes=int(value["nbytes"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HandoffError(f"invalid progressive tensor plan: {exc}") from exc

    def validate(self) -> None:
        if self.ordinal < 0 or not self.name or not self.dtype:
            raise HandoffError("progressive tensor identity is incomplete")
        if not self.shape or any(size <= 0 for size in self.shape) or self.nbytes <= 0:
            raise HandoffError(f"progressive tensor {self.name!r} has invalid geometry")


@dataclass(frozen=True)
class ProgressivePlan:
    """Early immutable cache geometry, before any content digest exists."""

    protocol: int
    job_id: str
    generation: int
    plan_id: str
    model_id: str
    model_revision: str
    prompt_sha256: str
    cached_tokens: int
    mtp_cached_tokens: int
    world_size: int
    long_context: bool
    tensors: tuple[ProgressiveTensorSpec, ...]

    @classmethod
    def build(
        cls,
        *,
        job_id: str,
        generation: int,
        model_id: str,
        model_revision: str,
        prompt_sha256: str,
        cached_tokens: int,
        tensors: Iterable[ProgressiveTensorSpec],
        mtp_cached_tokens: int | None = None,
        world_size: int = 2,
        long_context: bool = False,
    ) -> "ProgressivePlan":
        ordered = tuple(tensors)
        mtp_tokens = int(cached_tokens if mtp_cached_tokens is None else mtp_cached_tokens)
        body = {
            "protocol": PROGRESSIVE_PROTOCOL_VERSION,
            "job_id": job_id,
            "generation": int(generation),
            "model_id": model_id,
            "model_revision": model_revision,
            "prompt_sha256": prompt_sha256,
            "cached_tokens": int(cached_tokens),
            "mtp_cached_tokens": mtp_tokens,
            "world_size": int(world_size),
            "long_context": bool(long_context),
            "tensors": [asdict(item) for item in ordered],
        }
        plan = cls(
            protocol=PROGRESSIVE_PROTOCOL_VERSION,
            job_id=job_id,
            generation=int(generation),
            plan_id=hashlib.sha256(_canonical_json(body)).hexdigest(),
            model_id=model_id,
            model_revision=model_revision,
            prompt_sha256=prompt_sha256,
            cached_tokens=int(cached_tokens),
            mtp_cached_tokens=mtp_tokens,
            world_size=int(world_size),
            long_context=bool(long_context),
            tensors=ordered,
        )
        plan.validate()
        return plan

    def _identity_body(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "job_id": self.job_id,
            "generation": self.generation,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_sha256": self.prompt_sha256,
            "cached_tokens": self.cached_tokens,
            "mtp_cached_tokens": self.mtp_cached_tokens,
            "world_size": self.world_size,
            "long_context": self.long_context,
            "tensors": [asdict(item) for item in self.tensors],
        }

    def validate(self) -> None:
        if self.protocol != PROGRESSIVE_PROTOCOL_VERSION:
            raise HandoffError(f"unsupported progressive protocol {self.protocol}")
        if not _is_hex(self.job_id, 32) or not _is_hex(self.plan_id, 64):
            raise HandoffError("progressive job or plan identity is invalid")
        if self.generation <= 0:
            raise HandoffError("progressive plan generation is invalid")
        if not self.model_id or not self.model_revision or not _is_hex(self.prompt_sha256, 64):
            raise HandoffError("progressive model or prompt identity is incomplete")
        if self.cached_tokens <= 0 or self.world_size != 2:
            raise HandoffError("progressive cache position or Spark world size is invalid")
        if not 0 <= self.mtp_cached_tokens <= self.cached_tokens:
            raise HandoffError("progressive MTP cache position is invalid")
        if self.long_context:
            raise HandoffError(
                f"long-context index state is not supported by progressive handoff "
                f"v{PROGRESSIVE_PROTOCOL_VERSION}"
            )
        if not self.tensors:
            raise HandoffError("progressive cache plan has no tensors")
        names = [item.name for item in self.tensors]
        ordinals = [item.ordinal for item in self.tensors]
        if len(names) != len(set(names)) or ordinals != list(range(len(self.tensors))):
            raise HandoffError("progressive tensor names or execution order are invalid")
        for item in self.tensors:
            item.validate()
        want = hashlib.sha256(_canonical_json(self._identity_body())).hexdigest()
        if self.plan_id != want:
            raise HandoffError("progressive plan identity does not match its contents")

    @property
    def total_nbytes(self) -> int:
        return sum(item.nbytes for item in self.tensors)

    def tensor(self, name: str) -> ProgressiveTensorSpec:
        for item in self.tensors:
            if item.name == name:
                return item
        raise HandoffError(f"progressive plan has no tensor {name!r}")

    def require_identity(
        self,
        *,
        model_id: str,
        model_revision: str,
        prompt_sha256: str,
        cached_tokens: int,
        generation: int,
        mtp_cached_tokens: int | None = None,
    ) -> None:
        self.validate()
        expected_mtp = int(cached_tokens if mtp_cached_tokens is None else mtp_cached_tokens)
        if (
            self.model_id,
            self.model_revision,
            self.prompt_sha256,
            self.cached_tokens,
            self.mtp_cached_tokens,
            self.generation,
        ) != (model_id, model_revision, prompt_sha256, int(cached_tokens), expected_mtp, int(generation)):
            raise HandoffError("progressive plan differs from the decode request")

    def to_bytes(self) -> bytes:
        self.validate()
        return _canonical_json({**self._identity_body(), "plan_id": self.plan_id})

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ProgressivePlan":
        try:
            value = json.loads(raw.decode("ascii"))
            tensors = tuple(ProgressiveTensorSpec.from_dict(item) for item in value.pop("tensors"))
            plan = cls(tensors=tensors, **value)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise HandoffError(f"invalid progressive plan: {exc}") from exc
        plan.validate()
        return plan


@dataclass(frozen=True)
class TensorDigest:
    name: str
    nbytes: int
    sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TensorDigest":
        try:
            return cls(name=str(value["name"]), nbytes=int(value["nbytes"]), sha256=str(value["sha256"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise HandoffError(f"invalid progressive tensor digest: {exc}") from exc

    def validate(self) -> None:
        if not self.name or self.nbytes <= 0 or not _is_hex(self.sha256, 64):
            raise HandoffError("progressive tensor digest is invalid")


@dataclass(frozen=True)
class ProgressiveSeal:
    """Final cache identity; decode is forbidden until this seal verifies."""

    protocol: int
    job_id: str
    plan_id: str
    transfer_id: str
    spark_first_token: int
    tensors: tuple[TensorDigest, ...]

    @classmethod
    def build(
        cls,
        plan: ProgressivePlan,
        buffers: Mapping[str, bytes | bytearray | memoryview],
        *,
        spark_first_token: int,
    ) -> "ProgressiveSeal":
        plan.validate()
        digests: list[TensorDigest] = []
        for spec in plan.tensors:
            try:
                raw = memoryview(buffers[spec.name]).cast("B")
            except KeyError as exc:
                raise HandoffError(f"progressive cache is missing tensor {spec.name!r}") from exc
            if raw.nbytes != spec.nbytes:
                raise HandoffError(f"progressive tensor {spec.name!r} is incomplete")
            digests.append(TensorDigest(spec.name, spec.nbytes, hashlib.sha256(raw).hexdigest()))
        return cls.build_from_digests(
            plan,
            digests,
            spark_first_token=spark_first_token,
        )

    @classmethod
    def build_from_digests(
        cls,
        plan: ProgressivePlan,
        digests: Iterable[TensorDigest],
        *,
        spark_first_token: int,
    ) -> "ProgressiveSeal":
        """Build a seal from hashes already computed over immutable tensors."""

        plan.validate()
        ordered = tuple(digests)
        expected = [(item.name, item.nbytes) for item in plan.tensors]
        actual = [(item.name, item.nbytes) for item in ordered]
        if actual != expected:
            raise HandoffError("progressive tensor digests differ from the plan")
        for item in ordered:
            item.validate()
        body = {
            "protocol": PROGRESSIVE_PROTOCOL_VERSION,
            "job_id": plan.job_id,
            "plan_id": plan.plan_id,
            "spark_first_token": int(spark_first_token),
            "tensors": [asdict(item) for item in ordered],
        }
        seal = cls(
            transfer_id=hashlib.sha256(_canonical_json(body)).hexdigest(),
            tensors=ordered,
            **{key: value for key, value in body.items() if key != "tensors"},
        )
        seal.validate_for(plan)
        return seal

    def _identity_body(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "job_id": self.job_id,
            "plan_id": self.plan_id,
            "spark_first_token": self.spark_first_token,
            "tensors": [asdict(item) for item in self.tensors],
        }

    def validate(self) -> None:
        if self.protocol != PROGRESSIVE_PROTOCOL_VERSION:
            raise HandoffError(f"unsupported progressive seal protocol {self.protocol}")
        if not _is_hex(self.job_id, 32) or not _is_hex(self.plan_id, 64) or not _is_hex(self.transfer_id, 64):
            raise HandoffError("progressive seal identity is invalid")
        if not 0 <= self.spark_first_token <= 0xFFFFFFFF or not self.tensors:
            raise HandoffError("progressive seal token or tensor list is invalid")
        for item in self.tensors:
            item.validate()
        want = hashlib.sha256(_canonical_json(self._identity_body())).hexdigest()
        if self.transfer_id != want:
            raise HandoffError("progressive transfer identity does not match its seal")

    def validate_for(self, plan: ProgressivePlan) -> None:
        plan.validate()
        self.validate()
        if self.job_id != plan.job_id or self.plan_id != plan.plan_id:
            raise HandoffError("progressive seal belongs to a different plan")
        expected = [(item.name, item.nbytes) for item in plan.tensors]
        actual = [(item.name, item.nbytes) for item in self.tensors]
        if actual != expected:
            raise HandoffError("progressive seal tensor map differs from its plan")

    def to_bytes(self) -> bytes:
        self.validate()
        return _canonical_json({**self._identity_body(), "transfer_id": self.transfer_id})

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ProgressiveSeal":
        try:
            value = json.loads(raw.decode("ascii"))
            tensors = tuple(TensorDigest.from_dict(item) for item in value.pop("tensors"))
            seal = cls(tensors=tensors, **value)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise HandoffError(f"invalid progressive seal: {exc}") from exc
        seal.validate()
        return seal


@dataclass(frozen=True)
class ProgressiveManifest:
    """Sealed v3 cache identity with the read interface used by Mac import.

    The early plan deliberately has no content hashes or first-token result.
    This view exists only after the final seal verifies and prevents the Mac
    importer from accidentally treating an unsealed plan as a usable cache.
    """

    plan: ProgressivePlan
    seal: ProgressiveSeal

    def validate(self) -> None:
        self.seal.validate_for(self.plan)

    @property
    def protocol(self) -> int:
        return self.plan.protocol

    @property
    def transfer_id(self) -> str:
        return self.seal.transfer_id

    @property
    def model_id(self) -> str:
        return self.plan.model_id

    @property
    def model_revision(self) -> str:
        return self.plan.model_revision

    @property
    def prompt_sha256(self) -> str:
        return self.plan.prompt_sha256

    @property
    def cached_tokens(self) -> int:
        return self.plan.cached_tokens

    @property
    def mtp_cached_tokens(self) -> int:
        return self.plan.mtp_cached_tokens

    @property
    def spark_first_token(self) -> int:
        return self.seal.spark_first_token

    @property
    def world_size(self) -> int:
        return self.plan.world_size

    @property
    def long_context(self) -> bool:
        return self.plan.long_context

    @property
    def tensors(self) -> tuple[ProgressiveTensorSpec, ...]:
        return self.plan.tensors

    def tensor(self, name: str) -> ProgressiveTensorSpec:
        return self.plan.tensor(name)

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
        self.plan.require_identity(
            model_id=model_id,
            model_revision=model_revision,
            prompt_sha256=prompt_sha256,
            cached_tokens=cached_tokens,
            mtp_cached_tokens=mtp_cached_tokens,
            generation=self.plan.generation,
        )


def pack_plan(plan: ProgressivePlan) -> bytes:
    return PLAN_MAGIC + plan.to_bytes()


def unpack_plan(raw: bytes) -> ProgressivePlan:
    if not raw.startswith(PLAN_MAGIC):
        raise HandoffError("progressive response is not a plan")
    return ProgressivePlan.from_bytes(raw[len(PLAN_MAGIC):])


def pack_seal(seal: ProgressiveSeal) -> bytes:
    return SEAL_MAGIC + seal.to_bytes()


def unpack_seal(raw: bytes) -> ProgressiveSeal:
    if not raw.startswith(SEAL_MAGIC):
        raise HandoffError("progressive response is not a seal")
    return ProgressiveSeal.from_bytes(raw[len(SEAL_MAGIC):])


@dataclass(frozen=True)
class ProgressiveRequest:
    """One generation-bound Mac request for a progressive Spark job."""

    operation: str
    generation: int
    job_id: str
    plan_id: str | None = None
    transfer_id: str | None = None
    tokens: tuple[int, ...] = ()
    prompt_sha256: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    offset: int = 0
    length: int = MAX_FRAME_BYTES


def encode_progressive_request(request: ProgressiveRequest) -> bytes:
    if request.operation not in {"prepare", "frame", "seal", "release"}:
        raise HandoffError(f"unsupported progressive operation {request.operation!r}")
    if request.generation <= 0 or not _is_hex(request.job_id, 32):
        raise HandoffError("progressive request identity is invalid")
    value: dict[str, object] = {
        "protocol": PROGRESSIVE_PROTOCOL_VERSION,
        "operation": request.operation,
        "generation": int(request.generation),
        "job_id": request.job_id,
    }
    if request.operation == "prepare":
        if request.plan_id is not None or request.transfer_id is not None:
            raise HandoffError("progressive prepare request has a premature cache identity")
        if len(request.tokens) < 2 or any(
            not 0 <= int(token) <= 0xFFFFFFFF for token in request.tokens
        ):
            raise HandoffError("progressive preparation needs at least two uint32 token IDs")
        from experiments.three_machine.cache_handoff import prompt_digest

        digest = prompt_digest(request.tokens)
        if request.prompt_sha256 not in (None, digest):
            raise HandoffError("progressive prepare digest does not match its tokens")
        if not request.model_id or not request.model_revision:
            raise HandoffError("progressive prepare model identity is incomplete")
        value.update(
            tokens=list(request.tokens),
            prompt_sha256=digest,
            model_id=request.model_id,
            model_revision=request.model_revision,
        )
    else:
        if not request.plan_id or not _is_hex(request.plan_id, 64):
            raise HandoffError("progressive request has no valid plan identity")
        if request.tokens or request.prompt_sha256 is not None or request.model_id is not None \
                or request.model_revision is not None:
            raise HandoffError("progressive follow-up request repeats preparation fields")
        value["plan_id"] = request.plan_id
        if request.operation == "frame":
            if request.transfer_id is not None or request.offset < 0:
                raise HandoffError("progressive frame request identity or range is invalid")
            if not 0 < request.length <= MAX_FRAME_BYTES:
                raise HandoffError("progressive frame request exceeds the safe payload")
            value.update(offset=int(request.offset), length=int(request.length))
        elif request.operation == "seal":
            if request.transfer_id is not None:
                raise HandoffError("progressive seal request has a premature transfer identity")
        else:
            if request.transfer_id is not None and not _is_hex(request.transfer_id, 64):
                raise HandoffError("progressive release transfer identity is invalid")
            value["transfer_id"] = request.transfer_id
    return PROGRESSIVE_REQUEST_MAGIC + _canonical_json(value)


def decode_progressive_request(raw: bytes) -> ProgressiveRequest:
    if not raw.startswith(PROGRESSIVE_REQUEST_MAGIC):
        raise HandoffError("progressive request magic is invalid")
    try:
        value = json.loads(raw[len(PROGRESSIVE_REQUEST_MAGIC):].decode("ascii"))
        if not isinstance(value, dict) or int(value.pop("protocol")) != PROGRESSIVE_PROTOCOL_VERSION:
            raise HandoffError("progressive request protocol is unsupported")
        operation = str(value.pop("operation"))
        if operation == "prepare":
            tokens = tuple(int(token) for token in value.pop("tokens"))
            request = ProgressiveRequest(operation=operation, tokens=tokens, **value)
        else:
            request = ProgressiveRequest(operation=operation, **value)
        encode_progressive_request(request)
        return request
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HandoffError(f"invalid progressive request: {exc}") from exc


def is_progressive_request(raw: bytes) -> bool:
    return raw.startswith(PROGRESSIVE_REQUEST_MAGIC)


def pack_progressive_ack(request: ProgressiveRequest, transfer_id: str | None = None) -> bytes:
    encode_progressive_request(request)
    if request.operation != "release" or not request.plan_id:
        raise HandoffError("only progressive release has an acknowledgement")
    if transfer_id is not None and not _is_hex(transfer_id, 64):
        raise HandoffError("progressive acknowledgement transfer identity is invalid")
    return PROGRESSIVE_ACK_MAGIC + _canonical_json({
        "protocol": PROGRESSIVE_PROTOCOL_VERSION,
        "operation": request.operation,
        "generation": request.generation,
        "job_id": request.job_id,
        "plan_id": request.plan_id,
        "transfer_id": transfer_id,
    })


def unpack_progressive_ack(raw: bytes, request: ProgressiveRequest) -> str | None:
    if not raw.startswith(PROGRESSIVE_ACK_MAGIC):
        raise HandoffError("progressive response is not an acknowledgement")
    try:
        value = json.loads(raw[len(PROGRESSIVE_ACK_MAGIC):].decode("ascii"))
        expected = {
            "protocol": PROGRESSIVE_PROTOCOL_VERSION,
            "operation": request.operation,
            "generation": request.generation,
            "job_id": request.job_id,
            "plan_id": request.plan_id,
        }
        if not isinstance(value, dict) or set(value) != {*expected, "transfer_id"}:
            raise HandoffError("progressive acknowledgement fields are invalid")
        if any(value[key] != expected_value for key, expected_value in expected.items()):
            raise HandoffError("progressive acknowledgement does not match its request")
        transfer_id = value["transfer_id"]
        if transfer_id is not None and not _is_hex(str(transfer_id), 64):
            raise HandoffError("progressive acknowledgement transfer identity is invalid")
        if request.transfer_id is not None and transfer_id != request.transfer_id:
            raise HandoffError("progressive acknowledgement transfer identity differs from its request")
        return None if transfer_id is None else str(transfer_id)
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HandoffError(f"invalid progressive acknowledgement: {exc}") from exc


@dataclass(frozen=True)
class ProgressiveSegment:
    name: str
    offset: int
    length: int


@dataclass(frozen=True)
class ProgressiveFrame:
    job_id: str
    plan_id: str
    offset: int
    total: int
    segments: tuple[ProgressiveSegment, ...]
    segment_sha256: tuple[str, ...]
    payload: memoryview


def _segments(plan: ProgressivePlan, offset: int, length: int) -> tuple[ProgressiveSegment, ...]:
    if offset < 0 or offset >= plan.total_nbytes or not 0 < length <= MAX_FRAME_BYTES:
        raise HandoffError("progressive frame range is outside the plan")
    remaining = min(length, plan.total_nbytes - offset)
    cursor = 0
    segments: list[ProgressiveSegment] = []
    for spec in plan.tensors:
        end = cursor + spec.nbytes
        if offset < end and remaining:
            tensor_offset = max(0, offset - cursor)
            count = min(remaining, spec.nbytes - tensor_offset)
            segments.append(ProgressiveSegment(spec.name, tensor_offset, count))
            offset += count
            remaining -= count
        cursor = end
        if not remaining:
            break
    if remaining or not segments:
        raise HandoffError("progressive frame range could not be mapped to tensors")
    return tuple(segments)


def progressive_frame_ranges(
    plan: ProgressivePlan,
    max_payload: int = DEFAULT_PROGRESSIVE_FRAME_BYTES,
) -> tuple[tuple[int, int], ...]:
    """Group execution-ordered tensors without delaying a frame for the next one."""

    plan.validate()
    if not 0 < max_payload <= MAX_FRAME_BYTES:
        raise HandoffError("progressive frame payload limit is invalid")
    ranges: list[tuple[int, int]] = []
    frame_start = 0
    frame_length = 0
    cursor = 0
    for spec in plan.tensors:
        remaining = spec.nbytes
        while remaining:
            room = max_payload - frame_length
            if frame_length and remaining > room:
                ranges.append((frame_start, frame_length))
                frame_start = cursor
                frame_length = 0
                room = max_payload
            take = min(remaining, room)
            frame_length += take
            cursor += take
            remaining -= take
            if frame_length == max_payload:
                ranges.append((frame_start, frame_length))
                frame_start = cursor
                frame_length = 0
    if frame_length:
        ranges.append((frame_start, frame_length))
    if cursor != plan.total_nbytes or sum(length for _, length in ranges) != plan.total_nbytes:
        raise HandoffError("progressive frame plan does not cover the cache")
    return tuple(ranges)


def pack_progressive_frame(
    plan: ProgressivePlan,
    buffers: Mapping[str, bytes | bytearray | memoryview],
    offset: int,
    length: int,
    *,
    tensor_digests: Mapping[str, str] | None = None,
) -> bytes:
    header, parts = _progressive_frame_parts(
        plan,
        buffers,
        offset,
        length,
        tensor_digests=tensor_digests,
    )
    size = len(PROGRESSIVE_FRAME_MAGIC) + _HEADER_LENGTH.size + len(header)
    size += sum(segment.length for segment, _part in parts)
    raw = bytearray(size)
    written = _write_progressive_frame_parts(memoryview(raw), header, parts)
    if written != size:
        raise HandoffError("progressive frame size changed while it was packed")
    return bytes(raw)


def _progressive_frame_parts(
    plan: ProgressivePlan,
    buffers: Mapping[str, bytes | bytearray | memoryview],
    offset: int,
    length: int,
    *,
    tensor_digests: Mapping[str, str] | None = None,
) -> tuple[bytes, tuple[tuple[ProgressiveSegment, memoryview], ...]]:
    """Freeze one frame's source views and authenticated header."""

    plan.validate()
    segments = _segments(plan, offset, length)
    parts: list[tuple[ProgressiveSegment, memoryview]] = []
    proofs: list[dict[str, object]] = []
    for segment in segments:
        try:
            raw = memoryview(buffers[segment.name]).cast("B")
        except KeyError as exc:
            raise HandoffError(f"progressive frame is missing tensor {segment.name!r}") from exc
        if raw.nbytes != plan.tensor(segment.name).nbytes:
            raise HandoffError(f"progressive tensor {segment.name!r} storage has the wrong size")
        part = raw[segment.offset:segment.offset + segment.length]
        parts.append((segment, part))
        digest = None if tensor_digests is None else tensor_digests.get(segment.name)
        if digest is not None and not _is_hex(digest, 64):
            raise HandoffError(f"progressive tensor {segment.name!r} has an invalid cached digest")
        if digest is None or segment.offset != 0 or segment.length != raw.nbytes:
            digest = hashlib.sha256(part).hexdigest()
        proofs.append({**asdict(segment), "sha256": digest})
    header = _canonical_json({
        "protocol": PROGRESSIVE_PROTOCOL_VERSION,
        "job_id": plan.job_id,
        "plan_id": plan.plan_id,
        "offset": int(offset),
        "total": plan.total_nbytes,
        "segments": proofs,
    })
    if len(header) > MAX_HEADER_BYTES:
        raise HandoffError("progressive frame header is too large")
    return header, tuple(parts)


def _write_progressive_frame_parts(
    target: memoryview,
    header: bytes,
    parts: tuple[tuple[ProgressiveSegment, memoryview], ...],
) -> int:
    payload_bytes = sum(segment.length for segment, _part in parts)
    total_bytes = len(PROGRESSIVE_FRAME_MAGIC) + _HEADER_LENGTH.size + len(header) + payload_bytes
    area = memoryview(target).cast("B")
    if area.readonly or area.nbytes < total_bytes:
        raise HandoffError("progressive frame does not fit the reply area")
    cursor = 0
    area[cursor:cursor + len(PROGRESSIVE_FRAME_MAGIC)] = PROGRESSIVE_FRAME_MAGIC
    cursor += len(PROGRESSIVE_FRAME_MAGIC)
    area[cursor:cursor + _HEADER_LENGTH.size] = _HEADER_LENGTH.pack(len(header))
    cursor += _HEADER_LENGTH.size
    area[cursor:cursor + len(header)] = header
    cursor += len(header)
    for segment, part in parts:
        area[cursor:cursor + segment.length] = part
        cursor += segment.length
    return cursor


def write_progressive_frame(
    plan: ProgressivePlan,
    buffers: Mapping[str, bytes | bytearray | memoryview],
    target: memoryview,
    offset: int,
    length: int,
    *,
    tensor_digests: Mapping[str, str] | None = None,
) -> int:
    """Write one authenticated frame directly into MCDMA reply storage."""

    header, parts = _progressive_frame_parts(
        plan,
        buffers,
        offset,
        length,
        tensor_digests=tensor_digests,
    )
    return _write_progressive_frame_parts(target, header, parts)


def unpack_progressive_frame(raw: bytes | bytearray | memoryview) -> ProgressiveFrame:
    view = memoryview(raw).cast("B")
    prefix = len(PROGRESSIVE_FRAME_MAGIC) + _HEADER_LENGTH.size
    if view.nbytes <= prefix or bytes(view[:len(PROGRESSIVE_FRAME_MAGIC)]) != PROGRESSIVE_FRAME_MAGIC:
        raise HandoffError("progressive frame magic is invalid")
    header_len = _HEADER_LENGTH.unpack_from(view, len(PROGRESSIVE_FRAME_MAGIC))[0]
    if not 0 < header_len <= MAX_HEADER_BYTES or view.nbytes <= prefix + header_len:
        raise HandoffError("progressive frame header length is invalid")
    try:
        header = json.loads(bytes(view[prefix:prefix + header_len]).decode("ascii"))
        if not isinstance(header, dict) or set(header) != {
            "protocol", "job_id", "plan_id", "offset", "total", "segments"
        }:
            raise HandoffError("progressive frame header has unexpected fields")
        if int(header["protocol"]) != PROGRESSIVE_PROTOCOL_VERSION:
            raise HandoffError("progressive frame protocol is unsupported")
        job_id = str(header["job_id"])
        plan_id = str(header["plan_id"])
        frame_offset = int(header["offset"])
        total = int(header["total"])
        if not _is_hex(job_id, 32) or not _is_hex(plan_id, 64):
            raise HandoffError("progressive frame identity is invalid")
        if frame_offset < 0 or total <= 0:
            raise HandoffError("progressive frame range or digest is invalid")
        payload = view[prefix + header_len:].toreadonly()
        if not payload or payload.nbytes > MAX_FRAME_BYTES:
            raise HandoffError("progressive frame payload size is invalid")
        segment_values = header["segments"]
        if not isinstance(segment_values, list) or not segment_values:
            raise HandoffError("progressive frame has no tensor segments")
        if any(not isinstance(item, dict) or set(item) != {"name", "offset", "length", "sha256"}
               for item in segment_values):
            raise HandoffError("progressive frame tensor map is invalid")
        segments = tuple(ProgressiveSegment(
            name=str(item["name"]), offset=int(item["offset"]), length=int(item["length"])
        ) for item in segment_values)
        if not segments or any(item.offset < 0 or item.length <= 0 for item in segments):
            raise HandoffError("progressive frame tensor map is invalid")
        if sum(item.length for item in segments) != payload.nbytes:
            raise HandoffError("progressive frame segments do not cover its payload")
        digests = tuple(str(item["sha256"]) for item in segment_values)
        cursor = 0
        for segment, digest in zip(segments, digests):
            if not _is_hex(digest, 64):
                raise HandoffError("progressive frame tensor digest is invalid")
            end = cursor + segment.length
            if hashlib.sha256(payload[cursor:end]).hexdigest() != digest:
                raise HandoffError("progressive frame tensor digest mismatch")
            cursor = end
        return ProgressiveFrame(
            job_id=job_id,
            plan_id=plan_id,
            offset=frame_offset,
            total=total,
            segments=segments,
            segment_sha256=digests,
            payload=payload,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HandoffError(f"invalid progressive frame: {exc}") from exc


class ProgressiveAssembler:
    """Mac-side staging that exposes no cache bytes before the final seal."""

    def __init__(self, plan: ProgressivePlan) -> None:
        plan.validate()
        self.plan = plan
        self._data = {item.name: bytearray(item.nbytes) for item in plan.tensors}
        self._written = {item.name: 0 for item in plan.tensors}
        self._hashers = {item.name: hashlib.sha256() for item in plan.tensors}
        self._digests: dict[str, str] = {}
        self._frame_offset = 0
        self._seal: ProgressiveSeal | None = None
        self._failed: str | None = None

    @property
    def staged_nbytes(self) -> int:
        return self._frame_offset

    @property
    def sealed(self) -> bool:
        return self._seal is not None

    @property
    def manifest(self) -> ProgressiveManifest:
        if self._failed is not None:
            raise HandoffError(f"progressive assembly was discarded: {self._failed}")
        if self._seal is None:
            raise HandoffError("progressive cache is not sealed for decode")
        return ProgressiveManifest(self.plan, self._seal)

    def _require_open(self) -> None:
        if self._failed is not None:
            raise HandoffError(f"progressive assembly was discarded: {self._failed}")
        if self._seal is not None:
            raise HandoffError("progressive assembly is already sealed")

    def accept_frame(self, raw: bytes | bytearray | memoryview) -> bool:
        self._require_open()
        try:
            frame = unpack_progressive_frame(raw)
            if frame.job_id != self.plan.job_id or frame.plan_id != self.plan.plan_id:
                raise HandoffError("progressive frame belongs to a different plan")
            if frame.total != self.plan.total_nbytes or frame.offset != self._frame_offset:
                raise HandoffError("progressive frame size or order differs from the plan")
            expected = _segments(self.plan, frame.offset, frame.payload.nbytes)
            if frame.segments != expected:
                raise HandoffError("progressive frame tensor map differs from the plan")
            cursor = 0
            for segment, digest in zip(frame.segments, frame.segment_sha256):
                written = self._written[segment.name]
                if segment.offset != written:
                    raise HandoffError("progressive tensor bytes are out of order")
                end = cursor + segment.length
                self._data[segment.name][written:written + segment.length] = frame.payload[cursor:end]
                self._written[segment.name] = written + segment.length
                spec = self.plan.tensor(segment.name)
                if segment.offset == 0 and segment.length == spec.nbytes:
                    self._digests[segment.name] = digest
                else:
                    self._hashers[segment.name].update(frame.payload[cursor:end])
                    if self._written[segment.name] == spec.nbytes:
                        self._digests[segment.name] = self._hashers[segment.name].hexdigest()
                cursor = end
            self._frame_offset += frame.payload.nbytes
            return self._frame_offset == self.plan.total_nbytes
        except Exception as exc:
            self.abort(str(exc))
            raise

    def accept_seal(self, seal: ProgressiveSeal) -> str:
        self._require_open()
        try:
            seal.validate_for(self.plan)
            if self._frame_offset != self.plan.total_nbytes:
                raise HandoffError("progressive cache is incomplete at seal")
            by_name = {item.name: item for item in seal.tensors}
            for spec in self.plan.tensors:
                digest = by_name[spec.name]
                if self._written[spec.name] != spec.nbytes:
                    raise HandoffError(f"progressive tensor {spec.name!r} is incomplete at seal")
                if self._digests.get(spec.name) != digest.sha256:
                    raise HandoffError(f"progressive tensor {spec.name!r} digest mismatch at seal")
        except Exception as exc:
            self.abort(str(exc))
            raise
        self._seal = seal
        return seal.transfer_id

    def abort(self, reason: str) -> None:
        self._data.clear()
        self._written.clear()
        self._hashers.clear()
        self._digests.clear()
        self._frame_offset = 0
        self._failed = reason or "aborted"
        self._seal = None

    def buffer_for(self, name: str) -> memoryview:
        if self._failed is not None:
            raise HandoffError(f"progressive assembly was discarded: {self._failed}")
        if self._seal is None:
            raise HandoffError("progressive cache is not sealed for decode")
        self.plan.tensor(name)
        return memoryview(self._data[name]).toreadonly()

    def array(self, name: str) -> np.ndarray:
        spec = self.plan.tensor(name)
        try:
            dtype = np.dtype(spec.dtype)
        except TypeError as exc:
            raise HandoffError(f"progressive tensor {name!r} uses host-specific dtype {spec.dtype!r}") from exc
        try:
            return np.frombuffer(self.buffer_for(name), dtype=dtype).reshape(spec.shape)
        except ValueError as exc:
            raise HandoffError(f"progressive tensor {name!r} byte count does not match its shape") from exc


class ProgressiveJob:
    """Spark-side ordered staging with bounded waits and reconnect invalidation."""

    def __init__(self, plan: ProgressivePlan, *, generation: int) -> None:
        plan.validate()
        if generation <= 0:
            raise HandoffError("progressive job needs a positive link generation")
        self.plan = plan
        self.generation = int(generation)
        self._condition = threading.Condition()
        self._data = {item.name: bytearray(item.nbytes) for item in plan.tensors}
        self._written = {item.name: 0 for item in plan.tensors}
        self._hashers = {item.name: hashlib.sha256() for item in plan.tensors}
        self._digests: dict[str, str] = {}
        self._next_tensor = 0
        self._ready = 0
        self._seal: ProgressiveSeal | None = None
        self._terminal: str | None = None
        self._released = False

    @property
    def ready_nbytes(self) -> int:
        with self._condition:
            return self._ready

    @property
    def buffered_nbytes(self) -> int:
        with self._condition:
            return sum(len(value) for value in self._data.values())

    def _check_generation(self, generation: int) -> None:
        if generation != self.generation:
            raise HandoffError("progressive job belongs to an old link generation")

    def _check_live(self) -> None:
        if self._released:
            raise HandoffError("progressive job was released")
        if self._terminal is not None:
            raise HandoffError(self._terminal)

    def publish_piece(self, name: str, offset: int, payload: bytes | bytearray | memoryview) -> bool:
        view = memoryview(payload).cast("B")
        if not view:
            raise HandoffError("progressive cache piece is empty")
        with self._condition:
            self._check_live()
            if self._seal is not None:
                raise HandoffError("progressive job is already sealed")
            if self._next_tensor >= len(self.plan.tensors):
                raise HandoffError("progressive cache already has every planned tensor")
            spec = self.plan.tensors[self._next_tensor]
            written = self._written[spec.name]
            if name != spec.name or offset != written:
                raise HandoffError("progressive cache pieces must follow execution order")
            if written + view.nbytes > spec.nbytes:
                raise HandoffError("progressive cache piece exceeds its tensor")
            self._data[name][written:written + view.nbytes] = view
            stored = memoryview(self._data[name])[written:written + view.nbytes]
            self._hashers[name].update(stored)
            self._written[name] = written + view.nbytes
            self._ready += view.nbytes
            complete = self._written[name] == spec.nbytes
            if complete:
                self._digests[name] = self._hashers[name].hexdigest()
                self._next_tensor += 1
            self._condition.notify_all()
            return complete

    def _wait(self, predicate: Callable[[], bool], timeout: float, generation: int) -> None:
        if timeout <= 0:
            raise HandoffError("progressive wait needs a positive timeout")
        deadline = time.monotonic() + timeout
        while True:
            self._check_generation(generation)
            self._check_live()
            if predicate():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HandoffError("progressive cache wait timed out")
            self._condition.wait(remaining)

    def frame(self, offset: int, length: int, *, timeout: float, generation: int) -> bytes:
        if offset < 0 or offset >= self.plan.total_nbytes or not 0 < length <= MAX_FRAME_BYTES:
            raise HandoffError("progressive frame request is outside the plan")
        count = min(length, self.plan.total_nbytes - offset)
        with self._condition:
            self._check_generation(generation)
            self._wait(lambda: self._ready >= offset + count, timeout, generation)
            return pack_progressive_frame(
                self.plan,
                self._data,
                offset,
                count,
                tensor_digests=self._digests,
            )

    def write_frame(
        self,
        target: memoryview,
        offset: int,
        length: int,
        *,
        timeout: float,
        generation: int,
    ) -> int:
        """Wait for a range and write it directly into stable reply storage."""

        if offset < 0 or offset >= self.plan.total_nbytes or not 0 < length <= MAX_FRAME_BYTES:
            raise HandoffError("progressive frame request is outside the plan")
        count = min(length, self.plan.total_nbytes - offset)
        with self._condition:
            self._check_generation(generation)
            self._wait(lambda: self._ready >= offset + count, timeout, generation)
            return write_progressive_frame(
                self.plan,
                self._data,
                target,
                offset,
                count,
                tensor_digests=self._digests,
            )

    def seal(self, *, spark_first_token: int, generation: int) -> ProgressiveSeal:
        with self._condition:
            self._check_generation(generation)
            self._check_live()
            if self._seal is not None:
                if self._seal.spark_first_token != spark_first_token:
                    raise HandoffError("progressive job was sealed with another first token")
                return self._seal
            if self._ready != self.plan.total_nbytes:
                raise HandoffError("progressive cache cannot seal before every tensor is ready")
            self._seal = ProgressiveSeal.build_from_digests(
                self.plan,
                tuple(
                    TensorDigest(spec.name, spec.nbytes, self._digests[spec.name])
                    for spec in self.plan.tensors
                ),
                spark_first_token=spark_first_token,
            )
            self._condition.notify_all()
            return self._seal

    def wait_seal(self, *, timeout: float, generation: int) -> ProgressiveSeal:
        with self._condition:
            self._check_generation(generation)
            self._wait(lambda: self._seal is not None, timeout, generation)
            assert self._seal is not None
            return self._seal

    def _discard(self, reason: str, *, released: bool = False) -> None:
        self._terminal = reason
        self._released = released
        self._data.clear()
        self._written.clear()
        self._hashers.clear()
        self._digests.clear()
        self._seal = None
        self._condition.notify_all()

    def fail(self, reason: str) -> None:
        with self._condition:
            if self._released:
                return
            self._discard(reason or "progressive producer failed")

    def release(self, *, generation: int) -> None:
        with self._condition:
            self._check_generation(generation)
            self._discard("progressive job was released", released=True)

    def invalidate_generation(self, generation: int) -> None:
        if generation <= 0 or generation == self.generation:
            raise HandoffError("new progressive generation must be positive and different")
        with self._condition:
            self.generation = int(generation)
            self._discard("progressive link generation changed")


def reply_progressive_frame(
    mailbox: object,
    sequence: int,
    job: ProgressiveJob,
    offset: int,
    length: int,
    *,
    timeout: float,
    generation: int,
) -> None:
    """Publish directly from the job into a mailbox reply area when available."""

    reply_area = getattr(mailbox, "reply_area", None)
    publish = getattr(mailbox, "publish", None)
    if callable(reply_area) and callable(publish):
        written = job.write_frame(
            reply_area(), offset, length, timeout=timeout, generation=generation
        )
        publish(sequence, written)
        return
    mailbox.reply(  # type: ignore[attr-defined]
        sequence,
        job.frame(offset, length, timeout=timeout, generation=generation),
    )
