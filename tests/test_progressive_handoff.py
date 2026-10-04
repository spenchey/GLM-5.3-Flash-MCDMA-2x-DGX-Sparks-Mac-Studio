import hashlib
import json
import threading
import time

import numpy as np
import pytest

from experiments.three_machine.cache_handoff import HandoffError
from experiments.three_machine.progressive_handoff import (
    ProgressiveAssembler,
    ProgressiveJob,
    ProgressivePlan,
    ProgressiveRequest,
    ProgressiveSeal,
    ProgressiveTensorSpec,
    decode_progressive_request,
    encode_progressive_request,
    is_progressive_request,
    new_job_id,
    pack_plan,
    pack_progressive_frame,
    pack_progressive_ack,
    pack_seal,
    progressive_frame_ranges,
    reply_progressive_frame,
    unpack_plan,
    unpack_progressive_ack,
    unpack_progressive_frame,
    unpack_seal,
    write_progressive_frame,
)


MODEL = "TensorFold/GLM-5.3-Flash-MLX-4bit-MTP"
REVISION = "76add2a341a1cd90ad0e86bb69839ea9c35827c6"
PROMPT_SHA256 = hashlib.sha256(b"progressive prompt tokens").hexdigest()


def sample_plan(*, job_id="ab" * 16):
    arrays = {
        # Deliberately not lexical: protocol order is CUDA execution order.
        "layers.0.z_state": np.arange(6, dtype=np.uint16),
        "layers.0.a_state": np.arange(10, 15, dtype=np.uint16),
    }
    specs = tuple(
        ProgressiveTensorSpec.from_array(ordinal, name, value)
        for ordinal, (name, value) in enumerate(arrays.items())
    )
    plan = ProgressivePlan.build(
        job_id=job_id,
        generation=7,
        model_id=MODEL,
        model_revision=REVISION,
        prompt_sha256=PROMPT_SHA256,
        cached_tokens=33,
        mtp_cached_tokens=32,
        tensors=specs,
    )
    buffers = {
        name: memoryview(np.ascontiguousarray(value)).cast("B").tobytes()
        for name, value in arrays.items()
    }
    return plan, arrays, buffers


def stage_all(plan, buffers, *, frame_bytes=7):
    assembler = ProgressiveAssembler(plan)
    offset = 0
    while offset < plan.total_nbytes:
        length = min(frame_bytes, plan.total_nbytes - offset)
        complete = assembler.accept_frame(
            pack_progressive_frame(plan, buffers, offset, length)
        )
        offset += length
    assert complete
    return assembler


def test_plan_round_trip_binds_request_and_preserves_execution_order():
    plan, _arrays, _buffers = sample_plan()
    assert unpack_plan(pack_plan(plan)) == plan
    assert [item.name for item in plan.tensors] == [
        "layers.0.z_state",
        "layers.0.a_state",
    ]
    plan.require_identity(
        model_id=MODEL,
        model_revision=REVISION,
        prompt_sha256=PROMPT_SHA256,
        cached_tokens=33,
        generation=7,
        mtp_cached_tokens=32,
    )
    with pytest.raises(HandoffError, match="differs from the decode request"):
        plan.require_identity(
            model_id=MODEL,
            model_revision=REVISION,
            prompt_sha256=hashlib.sha256(b"other prompt").hexdigest(),
            cached_tokens=33,
            generation=7,
            mtp_cached_tokens=32,
        )


def test_plan_refuses_tampering_and_job_ids_are_attempt_scoped():
    plan, _arrays, _buffers = sample_plan()
    value = json.loads(plan.to_bytes())
    value["cached_tokens"] += 1
    with pytest.raises(HandoffError, match="identity does not match"):
        ProgressivePlan.from_bytes(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
    assert new_job_id() != new_job_id()


def test_frame_ranges_preserve_tensor_readiness_and_split_only_large_tensors():
    specs = tuple(
        ProgressiveTensorSpec(index, f"tensor.{index}", "|u1", (size,), size)
        for index, size in enumerate((10, 5, 30, 40))
    )
    plan = ProgressivePlan.build(
        job_id="ab" * 16,
        generation=7,
        model_id=MODEL,
        model_revision=REVISION,
        prompt_sha256=PROMPT_SHA256,
        cached_tokens=33,
        tensors=specs,
    )
    assert progressive_frame_ranges(plan, 32) == (
        (0, 15),
        (15, 30),
        (45, 32),
        (77, 8),
    )


def test_progressive_requests_round_trip_and_bind_link_generation():
    job_id = new_job_id()
    prepare = ProgressiveRequest(
        operation="prepare",
        generation=7,
        job_id=job_id,
        tokens=(1, 2, 3),
        prompt_sha256=hashlib.sha256(np.asarray((1, 2, 3), dtype=">u4").tobytes()).hexdigest(),
        model_id=MODEL,
        model_revision=REVISION,
    )
    encoded = encode_progressive_request(prepare)
    assert is_progressive_request(encoded)
    assert decode_progressive_request(encoded) == prepare

    plan, _arrays, _buffers = sample_plan(job_id=job_id)
    requests = (
        ProgressiveRequest(
            operation="frame", generation=7, job_id=job_id,
            plan_id=plan.plan_id, offset=3, length=9,
        ),
        ProgressiveRequest(
            operation="seal", generation=7, job_id=job_id, plan_id=plan.plan_id,
        ),
        ProgressiveRequest(
            operation="release", generation=7, job_id=job_id, plan_id=plan.plan_id,
        ),
    )
    for request in requests:
        assert decode_progressive_request(encode_progressive_request(request)) == request


def test_progressive_request_rejects_replay_and_unknown_fields():
    plan, _arrays, _buffers = sample_plan()
    with pytest.raises(HandoffError, match="identity is invalid"):
        encode_progressive_request(ProgressiveRequest(
            operation="seal", generation=0, job_id=plan.job_id, plan_id=plan.plan_id
        ))

    request = ProgressiveRequest(
        operation="seal", generation=7, job_id=plan.job_id, plan_id=plan.plan_id
    )
    encoded = encode_progressive_request(request)
    value = json.loads(encoded[4:])
    value["unexpected"] = True
    tampered = encoded[:4] + json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    with pytest.raises(HandoffError, match="invalid progressive request"):
        decode_progressive_request(tampered)


def test_progressive_release_ack_binds_every_identity():
    plan, _arrays, buffers = sample_plan()
    seal = ProgressiveSeal.build(plan, buffers, spark_first_token=42)
    request = ProgressiveRequest(
        operation="release",
        generation=7,
        job_id=plan.job_id,
        plan_id=plan.plan_id,
        transfer_id=seal.transfer_id,
    )
    raw = pack_progressive_ack(request, seal.transfer_id)
    assert unpack_progressive_ack(raw, request) == seal.transfer_id

    wrong = ProgressiveRequest(
        operation="release",
        generation=8,
        job_id=plan.job_id,
        plan_id=plan.plan_id,
        transfer_id=seal.transfer_id,
    )
    with pytest.raises(HandoffError, match="does not match"):
        unpack_progressive_ack(raw, wrong)


def test_frames_stage_early_but_decode_waits_for_the_final_seal():
    plan, arrays, buffers = sample_plan()
    assembler = ProgressiveAssembler(plan)
    first = pack_progressive_frame(plan, buffers, 0, 7)
    decoded = unpack_progressive_frame(memoryview(first))
    assert decoded.offset == 0
    assert not assembler.accept_frame(first)
    assert assembler.staged_nbytes == 7
    with pytest.raises(HandoffError, match="not sealed"):
        assembler.buffer_for(plan.tensors[0].name)

    offset = 7
    while offset < plan.total_nbytes:
        length = min(7, plan.total_nbytes - offset)
        assembler.accept_frame(pack_progressive_frame(plan, buffers, offset, length))
        offset += length

    seal = ProgressiveSeal.build(plan, buffers, spark_first_token=42)
    assert unpack_seal(pack_seal(seal)) == seal
    assert assembler.accept_seal(seal) == seal.transfer_id
    assert assembler.sealed
    for name, expected in arrays.items():
        actual = assembler.array(name)
        assert not actual.flags.writeable
        assert np.array_equal(actual, expected)


def test_frame_can_be_written_directly_into_mailbox_storage():
    plan, _arrays, buffers = sample_plan()
    expected = pack_progressive_frame(plan, buffers, 0, plan.total_nbytes)
    target = bytearray(len(expected) + 19)
    target[-19:] = b"x" * 19
    written = write_progressive_frame(
        plan, buffers, memoryview(target), 0, plan.total_nbytes
    )
    assert written == len(expected)
    assert bytes(target[:written]) == expected
    assert target[-19:] == b"x" * 19

    with pytest.raises(HandoffError, match="does not fit"):
        write_progressive_frame(
            plan, buffers, memoryview(bytearray(written - 1)), 0, plan.total_nbytes
        )
    with pytest.raises(HandoffError, match="does not fit"):
        write_progressive_frame(
            plan, buffers, memoryview(bytes(written)), 0, plan.total_nbytes
        )


def test_job_publishes_directly_into_mailbox_reply_area():
    plan, _arrays, buffers = sample_plan()
    job = ProgressiveJob(plan, generation=7)
    for spec in plan.tensors:
        job.publish_piece(spec.name, 0, buffers[spec.name])

    class DirectMailbox:
        def __init__(self):
            self.area = bytearray(4096)
            self.published = None

        def reply_area(self):
            return memoryview(self.area)

        def publish(self, sequence, length):
            self.published = (sequence, length)

        def reply(self, *_args):  # pragma: no cover - fast path must not call it
            raise AssertionError("direct progressive reply unexpectedly allocated bytes")

    mailbox = DirectMailbox()
    reply_progressive_frame(
        mailbox,
        11,
        job,
        0,
        plan.total_nbytes,
        timeout=0.1,
        generation=7,
    )
    assert mailbox.published is not None
    sequence, written = mailbox.published
    assert sequence == 11
    frame = unpack_progressive_frame(memoryview(mailbox.area)[:written])
    assert bytes(frame.payload) == b"".join(buffers[item.name] for item in plan.tensors)


def test_job_progressive_reply_has_compatible_bytes_fallback():
    plan, _arrays, buffers = sample_plan()
    job = ProgressiveJob(plan, generation=7)
    for spec in plan.tensors:
        job.publish_piece(spec.name, 0, buffers[spec.name])

    class FallbackMailbox:
        def __init__(self):
            self.answer = None

        def reply(self, sequence, payload):
            self.answer = (sequence, payload)

    mailbox = FallbackMailbox()
    reply_progressive_frame(
        mailbox,
        12,
        job,
        0,
        plan.total_nbytes,
        timeout=0.1,
        generation=7,
    )
    assert mailbox.answer is not None
    sequence, payload = mailbox.answer
    assert sequence == 12
    frame = unpack_progressive_frame(payload)
    assert bytes(frame.payload) == b"".join(buffers[item.name] for item in plan.tensors)


def test_seal_identity_is_stable_for_one_attempt_and_changes_between_attempts():
    plan, _arrays, buffers = sample_plan()
    seal = ProgressiveSeal.build(plan, buffers, spark_first_token=42)
    assert ProgressiveSeal.build(plan, buffers, spark_first_token=42) == seal

    other, _arrays, other_buffers = sample_plan(job_id="cd" * 16)
    other_seal = ProgressiveSeal.build(other, other_buffers, spark_first_token=42)
    assert other_seal.transfer_id != seal.transfer_id


def test_bad_frame_discards_all_mac_staging():
    plan, _arrays, buffers = sample_plan()
    assembler = ProgressiveAssembler(plan)
    corrupted = bytearray(pack_progressive_frame(plan, buffers, 0, 7))
    corrupted[-1] ^= 0xFF
    with pytest.raises(HandoffError, match="digest mismatch"):
        assembler.accept_frame(corrupted)
    assert assembler.staged_nbytes == 0
    with pytest.raises(HandoffError, match="discarded"):
        assembler.accept_frame(pack_progressive_frame(plan, buffers, 0, 7))


@pytest.mark.parametrize("mode", ["duplicate", "out-of-order"])
def test_duplicate_or_out_of_order_frame_discards_mac_staging(mode):
    plan, _arrays, buffers = sample_plan()
    assembler = ProgressiveAssembler(plan)
    first = pack_progressive_frame(plan, buffers, 0, 7)
    if mode == "duplicate":
        assert not assembler.accept_frame(first)
        invalid = first
    else:
        invalid = pack_progressive_frame(plan, buffers, 7, 7)
    with pytest.raises(HandoffError, match="order"):
        assembler.accept_frame(invalid)
    assert assembler.staged_nbytes == 0


def test_final_seal_detects_validly_framed_wrong_bytes_and_discards_them():
    plan, _arrays, buffers = sample_plan()
    changed = dict(buffers)
    changed[plan.tensors[0].name] = bytes([0xA5]) * plan.tensors[0].nbytes
    assembler = stage_all(plan, changed)
    seal = ProgressiveSeal.build(plan, buffers, spark_first_token=42)
    with pytest.raises(HandoffError, match="digest mismatch at seal"):
        assembler.accept_seal(seal)
    assert assembler.staged_nbytes == 0
    with pytest.raises(HandoffError, match="discarded"):
        assembler.buffer_for(plan.tensors[0].name)


def test_cached_producer_digest_detects_storage_change_before_frame_send():
    plan, _arrays, buffers = sample_plan()
    job = ProgressiveJob(plan, generation=7)
    for spec in plan.tensors:
        job.publish_piece(spec.name, 0, buffers[spec.name])

    # The producer hashes each immutable tensor as it is published.  If its
    # backing storage changes before MCDMA copies it, the receiver's one-pass
    # segment verification must reject the frame rather than trusting the
    # cached proof.
    first = plan.tensors[0]
    job._data[first.name][0] ^= 0xFF
    raw = job.frame(0, plan.total_nbytes, timeout=0.1, generation=7)
    with pytest.raises(HandoffError, match="tensor digest mismatch"):
        unpack_progressive_frame(raw)


def test_producer_frame_waits_until_every_requested_byte_is_ready():
    plan, _arrays, buffers = sample_plan()
    job = ProgressiveJob(plan, generation=7)
    started = threading.Event()
    result = {}

    def fetch():
        started.set()
        try:
            result["frame"] = job.frame(
                0, plan.total_nbytes, timeout=1.0, generation=7
            )
        except Exception as exc:  # pragma: no cover - asserted below
            result["error"] = exc

    thread = threading.Thread(target=fetch)
    thread.start()
    assert started.wait(1.0)
    time.sleep(0.02)
    assert thread.is_alive()

    for spec in plan.tensors:
        assert job.publish_piece(spec.name, 0, buffers[spec.name])
    thread.join(1.0)
    assert not thread.is_alive()
    assert "error" not in result
    frame = unpack_progressive_frame(result["frame"])
    assert bytes(frame.payload) == b"".join(buffers[item.name] for item in plan.tensors)


def test_producer_failure_unblocks_waiter_and_discards_buffers():
    plan, _arrays, _buffers = sample_plan()
    job = ProgressiveJob(plan, generation=7)
    started = threading.Event()
    result = {}

    def fetch():
        started.set()
        try:
            job.frame(0, plan.total_nbytes, timeout=1.0, generation=7)
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=fetch)
    thread.start()
    assert started.wait(1.0)
    time.sleep(0.02)
    job.fail("CUDA prefill failed")
    thread.join(1.0)
    assert not thread.is_alive()
    assert isinstance(result.get("error"), HandoffError)
    assert "CUDA prefill failed" in str(result["error"])
    assert job.buffered_nbytes == 0


def test_generation_change_unblocks_old_waiter_and_invalidates_data():
    plan, _arrays, _buffers = sample_plan()
    job = ProgressiveJob(plan, generation=7)
    started = threading.Event()
    result = {}

    def fetch():
        started.set()
        try:
            job.frame(0, plan.total_nbytes, timeout=1.0, generation=7)
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=fetch)
    thread.start()
    assert started.wait(1.0)
    time.sleep(0.02)
    job.invalidate_generation(8)
    thread.join(1.0)
    assert not thread.is_alive()
    assert isinstance(result.get("error"), HandoffError)
    assert "old link generation" in str(result["error"])
    assert job.buffered_nbytes == 0
    with pytest.raises(HandoffError, match="link generation changed"):
        job.frame(0, 1, timeout=0.1, generation=8)


def test_incomplete_job_cannot_seal_and_release_erases_completed_job():
    plan, _arrays, buffers = sample_plan()
    job = ProgressiveJob(plan, generation=7)
    with pytest.raises(HandoffError, match="before every tensor"):
        job.seal(spark_first_token=42, generation=7)

    for spec in plan.tensors:
        job.publish_piece(spec.name, 0, buffers[spec.name])
    seal = job.seal(spark_first_token=42, generation=7)
    assert job.wait_seal(timeout=0.1, generation=7) == seal
    job.release(generation=7)
    assert job.buffered_nbytes == 0
    with pytest.raises(HandoffError, match="released"):
        job.wait_seal(timeout=0.1, generation=7)


def test_producer_rejects_wrong_piece_order_and_bounded_wait_timeout():
    plan, _arrays, buffers = sample_plan()
    job = ProgressiveJob(plan, generation=7)
    second = plan.tensors[1]
    with pytest.raises(HandoffError, match="execution order"):
        job.publish_piece(second.name, 0, buffers[second.name])
    with pytest.raises(HandoffError, match="timed out"):
        job.frame(0, 1, timeout=0.001, generation=7)
    decode_progressive_request,
    encode_progressive_request,
    is_progressive_request,
