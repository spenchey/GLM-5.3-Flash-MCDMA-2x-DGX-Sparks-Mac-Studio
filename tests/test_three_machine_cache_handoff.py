import hashlib

import numpy as np
import pytest

from experiments.three_machine import cuda_prefill
from experiments.three_machine import cache_replay
from experiments.three_machine.cache_handoff import (
    ChunkAssembler,
    DEFAULT_FRAME_BYTES,
    HandoffError,
    HandoffManifest,
    Request,
    TensorArchive,
    StoredTensorArchive,
    decode_request,
    encode_request,
    frame_ranges,
    merge_kda_conv,
    merge_kda_recurrent,
    prompt_digest,
    pack_ack,
    pack_error,
    pack_manifest,
    persist_archive,
    require_replicated,
    reply_frame,
    unpack_ack,
    unpack_frame,
    unpack_manifest,
)


MODEL = "Vontra/GLM-5.3-Flash-MLX-4bit-MTP"
REVISION = "76add2a341a1cd90ad0e86bb69839ea9c35827c6"


def archive(arrays=None):
    return TensorArchive(
        arrays or {"kda.0.conv": np.arange(48, dtype=np.int16).reshape(2, 3, 8)},
        model_id=MODEL,
        model_revision=REVISION,
        prompt_sha256=prompt_digest([1, 2, 3, 4]),
        cached_tokens=3,
    )


def test_conv_merge_preserves_global_q_k_v_head_order():
    # Two local heads, q/k/v, head width two.  The rank marker makes a naive
    # concat visibly wrong.
    rank0 = np.arange(12, dtype=np.int16).reshape(1, 1, 12)
    rank1 = (100 + np.arange(12, dtype=np.int16)).reshape(1, 1, 12)
    merged = merge_kda_conv([rank0, rank1], head_dim=2).reshape(1, 1, 3, 4, 2)
    assert merged[0, 0, 0].tolist() == [[0, 1], [2, 3], [100, 101], [102, 103]]
    assert merged[0, 0, 1].tolist() == [[4, 5], [6, 7], [104, 105], [106, 107]]
    assert merged[0, 0, 2].tolist() == [[8, 9], [10, 11], [108, 109], [110, 111]]


def test_conv_merge_accepts_live_unbatched_cuda_state():
    rank0 = np.arange(12, dtype=np.int16).reshape(1, 12)
    rank1 = (100 + np.arange(12, dtype=np.int16)).reshape(1, 12)
    merged = merge_kda_conv([rank0, rank1], head_dim=2).reshape(1, 3, 4, 2)
    assert merged[0, 0].tolist() == [[0, 1], [2, 3], [100, 101], [102, 103]]
    assert merged[0, 1].tolist() == [[4, 5], [6, 7], [104, 105], [106, 107]]
    assert merged[0, 2].tolist() == [[8, 9], [10, 11], [108, 109], [110, 111]]


def test_recurrent_merge_selects_each_layers_committed_parity_then_heads():
    rank0 = np.zeros((2, 2, 1, 2, 2), dtype=np.float32)
    rank1 = np.zeros_like(rank0)
    rank0[0, 0], rank0[1, 1] = 10, 11
    rank1[1, 0], rank1[0, 1] = 20, 21
    merged = merge_kda_recurrent([rank0, rank1], [[0, 1], [1, 0]])
    assert merged.shape == (2, 2, 2, 2)
    assert np.all(merged[0, 0] == 10) and np.all(merged[0, 1] == 20)
    assert np.all(merged[1, 0] == 11) and np.all(merged[1, 1] == 21)


def test_replicated_attention_state_must_match_exactly():
    left = np.arange(8, dtype=np.uint16).reshape(2, 4)
    assert np.array_equal(require_replicated([left, left.copy()], name="dsa.0.keys"), left)
    right = left.copy()
    right[0, 0] += 1
    with pytest.raises(HandoffError, match="differs between Sparks"):
        require_replicated([left, right], name="dsa.0.keys")


def test_prefill_shutdown_waits_for_mcdma_to_accept_final_reply(monkeypatch):
    class Mailbox:
        replies = []
        accepted = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def next_request(self, _timeout):
            return 7, encode_request(Request(operation="shutdown"))

        def reply(self, sequence, raw):
            self.replies.append((sequence, raw))

        def wait_reply_accepted(self, sequence, length, timeout):
            self.accepted.append((sequence, length, timeout))

    mailbox = Mailbox()
    controls = []
    monkeypatch.setattr(cuda_prefill, "ServiceMailbox", lambda *_args, **_kwargs: mailbox)
    monkeypatch.setattr(cuda_prefill, "_control", lambda _comm, rank, values: controls.append((rank, values)))
    exporter = type("Exporter", (), {
        "comm": object(), "model_id": MODEL, "model_revision": REVISION,
    })()

    assert cuda_prefill.rank_zero(exporter, "test", "/tmp/test.sock", 5) == {
        "rank": 0, "prepared": 0,
    }
    assert controls == [(0, (cuda_prefill.ACTION_SHUTDOWN, 0, 0))]
    assert mailbox.replies and mailbox.replies[0][0] == 7
    assert mailbox.accepted == [(7, len(mailbox.replies[0][1]), 5)]


def test_manifest_binds_model_prompt_position_and_tensor_digests():
    item = archive()
    decoded = HandoffManifest.from_bytes(item.manifest.to_bytes())
    decoded.require_identity(
        model_id=MODEL,
        model_revision=REVISION,
        prompt_sha256=prompt_digest([1, 2, 3, 4]),
        cached_tokens=3,
    )
    with pytest.raises(HandoffError, match="differs from the decode request"):
        decoded.require_identity(
            model_id=MODEL,
            model_revision=REVISION,
            prompt_sha256=prompt_digest([1, 2, 3, 5]),
            cached_tokens=3,
        )


def test_chunks_round_trip_and_refuse_out_of_order_or_corrupt_data():
    source = archive({"tensor": np.arange(1000, dtype=np.int32)})
    spec = source.manifest.tensor("tensor")
    assembler = ChunkAssembler(source.manifest)
    first = source.chunk("tensor", 0, 257)
    assert not assembler.accept(first)
    with pytest.raises(HandoffError, match="order"):
        assembler.accept(source.chunk("tensor", 514, 257))

    corrupted = bytearray(source.chunk("tensor", 257, 257))
    corrupted[-1] ^= 0xFF
    with pytest.raises(HandoffError, match="digest mismatch"):
        assembler.accept(bytes(corrupted))

    offset = 257
    while offset < spec.nbytes:
        assembler.accept(source.chunk("tensor", offset, 257))
        offset += 257
    result = assembler.array("tensor")
    assert np.array_equal(result, np.arange(1000, dtype=np.int32))
    assert hashlib.sha256(result.tobytes()).hexdigest() == spec.sha256


def test_incomplete_tensor_is_never_returned():
    source = archive({"tensor": np.arange(32, dtype=np.uint8)})
    assembler = ChunkAssembler(source.manifest)
    assembler.accept(source.chunk("tensor", 0, 8))
    with pytest.raises(HandoffError, match="incomplete"):
        assembler.bytes_for("tensor")


def test_prepare_request_binds_the_declared_prompt_digest():
    request = Request(
        operation="prepare",
        tokens=(1, 2, 3, 4),
        prompt_sha256=prompt_digest([1, 2, 3, 4]),
        model_id=MODEL,
        model_revision=REVISION,
    )
    assert decode_request(encode_request(request)) == request
    with pytest.raises(HandoffError, match="does not match"):
        encode_request(Request(
            operation="prepare",
            tokens=(1, 2, 3, 4),
            prompt_sha256=prompt_digest([4, 3, 2, 1]),
            model_id=MODEL,
            model_revision=REVISION,
        ))


def test_chunk_request_is_bounded_and_round_trips():
    request = Request(operation="chunk", transfer_id="a" * 64, tensor="layer.0.conv", offset=19, length=1024)
    assert decode_request(encode_request(request)) == request
    with pytest.raises(HandoffError, match="safe mailbox"):
        encode_request(Request(
            operation="chunk", transfer_id="a" * 64, tensor="layer.0.conv", length=9 * 1024 * 1024
        ))


def test_frame_request_is_bounded_and_round_trips():
    request = Request(operation="frame", transfer_id="a" * 64, offset=19, length=1024)
    assert decode_request(encode_request(request)) == request
    with pytest.raises(HandoffError, match="safe mailbox"):
        encode_request(Request(
            operation="frame", transfer_id="a" * 64,
            length=DEFAULT_FRAME_BYTES + 1,
        ))


def test_multi_tensor_frames_reconstruct_exactly_and_reject_corruption():
    source = archive({
        "a": np.arange(11, dtype=np.uint8),
        "b": np.arange(20, dtype=np.uint16),
        "c": np.arange(9, dtype=np.uint32),
    })
    assembler = ChunkAssembler(source.manifest)
    total = sum(spec.nbytes for spec in source.manifest.tensors)
    frames = [source.frame(offset, length) for offset, length in frame_ranges(total, 17)]
    assert any(len(unpack_frame(frame).segments) > 1 for frame in frames)

    corrupted = bytearray(frames[0])
    corrupted[-1] ^= 1
    with pytest.raises(HandoffError, match="frame digest mismatch"):
        ChunkAssembler(source.manifest).accept_frame(bytes(corrupted))

    for frame in frames:
        assembler.accept_frame(frame)
    assembler.verify_complete()
    for spec in source.manifest.tensors:
        assert assembler.bytes_for(spec.name) == source.bytes_for(spec.name)


def test_148_mib_cache_uses_three_64_mib_frames_without_allocating_it():
    mib = 1024 * 1024
    data_frames = frame_ranges(148 * mib)
    assert data_frames == (
        (0, 64 * mib),
        (64 * mib, 64 * mib),
        (128 * mib, 20 * mib),
    )
    assert 1 + len(data_frames) + 1 == 5  # prepare + frames + release


def test_frame_can_be_built_directly_in_mailbox_reply_area():
    source = archive({"a": np.arange(13, dtype=np.uint8), "b": np.arange(8, dtype=np.uint16)})

    class DirectMailbox:
        def __init__(self):
            self.area = bytearray(4096)
            self.published = None

        def reply_area(self):
            return memoryview(self.area)

        def publish(self, sequence, length):
            self.published = (sequence, length)

        def reply(self, *_args):
            raise AssertionError("bytes fallback must not be used")

    mailbox = DirectMailbox()
    total = sum(spec.nbytes for spec in source.manifest.tensors)
    reply_frame(mailbox, 7, source, 0, total)
    assert mailbox.published is not None
    sequence, length = mailbox.published
    assert sequence == 7
    decoded = unpack_frame(bytes(mailbox.area[:length]))
    assert decoded.payload == b"".join(source.bytes_for(spec.name) for spec in source.manifest.tensors)


def test_verified_buffer_does_not_allocate_a_second_tensor_copy():
    source = archive({"tensor": np.arange(64, dtype=np.uint8)})
    assembler = ChunkAssembler(source.manifest)
    assembler.accept(source.chunk("tensor", 0, 64))
    view = assembler.buffer_for("tensor")
    assert view.readonly
    assert view.obj is assembler._data["tensor"]
    assert bytes(view) == source.bytes_for("tensor")


def test_typed_responses_refuse_errors_and_wrong_acknowledgements():
    source = archive()
    assert unpack_manifest(pack_manifest(source.manifest)) == source.manifest
    assert unpack_ack(pack_ack("release", source.manifest.transfer_id), "release") == source.manifest.transfer_id
    with pytest.raises(HandoffError, match="boom"):
        unpack_ack(pack_error(RuntimeError("boom")), "release")
    with pytest.raises(HandoffError, match="does not match"):
        unpack_ack(pack_ack("shutdown"), "release")


def test_persisted_archive_round_trips_and_refuses_corruption(tmp_path):
    source = archive({"tensor": np.arange(1000, dtype=np.int32)})
    path = persist_archive(source, tmp_path)
    stored = StoredTensorArchive(path)
    try:
        assert stored.manifest == source.manifest
        assembler = ChunkAssembler(stored.manifest)
        assembler.accept(stored.chunk("tensor", 0, 1024))
        offset = 1024
        while offset < stored.manifest.tensor("tensor").nbytes:
            assembler.accept(stored.chunk("tensor", offset, 1024))
            offset += 1024
        assert np.array_equal(assembler.array("tensor"), np.arange(1000, dtype=np.int32))
    finally:
        stored.close()
    tensor_file = path / "000.bin"
    damaged = bytearray(tensor_file.read_bytes())
    damaged[-1] ^= 1
    tensor_file.write_bytes(damaged)
    with pytest.raises(HandoffError, match="failed verification"):
        StoredTensorArchive(path)


def test_gpu_free_replay_serves_only_the_verified_prompt(tmp_path, monkeypatch):
    source = archive({"tensor": np.arange(16, dtype=np.uint8)})
    stored = StoredTensorArchive(persist_archive(source, tmp_path))
    requests = [
        encode_request(Request(
            operation="prepare", tokens=(1, 2, 3, 4),
            prompt_sha256=prompt_digest((1, 2, 3, 4)),
            model_id=MODEL, model_revision=REVISION,
        )),
        encode_request(Request(
            operation="frame", transfer_id=stored.manifest.transfer_id,
            offset=0, length=16,
        )),
        encode_request(Request(operation="release", transfer_id=stored.manifest.transfer_id)),
        encode_request(Request(operation="shutdown")),
    ]

    class Mailbox:
        def __init__(self):
            self.replies = []
            self.accepted = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def next_request(self, _timeout):
            return len(self.replies) + 1, requests[len(self.replies)]

        def reply(self, sequence, raw):
            self.replies.append((sequence, raw))

        def wait_reply_accepted(self, sequence, length, timeout):
            self.accepted.append((sequence, length, timeout))

    mailbox = Mailbox()
    monkeypatch.setattr(cache_replay, "ServiceMailbox", lambda *_args, **_kwargs: mailbox)
    try:
        result = cache_replay.serve(stored, "test", "/tmp/test.sock", 5)
    finally:
        stored.close()
    assert result == {"served": 1, "transfer_id": source.manifest.transfer_id}
    assert unpack_manifest(mailbox.replies[0][1]) == source.manifest
    assembler = ChunkAssembler(source.manifest)
    assert assembler.accept_frame(mailbox.replies[1][1])
    assert assembler.bytes_for("tensor") == source.bytes_for("tensor")
    assert unpack_ack(mailbox.replies[2][1], "release") == source.manifest.transfer_id
    assert unpack_ack(mailbox.replies[3][1], "shutdown") is None
    assert mailbox.accepted
