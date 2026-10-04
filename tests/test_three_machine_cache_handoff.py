import hashlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from experiments.three_machine import cuda_prefill
from experiments.three_machine import cache_replay
from experiments.three_machine.cache_handoff import (
    ChunkAssembler,
    DEFAULT_FRAME_BYTES,
    MAX_FRAME_BYTES,
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
from experiments.three_machine.progressive_handoff import (
    ProgressiveAssembler,
    ProgressiveJob,
    ProgressivePlan,
    ProgressiveRequest,
    ProgressiveTensorSpec,
    encode_progressive_request,
    pack_progressive_frame,
    unpack_plan,
    unpack_progressive_ack,
    unpack_progressive_frame,
    unpack_seal,
)


MODEL = "Vontra/GLM-5.3-Flash-MLX-4bit-MTP"
REVISION = "76add2a341a1cd90ad0e86bb69839ea9c35827c6"


def archive(arrays=None):
    return TensorArchive(
        arrays or {"kda.0.conv": np.arange(48, dtype=np.int16).reshape(2, 3, 8)},
        model_id=MODEL,
        model_revision=REVISION,
        prompt_sha256=prompt_digest([1, 2, 3, 4]),
        cached_tokens=4,
        mtp_cached_tokens=3,
        spark_first_token=7,
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


def test_rank_zero_serves_progressive_prepare_frames_seal_release_and_shutdown(monkeypatch):
    generation = 9
    job_id = "ab" * 16
    tokens = (1, 2, 3, 4)
    payload = np.arange(8, dtype=np.uint16).tobytes()

    class Exporter:
        comm = object()
        model_id = MODEL
        model_revision = REVISION

        def progressive_plan(self, prompt, *, job_id, generation):
            assert tuple(prompt) == tokens
            return ProgressivePlan.build(
                job_id=job_id,
                generation=generation,
                model_id=self.model_id,
                model_revision=self.model_revision,
                prompt_sha256=prompt_digest(prompt),
                cached_tokens=len(prompt),
                mtp_cached_tokens=len(prompt) - 1,
                tensors=(ProgressiveTensorSpec(
                    ordinal=0,
                    name="cache",
                    dtype=np.dtype(np.uint16).str,
                    shape=(8,),
                    nbytes=len(payload),
                ),),
            )

        def prepare_progressive(self, prompt, job, *, timeout):
            assert tuple(prompt) == tokens and timeout == 5
            job.publish_piece("cache", 0, payload)
            return job.seal(spark_first_token=7, generation=generation)

    exporter = Exporter()
    plan = exporter.progressive_plan(tokens, job_id=job_id, generation=generation)
    prepare = ProgressiveRequest(
        operation="prepare",
        generation=generation,
        job_id=job_id,
        tokens=tokens,
        prompt_sha256=prompt_digest(tokens),
        model_id=MODEL,
        model_revision=REVISION,
    )
    frame = ProgressiveRequest(
        operation="frame", generation=generation, job_id=job_id,
        plan_id=plan.plan_id, offset=0, length=len(payload),
    )
    seal = ProgressiveRequest(
        operation="seal", generation=generation, job_id=job_id,
        plan_id=plan.plan_id,
    )
    # The release identity is filled after the seal reply has been observed.

    class Mailbox:
        def __init__(self):
            self.step = 0
            self.replies = []
            self.accepted = []
            self.bound = []
            self.sealed = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def bind_generation(self, value):
            self.bound.append(value)

        def next_request(self, _timeout):
            self.step += 1
            if self.step == 1:
                return self.step, encode_progressive_request(prepare)
            if self.step == 2:
                return self.step, encode_progressive_request(frame)
            if self.step == 3:
                return self.step, encode_progressive_request(seal)
            if self.step == 4:
                release = ProgressiveRequest(
                    operation="release",
                    generation=generation,
                    job_id=job_id,
                    plan_id=plan.plan_id,
                    transfer_id=self.sealed.transfer_id,
                )
                self.release = release
                return self.step, encode_progressive_request(release)
            return self.step, encode_request(Request(operation="shutdown"))

        def reply(self, sequence, raw):
            self.replies.append((sequence, raw))
            if sequence == 3:
                self.sealed = unpack_seal(raw)

        def wait_reply_accepted(self, sequence, length, timeout):
            self.accepted.append((sequence, length, timeout))

    mailbox = Mailbox()
    controls = []
    monkeypatch.setattr(cuda_prefill, "ServiceMailbox", lambda *_args, **_kwargs: mailbox)
    monkeypatch.setattr(
        cuda_prefill,
        "_control",
        lambda _comm, rank, values: controls.append((rank, values)),
    )
    monkeypatch.setattr(
        cuda_prefill,
        "_token_ids",
        lambda _comm, rank, values, count: list(values),
    )

    assert cuda_prefill.rank_zero(exporter, "test", "/tmp/test.sock", 5) == {
        "rank": 0, "prepared": 1,
    }
    assert unpack_plan(mailbox.replies[0][1]) == plan
    decoded_frame = unpack_progressive_frame(mailbox.replies[1][1])
    assert bytes(decoded_frame.payload) == payload
    assert mailbox.sealed.spark_first_token == 7
    assert unpack_progressive_ack(mailbox.replies[3][1], mailbox.release) == mailbox.sealed.transfer_id
    assert mailbox.bound == [generation] * 4
    assert controls == [
        (0, (cuda_prefill.ACTION_PROGRESSIVE_PREPARE, len(tokens), 0)),
        (0, (cuda_prefill.ACTION_SHUTDOWN, 0, 0)),
    ]
    assert mailbox.accepted and mailbox.accepted[0][2] == 5


def test_cuda_layer_timeline_records_ready_time_without_layer_syncs():
    class Event:
        def __init__(self, clock):
            self.clock = clock
            self.at = None

        def record(self):
            self.at = self.clock[0]

        def elapsed_time(self, other):
            return (other.at - self.at) * 1000.0

    clock = [0.0]
    syncs = []
    torch = SimpleNamespace(cuda=SimpleNamespace(
        Event=lambda **_kwargs: Event(clock),
        synchronize=lambda: syncs.append(True),
    ))
    layers = [SimpleNamespace(index=0, kind="kda"), SimpleNamespace(index=1, kind="dsa")]
    forward = SimpleNamespace()

    def layer_forward(layer, *_args, **_kwargs):
        clock[0] += 0.010 if layer.index == 0 else 0.020

    forward.layer_forward = layer_forward

    def compute(_w, _st, _b, _rows, *_args, **_kwargs):
        clock[0] += 0.002  # embed
        for layer in layers:
            forward.layer_forward(layer)
        clock[0] += 0.003  # norm and head
        return "logits"

    decode = SimpleNamespace(compute=compute)
    timeline = cuda_prefill.CudaLayerTimeline(torch, decode, forward)
    with timeline:
        assert decode.compute(None, None, SimpleNamespace(prefill=True), 33) == "logits"

    report = timeline.report({0: 100, 1: 200})
    assert syncs == [True]
    assert len(report) == 1
    assert report[0]["rows"] == 33
    assert report[0]["compute_seconds"] == pytest.approx(0.035)
    assert report[0]["post_layers_seconds"] == pytest.approx(0.003)
    assert report[0]["layers"] == [
        {
            "index": 0,
            "kind": "kda",
            "ready_seconds": pytest.approx(0.012),
            "layer_seconds": pytest.approx(0.012),
            "cache_bytes": 100,
        },
        {
            "index": 1,
            "kind": "dsa",
            "ready_seconds": pytest.approx(0.032),
            "layer_seconds": pytest.approx(0.020),
            "cache_bytes": 200,
        },
    ]
    assert decode.compute is compute
    assert forward.layer_forward is layer_forward


def test_progressive_plan_is_available_before_contents_in_cuda_execution_order():
    bf16 = object()
    f32 = object()

    def tensor(shape, dtype):
        return SimpleNamespace(shape=shape, dtype=dtype)

    rec = np.empty((1, 1), dtype=object)
    rec[0, 0] = tensor((1, 2, 2), f32)
    state = SimpleNamespace(
        kda_index={0: 0},
        dsa_index={1: 0},
        conv=[tensor((2, 12), bf16)],
        rec=rec,
        kc=[tensor((32, 4), bf16)],
        mtp_kc=tensor((32, 4), bf16),
    )
    exporter = object.__new__(cuda_prefill.CudaPrefillExporter)
    exporter.torch = SimpleNamespace(bfloat16=bf16, float32=f32)
    exporter.comm = SimpleNamespace(world=2)
    exporter.model_id = MODEL
    exporter.model_revision = REVISION
    exporter.handoff_limit = 8
    exporter.w = SimpleNamespace(layers=[
        SimpleNamespace(index=0, kind="kda"),
        SimpleNamespace(index=1, kind="dsa"),
    ])
    exporter.engine = SimpleNamespace(
        st=state,
        pbuf=SimpleNamespace(fnormed=tensor((8, 3), bf16)),
    )

    plan = exporter.progressive_plan(
        [1, 2, 3, 4], job_id="ab" * 16, generation=9
    )
    assert [(spec.name, spec.dtype, spec.shape, spec.nbytes) for spec in plan.tensors] == [
        ("layers.0.conv", "bfloat16", (2, 24), 96),
        ("layers.0.ssm", np.dtype(np.float32).str, (2, 2, 2), 32),
        ("layers.1.keys", "bfloat16", (4, 4), 32),
        ("prompt.final_hidden", "bfloat16", (1, 3), 6),
        ("mtp.keys", "bfloat16", (3, 4), 24),
    ]


def test_progressive_cuda_collector_publishes_only_the_final_chunk(monkeypatch):
    bf16 = object()
    f32 = object()
    u16 = object()

    class Tensor:
        def __init__(self, value, dtype, device="cuda"):
            self.value = np.asarray(value)
            self.dtype = dtype
            self.device = device

        @property
        def shape(self):
            return self.value.shape

        def detach(self):
            return self

        def contiguous(self):
            return self

        def view(self, dtype):
            assert self.dtype is bf16 and dtype is u16
            return Tensor(self.value.view(np.uint16), u16, self.device)

        def copy_(self, source, non_blocking=False):
            assert non_blocking
            self.value[...] = source.value
            return self

        def numpy(self):
            return self.value

        def __getitem__(self, item):
            return Tensor(self.value[item], self.dtype, self.device)

    class Event:
        def record(self):
            return None

        def synchronize(self):
            return None

    class Stream:
        def wait_event(self, _event):
            return None

        def synchronize(self):
            return None

    class StreamContext:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    def empty(shape, *, dtype, device, pin_memory=False):
        assert not pin_memory or device == "cpu"
        numpy_dtype = np.float32 if dtype is f32 else np.uint16
        return Tensor(np.zeros(shape, dtype=numpy_dtype), dtype, device)

    torch = SimpleNamespace(
        bfloat16=bf16,
        float32=f32,
        uint16=u16,
        empty=empty,
        cuda=SimpleNamespace(
            Stream=Stream,
            Event=Event,
            stream=lambda _stream: StreamContext(),
        ),
    )

    class Comm:
        world = 2

        @staticmethod
        def all_gather(send, recv):
            recv.value[0] = send.value
            recv.value[1] = send.value

    layers = [
        SimpleNamespace(index=0, kind="kda"),
        SimpleNamespace(index=1, kind="dsa"),
    ]
    # A KDA convolution shard stores three head-dimension-wide groups.  Keep
    # the fake runtime geometry valid for the production 128-wide head so the
    # test exercises the collector rather than merge_kda_conv's shape guard.
    conv = Tensor(np.arange(2 * 384, dtype=np.uint16).reshape(2, 384), bf16)
    rec = Tensor(np.arange(8, dtype=np.float32).reshape(2, 1, 2, 2), f32)
    keys = Tensor(np.arange(32, dtype=np.uint16).reshape(8, 4), bf16)
    mtp = Tensor((100 + np.arange(32, dtype=np.uint16)).reshape(8, 4), bf16)
    hidden = Tensor(np.arange(24, dtype=np.uint16).reshape(8, 3), bf16)
    state = SimpleNamespace(
        pos=0,
        kda_index={0: 0},
        dsa_index={1: 0},
        conv=[conv],
        rec=rec,
        cur=[0],
        kc=[keys],
        mtp_kc=mtp,
    )
    exporter = object.__new__(cuda_prefill.CudaPrefillExporter)
    exporter.torch = torch
    exporter.comm = Comm()
    exporter.rank = 0
    exporter.model_id = MODEL
    exporter.model_revision = REVISION
    exporter.handoff_limit = 8
    exporter.w = SimpleNamespace(layers=layers)
    exporter.engine = SimpleNamespace(
        st=state,
        pbuf=SimpleNamespace(fnormed=hidden),
    )
    plan = exporter.progressive_plan(
        [1, 2, 3, 4], job_id="ab" * 16, generation=9
    )
    job = ProgressiveJob(plan, generation=9)

    forward = ModuleType("tensorfold.families.glm5_next.cuda.forward")
    def layer_forward(_layer, _w, _state, _buffers, _rows, *_args, **_kwargs):
        return None
    forward.layer_forward = layer_forward
    decode = ModuleType("tensorfold.families.glm5_next.cuda.decode")
    def compute(w, current, buffers, rows, *_args, **_kwargs):
        for layer in layers:
            forward.layer_forward(layer, w, current, buffers, rows)
        return "computed"
    decode.compute = compute
    cuda_package = ModuleType("tensorfold.families.glm5_next.cuda")
    cuda_package.decode = decode
    cuda_package.forward = forward
    monkeypatch.setitem(sys.modules, "tensorfold.families.glm5_next.cuda", cuda_package)
    monkeypatch.setitem(sys.modules, "tensorfold.families.glm5_next.cuda.decode", decode)
    monkeypatch.setitem(sys.modules, "tensorfold.families.glm5_next.cuda.forward", forward)

    collector = cuda_prefill.CudaProgressiveCollector(exporter, plan, job)
    buffers = SimpleNamespace(prefill=True)
    with collector:
        assert decode.compute(None, state, buffers, 2) == "computed"
        assert job.ready_nbytes == 0
        state.pos = 2
        assert decode.compute(None, state, buffers, 2) == "computed"
    collector.finish(hidden[-1:], mtp[:3], timeout=1)
    seal = job.seal(spark_first_token=7, generation=9)
    assembler = ProgressiveAssembler(plan)
    assembler.accept_frame(job.frame(0, plan.total_nbytes, timeout=1, generation=9))
    assembler.accept_seal(seal)
    def bf16_words(name):
        spec = plan.tensor(name)
        return np.frombuffer(assembler.buffer_for(name), dtype=np.uint16).reshape(spec.shape)

    assert bf16_words("layers.0.conv").shape == (2, 768)
    assert assembler.array("layers.0.ssm").shape == (4, 2)
    assert bf16_words("layers.1.keys").shape == (4, 4)
    assert bf16_words("prompt.final_hidden").shape == (1, 3)
    assert bf16_words("mtp.keys").shape == (3, 4)


def test_cache_bytes_are_grouped_by_layer_only():
    arrays = {
        "layers.0.conv": np.zeros((4,), dtype=np.uint8),
        "layers.0.ssm": np.zeros((3,), dtype=np.float32),
        "layers.3.keys": np.zeros((5,), dtype=np.uint16),
        "prompt.final_hidden": np.zeros((7,), dtype=np.uint8),
        "mtp.keys": np.zeros((9,), dtype=np.uint8),
    }
    assert cuda_prefill._cache_bytes_by_layer(arrays) == {0: 16, 3: 10}


def test_manifest_binds_model_prompt_position_and_tensor_digests():
    item = archive()
    decoded = HandoffManifest.from_bytes(item.manifest.to_bytes())
    decoded.require_identity(
        model_id=MODEL,
        model_revision=REVISION,
        prompt_sha256=prompt_digest([1, 2, 3, 4]),
        cached_tokens=4,
        mtp_cached_tokens=3,
    )
    with pytest.raises(HandoffError, match="differs from the decode request"):
        decoded.require_identity(
            model_id=MODEL,
            model_revision=REVISION,
            prompt_sha256=prompt_digest([1, 2, 3, 5]),
            cached_tokens=4,
            mtp_cached_tokens=3,
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
            length=MAX_FRAME_BYTES + 1,
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


def test_frame_payload_can_stay_a_view_until_assembly():
    source = archive({"tensor": np.arange(128, dtype=np.uint8)})
    raw = bytearray(source.frame(0, 128))
    decoded = unpack_frame(memoryview(raw))
    assert isinstance(decoded.payload, memoryview)
    assert decoded.payload.readonly
    assert decoded.payload.obj is raw


def test_tensor_archive_keeps_one_immutable_snapshot():
    original = np.arange(64, dtype=np.uint8)
    source = archive({"tensor": original})
    view = source.bytes_for("tensor")
    assert isinstance(view, memoryview)
    assert view.readonly
    original[:] = 0
    assert bytes(view) == bytes(range(64))


def test_148_mib_cache_uses_three_64_mib_frames_without_allocating_it():
    mib = 1024 * 1024
    data_frames = frame_ranges(148 * mib)
    assert data_frames == (
        (0, 64 * mib),
        (64 * mib, 64 * mib),
        (128 * mib, 20 * mib),
    )
    assert 1 + len(data_frames) + 1 == 5  # prepare + frames + release


def test_148_mib_cache_can_use_one_authenticated_256_mib_frame():
    mib = 1024 * 1024
    assert MAX_FRAME_BYTES == 256 * mib
    assert frame_ranges(148 * mib, MAX_FRAME_BYTES) == ((0, 148 * mib),)


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
    assert "tensor" in assembler._verified
    assert assembler.buffer_for("tensor").obj is assembler._data["tensor"]


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
    # Retain a same-prompt cache from an older protocol.  The current writer
    # must neither parse, replace, nor delete it.
    legacy = tmp_path / source.manifest.prompt_sha256
    legacy.mkdir()
    (legacy / "manifest.bin").write_bytes(b"old-protocol")
    path = persist_archive(source, tmp_path)
    assert path.name == f"v2-{source.manifest.prompt_sha256}"
    assert (legacy / "manifest.bin").read_bytes() == b"old-protocol"
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
