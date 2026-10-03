import json
import sys
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest

from experiments.three_machine.cache_handoff import (
    ChunkAssembler,
    HandoffError,
    Request,
    TensorArchive,
    decode_request,
    pack_ack,
    pack_manifest,
    prompt_digest,
)
from experiments.three_machine import mac_decode
from experiments.three_machine.mac_decode import (
    DecodeJob,
    fetch_cache,
    import_cache,
    render_user_prompt,
    run_decode,
    run_decode_batch,
    run_handoff_round,
)


MODEL = "Vontra/GLM-5.3-Flash-MLX-4bit-MTP"
REVISION = "76add2a341a1cd90ad0e86bb69839ea9c35827c6"


class KDA:
    def __init__(self):
        self.conv = self.ssm = None
        self.offset = 0


class MLA:
    def __init__(self):
        self.keys = self.ik = self.ig = self.pool = None
        self.offset = 0


class MTP(MLA):
    def __init__(self):
        super().__init__()
        self.drafted = 99


class Runtime:
    args = SimpleNamespace(
        index_head_dim=2,
        index_kpool=2,
        index_topk=8,
        kv_lora_rank=4,
        linear_num_heads=2,
        linear_head_dim=2,
        linear_conv=3,
    )

    def make_cache(self):
        return [KDA(), MLA(), MTP()]


class FakeMX:
    bfloat16 = np.uint16

    @staticmethod
    def array(value):
        return np.array(value, copy=True)

    @staticmethod
    def zeros(shape, dtype):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def eval(*_values):
        return None


def source_archive(extra=None):
    arrays = {
        "layers.0.conv": np.arange(24, dtype=np.uint16).reshape(2, 12),
        "layers.0.ssm": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        "layers.1.keys": np.arange(12, dtype=np.uint16).reshape(3, 4),
        "mtp.keys": (100 + np.arange(12, dtype=np.uint16)).reshape(3, 4),
    }
    if extra:
        arrays.update(extra)
    return TensorArchive(
        arrays,
        model_id=MODEL,
        model_revision=REVISION,
        prompt_sha256=prompt_digest([1, 2, 3, 4]),
        cached_tokens=3,
        dtype_names={name: "bfloat16" for name in arrays if not name.endswith(".ssm")},
    )


def complete(source):
    assembler = ChunkAssembler(source.manifest)
    for spec in source.manifest.tensors:
        assembler.accept(source.chunk(spec.name, 0, spec.nbytes))
    return assembler


def test_import_builds_every_glm_cache_at_the_verified_position():
    source = source_archive()
    cache = import_cache(Runtime(), source.manifest, complete(source), mx_module=FakeMX)
    assert cache[0].offset == cache[1].offset == cache[2].offset == 3
    assert cache[0].conv.shape == (2, 12)
    assert cache[0].ssm.shape == (1, 2, 2, 2)
    assert cache[0].ssm.dtype == np.float32
    assert cache[1].keys.tolist() == np.arange(12, dtype=np.uint16).reshape(3, 4).tolist()
    assert cache[1].ik.shape == cache[1].ig.shape == (3, 2)
    assert cache[1].pool.shape == (2, 2)
    assert cache[2].drafted == 0


def test_import_refuses_missing_extra_and_sparse_boundary_state():
    extra = source_archive({"unexpected": np.zeros((1,), dtype=np.uint8)})
    with pytest.raises(HandoffError, match="tensor set differs"):
        import_cache(Runtime(), extra.manifest, complete(extra), mx_module=FakeMX)

    source = source_archive()
    manifest = source.manifest.__class__.build(
        model_id=MODEL,
        model_revision=REVISION,
        prompt_sha256=source.manifest.prompt_sha256,
        cached_tokens=8,
        tensors=source.manifest.tensors,
    )
    assembler = ChunkAssembler(manifest)
    assembler._data = complete(source)._data
    with pytest.raises(HandoffError, match="dense-attention limit"):
        import_cache(Runtime(), manifest, assembler, mx_module=FakeMX)


class Mailbox:
    def __init__(self, source):
        self.source = source
        self.released = False
        self.frame_requests = []

    def call(self, raw, _timeout):
        request = decode_request(raw)
        if request.operation == "prepare":
            return pack_manifest(self.source.manifest)
        if request.operation == "chunk":
            return self.source.chunk(request.tensor, request.offset, request.length)
        if request.operation == "frame":
            self.frame_requests.append((request.offset, request.length))
            return self.source.frame(request.offset, request.length)
        if request.operation == "release":
            self.released = True
            return pack_ack("release", request.transfer_id)
        raise AssertionError(request)


class BoundedMailbox(Mailbox):
    # fetch_cache reserves the protocol's maximum header before choosing a
    # payload, so this leaves exactly seven bytes for each test frame.
    max_reply = 4 + 4 + 64 * 1024 + 7

    def __init__(self, source):
        super().__init__(source)
        self.frame_lengths = []
        self.reply_lengths = []

    def call(self, raw, timeout):
        request = decode_request(raw)
        if request.operation == "frame":
            self.frame_lengths.append(request.length)
        answer = super().call(raw, timeout)
        if request.operation == "frame":
            self.reply_lengths.append(len(answer))
        return answer


def test_fetch_clamps_large_chunks_to_the_live_reply_mailbox():
    source = source_archive()
    mailbox = BoundedMailbox(source)
    manifest, assembler, frame_lengths = fetch_cache(
        mailbox,
        [1, 2, 3, 4],
        model_id=MODEL,
        model_revision=REVISION,
        timeout=1,
        chunk_bytes=4 * 1024 * 1024,
    )
    assert manifest == source.manifest
    assert mailbox.frame_lengths
    assert max(mailbox.frame_lengths) == 7
    assert tuple(mailbox.frame_lengths) == frame_lengths
    assert max(mailbox.reply_lengths) <= mailbox.max_reply
    assert mailbox.released
    for spec in manifest.tensors:
        assert assembler.bytes_for(spec.name) == source.bytes_for(spec.name)


def test_fetch_verifies_all_chunks_and_releases_the_remote_archive():
    source = source_archive()
    mailbox = Mailbox(source)
    manifest, assembler, frame_lengths = fetch_cache(
        mailbox,
        [1, 2, 3, 4],
        model_id=MODEL,
        model_revision=REVISION,
        timeout=1,
        chunk_bytes=7,
    )
    assert manifest == source.manifest
    assert mailbox.released
    assert sum(frame_lengths) == sum(spec.nbytes for spec in manifest.tensors)
    for spec in manifest.tensors:
        assert assembler.bytes_for(spec.name) == source.bytes_for(spec.name)


def test_fetch_spans_many_tensors_in_one_mailbox_frame():
    source = source_archive()
    mailbox = Mailbox(source)
    manifest, assembler, frame_lengths = fetch_cache(
        mailbox,
        [1, 2, 3, 4],
        model_id=MODEL,
        model_revision=REVISION,
        timeout=1,
        chunk_bytes=64 * 1024 * 1024,
    )
    assert mailbox.frame_requests == [(0, sum(spec.nbytes for spec in manifest.tensors))]
    assert frame_lengths == (sum(spec.nbytes for spec in manifest.tensors),)
    assert mailbox.released
    for spec in manifest.tensors:
        assert assembler.bytes_for(spec.name) == source.bytes_for(spec.name)


def test_fetch_releases_the_remote_archive_after_a_corrupt_chunk():
    class CorruptMailbox(Mailbox):
        def call(self, raw, timeout):
            request = decode_request(raw)
            answer = super().call(raw, timeout)
            if request.operation == "frame":
                changed = bytearray(answer)
                changed[-1] ^= 1
                return bytes(changed)
            return answer

    mailbox = CorruptMailbox(source_archive())
    with pytest.raises(HandoffError, match="digest mismatch"):
        fetch_cache(
            mailbox,
            [1, 2, 3, 4],
            model_id=MODEL,
            model_revision=REVISION,
            timeout=1,
            chunk_bytes=7,
        )
    assert mailbox.released


def test_fetch_releases_after_fail_closed_manifest_identity_check():
    mailbox = Mailbox(source_archive())
    with pytest.raises(HandoffError, match="differs from the decode request"):
        fetch_cache(
            mailbox,
            [1, 2, 3, 4],
            model_id="wrong/model",
            model_revision=REVISION,
            timeout=1,
        )
    assert mailbox.released


def test_user_prompt_uses_tensorfolds_thinking_off_chat_template(monkeypatch):
    observed = {}

    def render(_tokenizer, messages, **kwargs):
        observed["messages"] = messages
        observed["kwargs"] = kwargs
        return [7, 8, 9]

    text = ModuleType("tensorfold.server.text")
    text.render_prompt_ids = render
    server = ModuleType("tensorfold.server")
    server.text = text
    tensorfold = ModuleType("tensorfold")
    tensorfold.server = server
    monkeypatch.setitem(sys.modules, "tensorfold", tensorfold)
    monkeypatch.setitem(sys.modules, "tensorfold.server", server)
    monkeypatch.setitem(sys.modules, "tensorfold.server.text", text)

    assert render_user_prompt(object(), "hello") == [7, 8, 9]
    assert observed["messages"] == [{"role": "user", "content": "hello"}]
    assert observed["kwargs"] == {
        "enable_thinking": False,
        "add_generation_prompt": True,
    }


def _fake_decode_engine(monkeypatch):
    lifecycle = {"release_rounds": 0, "reset": 0, "reclaim": 0}

    class Stream:
        def __init__(self, stream_id, prompt_ids, max_new_tokens, eos_ids, drafts):
            self.stream_id = stream_id
            self.prompt_ids = prompt_ids
            self.max_new_tokens = max_new_tokens
            self.eos_ids = eos_ids
            self.drafts = drafts
            self.emitted = []
            self.finished = False
            self.finish_reason = ""

    class Engine:
        def __init__(self):
            self.streams = []

        @property
        def active_count(self):
            return sum(not stream.finished for stream in self.streams)

        def add_stream(self, stream, *, cache, cached_tokens):
            assert cache
            assert cached_tokens == len(stream.prompt_ids) - 1
            self.streams.append(stream)

        def step(self):
            for stream in self.streams:
                if stream.finished:
                    continue
                stream.emitted.append(len(stream.emitted) + 1)
                if len(stream.emitted) >= stream.max_new_tokens:
                    stream.finished = True
                    stream.finish_reason = "length"

        def release_rounds(self):
            lifecycle["release_rounds"] += 1

        def reset(self):
            lifecycle["reset"] += 1
            self.streams.clear()

    lane = ModuleType("tensorfold.engine.lane_engine")
    lane.LaneStream = Stream
    engine = ModuleType("tensorfold.engine")
    engine.lane_engine = lane
    tensorfold = ModuleType("tensorfold")
    tensorfold.engine = engine
    monkeypatch.setitem(sys.modules, "tensorfold", tensorfold)
    monkeypatch.setitem(sys.modules, "tensorfold.engine", engine)
    monkeypatch.setitem(sys.modules, "tensorfold.engine.lane_engine", lane)
    monkeypatch.setattr(mac_decode, "_new_engine", lambda _runtime, imported: Engine())
    monkeypatch.setattr(
        mac_decode,
        "_reclaim_mlx_memory",
        lambda: lifecycle.__setitem__("reclaim", lifecycle["reclaim"] + 1),
    )
    return lifecycle


def test_runtime_loader_uses_public_tensorfold_residency_entry_point(monkeypatch, tmp_path):
    observed = {}
    loaded = (object(), object())

    def load(model_dir, **kwargs):
        observed["model_dir"] = model_dir
        observed["kwargs"] = kwargs
        return loaded

    family = ModuleType("tensorfold.families.glm5_next")
    family.load = load
    families = ModuleType("tensorfold.families")
    families.glm5_next = family
    tensorfold = ModuleType("tensorfold")
    tensorfold.families = families
    monkeypatch.setitem(sys.modules, "tensorfold", tensorfold)
    monkeypatch.setitem(sys.modules, "tensorfold.families", families)
    monkeypatch.setitem(sys.modules, "tensorfold.families.glm5_next", family)

    assert mac_decode._load_runtime(tmp_path) == loaded
    assert observed == {"model_dir": tmp_path, "kwargs": {"mtp_drafts": 3}}


def test_decode_timing_starts_when_the_first_token_is_emitted(monkeypatch):
    lifecycle = _fake_decode_engine(monkeypatch)
    tokenizer = SimpleNamespace(eos_token_id=99, decode=lambda tokens, **_kwargs: repr(tokens))
    result = run_decode(
        Runtime(), tokenizer, [10, 11, 12], max_new_tokens=3,
        cache=[object()], cached_tokens=2,
    )
    assert result["tokens"] == [1, 2, 3]
    assert result["first_token_seconds"] > 0
    assert result["generation_seconds"] > 0
    assert result["subsequent_tokens_per_second"] > 0
    assert result["end_to_end_tokens_per_second"] > 0
    assert lifecycle == {"release_rounds": 1, "reset": 1, "reclaim": 1}


def test_batch_decode_reports_real_aggregate_throughput(monkeypatch):
    lifecycle = _fake_decode_engine(monkeypatch)
    tokenizer = SimpleNamespace(eos_token_id=99, decode=lambda tokens, **_kwargs: repr(tokens))
    jobs = [
        DecodeJob(f"request-{index}", (10, 11, 12), 3, [object()], 2)
        for index in range(2)
    ]
    result = run_decode_batch(Runtime(), tokenizer, jobs)
    assert result["parallel"] == 2
    assert result["tokens"] == 6
    assert result["decode_tokens"] == 4
    assert result["first_token_seconds"] > 0
    assert result["aggregate_decode_tokens_per_second"] > 0
    assert [item["tokens"] for item in result["results"]] == [[1, 2, 3], [1, 2, 3]]
    assert lifecycle == {"release_rounds": 1, "reset": 1, "reclaim": 1}


def test_handoff_fetches_one_cache_and_reuses_it_for_all_lanes(monkeypatch):
    source = source_archive()
    calls = {"fetch": 0, "copy": 0}

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return object()

        def __exit__(self, *_args):
            return None

    def one_fetch(*_args, **_kwargs):
        calls["fetch"] += 1
        total = sum(spec.nbytes for spec in source.manifest.tensors)
        return source.manifest, complete(source), (total,)

    class LaneEngine:
        @staticmethod
        def copy_single_cache(cache):
            calls["copy"] += 1
            return list(cache)

    lane = ModuleType("tensorfold.engine.lane_engine")
    lane.LaneEngine = LaneEngine
    engine = ModuleType("tensorfold.engine")
    engine.lane_engine = lane
    tensorfold = ModuleType("tensorfold")
    tensorfold.engine = engine
    monkeypatch.setitem(sys.modules, "tensorfold", tensorfold)
    monkeypatch.setitem(sys.modules, "tensorfold.engine", engine)
    monkeypatch.setitem(sys.modules, "tensorfold.engine.lane_engine", lane)
    monkeypatch.setattr(mac_decode, "ClientMailbox", Client)
    monkeypatch.setattr(mac_decode, "fetch_cache", one_fetch)
    monkeypatch.setattr(mac_decode, "import_cache", lambda *_args: [object()])
    monkeypatch.setattr(mac_decode, "_reclaim_mlx_memory", lambda: None)
    monkeypatch.setattr(mac_decode, "run_decode_batch", lambda _runtime, _tokenizer, jobs, **_kwargs: {
        "results": [
            {
                "tokens": [1, 2, 3],
                "total_seconds": 0.01,
                "first_token_from_origin_seconds": 0.005,
                "completion_from_origin_seconds": 0.01,
            }
            for _job in jobs
        ],
        "total_seconds": 0.01,
    })

    result = run_handoff_round(
        Runtime(), SimpleNamespace(), [1, 2, 3, 4],
        parallel=8, max_new_tokens=3, mailbox_name="test", timeout=1,
        model_id=MODEL, model_revision=REVISION,
        local_reference={"tokens": [1, 2, 3]},
    )
    assert calls == {"fetch": 1, "copy": 7}
    assert result["cache_wire_transfers"] == 1
    assert result["cache_reuse_copies"] == 7
    assert result["parallel"] == 8
    assert result["completion_tokens"] == 24
    assert result["reclaim_seconds"] >= 0
    assert result["pipeline_seconds"] >= result["reclaim_seconds"]


def test_cli_keeps_tensorfold_notices_out_of_json_stdout(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        mac_decode,
        "_run_workload",
        lambda _args, _model: (print("tensorfold notice"), {"event": "proof"})[1],
    )
    monkeypatch.setattr(sys, "argv", [
        "mac_decode", "--model", str(tmp_path), "--prompt", "hello",
    ])
    assert mac_decode.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"event": "proof"}
    assert "tensorfold notice" in captured.err
