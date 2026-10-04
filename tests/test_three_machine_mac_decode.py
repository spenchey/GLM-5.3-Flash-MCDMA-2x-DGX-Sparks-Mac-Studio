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
    build_local_full_prompt_continuation_cache,
    build_local_full_prompt_continuation_state,
    build_local_handoff_cache,
    compare_aligned_cache_states,
    compare_kda_intermediates,
    compare_tensor_values,
    DecodeJob,
    fetch_cache,
    fetch_progressive_cache,
    import_cache,
    import_prefix,
    render_user_prompt,
    run_decode,
    run_decode_batch,
    run_handoff_round,
    trace_teacher_forced_divergence,
)
from experiments.three_machine.kda_intermediates import kda_intermediate_names
from experiments.three_machine.progressive_handoff import (
    ProgressiveJob,
    ProgressivePlan,
    ProgressiveRequest,
    ProgressiveTensorSpec,
    decode_progressive_request,
    pack_plan,
    pack_progressive_ack,
    pack_seal,
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


class DraftSlot:
    drafter = object()

    def get(self, _sampling):
        return None


class MeasuredDraftSlot:
    class Proposer:
        @staticmethod
        def telemetry():
            return {"dflash_ms": 12.5, "dflash_proposals": 3}

    proposer = Proposer()


class EmptyDraftSlot:
    proposer = None


class Runtime:
    args = SimpleNamespace(
        index_head_dim=2,
        index_kpool=2,
        index_topk=8,
        kv_lora_rank=4,
        linear_num_heads=2,
        linear_head_dim=2,
        linear_conv=3,
        hidden_size=3,
    )

    def make_cache(self):
        return [KDA(), MLA(), MTP()]


class DFlashRuntime(Runtime):
    def make_cache(self):
        return [KDA(), MLA(), DraftSlot()]


class FakeMX:
    bfloat16 = np.uint16
    float32 = np.float32

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
        "layers.1.keys": np.arange(16, dtype=np.uint16).reshape(4, 4),
        "mtp.keys": (100 + np.arange(12, dtype=np.uint16)).reshape(3, 4),
        "prompt.final_hidden": np.arange(3, dtype=np.uint16).reshape(1, 3),
    }
    if extra:
        arrays.update(extra)
    return TensorArchive(
        arrays,
        model_id=MODEL,
        model_revision=REVISION,
        prompt_sha256=prompt_digest([1, 2, 3, 4]),
        cached_tokens=4,
        mtp_cached_tokens=3,
        spark_first_token=7,
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
    assert cache[0].offset == cache[1].offset == 4
    assert cache[2].offset == 3
    assert cache[0].conv.shape == (2, 12)
    assert cache[0].ssm.shape == (1, 2, 2, 2)
    assert cache[0].ssm.dtype == np.float32
    assert cache[1].keys.tolist() == np.arange(16, dtype=np.uint16).reshape(4, 4).tolist()
    assert cache[1].ik.shape == cache[1].ig.shape == (4, 2)
    assert cache[1].pool.shape == (2, 2)
    assert cache[2].drafted == 0


def test_import_ignores_the_spark_mtp_tensor_for_a_dflash_runtime():
    source = source_archive()
    cache = import_cache(DFlashRuntime(), source.manifest, complete(source), mx_module=FakeMX)
    assert cache[0].offset == cache[1].offset == 4
    assert isinstance(cache[-1], DraftSlot)


def test_import_allows_only_an_explicit_complete_kda_diagnostic_set():
    diagnostic = {
        name: np.zeros((1,), dtype=np.float32)
        for name in kda_intermediate_names(0).values()
    }
    source = source_archive(diagnostic)
    assembler = complete(source)
    with pytest.raises(HandoffError, match="tensor set differs"):
        import_prefix(Runtime(), source.manifest, assembler, mx_module=FakeMX)

    cache, hidden = import_prefix(
        Runtime(),
        source.manifest,
        assembler,
        mx_module=FakeMX,
        diagnostic_kda_layer=0,
    )
    assert len(cache) == 3
    assert hidden.shape == (1, 3)

    incomplete = source_archive(dict(list(diagnostic.items())[:-1]))
    with pytest.raises(HandoffError, match="tensor set differs"):
        import_prefix(
            Runtime(),
            incomplete.manifest,
            complete(incomplete),
            mx_module=FakeMX,
            diagnostic_kda_layer=0,
        )


def test_cache_comparison_identifies_the_worst_changed_tensor():
    source = source_archive()
    imported = import_cache(Runtime(), source.manifest, complete(source), mx_module=FakeMX)
    local = import_cache(Runtime(), source.manifest, complete(source), mx_module=FakeMX)
    local_keys = np.zeros((8, 4), dtype=local[1].keys.dtype)
    local_keys[:4] = local[1].keys
    local[1].keys = local_keys
    local[1].keys[0, 0] += 1

    report = compare_aligned_cache_states(
        imported,
        local,
        source.manifest,
        dense_limit=8,
        mx_module=FakeMX,
    )

    assert report["compared_tensor_count"] == 4
    assert report["exact_tensor_count"] == 3
    assert report["worst_relative_l2_tensor"] == "layers.1.keys"
    assert report["sparse_index_state_used_for_this_prompt"] is False
    changed = next(item for item in report["tensors"] if item["name"] == "layers.1.keys")
    assert changed["exact_elements"] == changed["element_count"] - 1
    assert changed["max_abs"] == 1.0
    assert changed["imported_storage_shape"] == [4, 4]
    assert changed["local_storage_shape"] == [8, 4]


def test_tensor_comparison_rejects_shape_mismatch_and_non_finite_values():
    mismatch = compare_tensor_values(
        np.zeros((2,), dtype=np.float32),
        np.zeros((3,), dtype=np.float32),
    )
    assert mismatch["shape_match"] is False
    assert mismatch["max_abs"] is None
    with pytest.raises(HandoffError, match="non-finite"):
        compare_tensor_values(
            np.asarray([np.nan], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
        )


def test_diagnostic_tensor_comparison_records_mismatch_coordinates():
    imported = np.zeros((2, 4), dtype=np.float32)
    local = imported.copy()
    imported[0, 3] = 1.0
    imported[1, 1] = -2.0

    report = mac_decode._compare_diagnostic_tensor_values(imported, local)

    assert report["mismatch_summary"]["count"] == 2
    assert report["mismatch_summary"]["first_flat_index"] == 3
    assert report["mismatch_summary"]["last_flat_index"] == 5
    assert report["mismatch_summary"]["first"] == [
        {
            "index": [0, 3],
            "imported": 1.0,
            "local": 0.0,
            "difference": 1.0,
        },
        {
            "index": [1, 1],
            "imported": -2.0,
            "local": 0.0,
            "difference": -2.0,
        },
    ]
    assert report["mismatch_summary"]["axes"] == [
        {
            "axis": 0,
            "distinct_indices": 2,
            "minimum_index": 0,
            "maximum_index": 1,
            "top": [{"index": 0, "count": 1}, {"index": 1, "count": 1}],
        },
        {
            "axis": 1,
            "distinct_indices": 2,
            "minimum_index": 1,
            "maximum_index": 3,
            "top": [{"index": 1, "count": 1}, {"index": 3, "count": 1}],
        },
    ]


def test_kda_intermediate_report_finds_the_first_changed_stage():
    names = kda_intermediate_names(0)
    spark = {
        names["input"]: np.arange(6, dtype=np.float32).reshape(2, 3),
        names["projection"]: np.arange(8, dtype=np.float32).reshape(2, 4),
        names["forget_projection"]: np.arange(8, dtype=np.float32).reshape(2, 4),
        names["key"]: np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        names["value"]: np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        names["decay"]: np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        names["beta"]: np.arange(4, dtype=np.float32).reshape(2, 2),
        names["output_gate_projection"]: np.arange(8, dtype=np.float32).reshape(2, 4),
        names["output"]: np.arange(8, dtype=np.float32).reshape(2, 4),
        names["state"]: np.arange(8, dtype=np.float32).reshape(2, 2, 2),
    }
    archive = TensorArchive(
        spark,
        model_id=MODEL,
        model_revision=REVISION,
        prompt_sha256=prompt_digest([1, 2]),
        cached_tokens=2,
        mtp_cached_tokens=1,
        spark_first_token=7,
    )
    assembler = complete(archive)
    local = {
        stage: np.array(spark[name], copy=True)
        for stage, name in names.items()
    }
    for stage in ("key", "value", "decay", "beta", "state"):
        local[stage] = local[stage][None]
    local["decay"][0, 0, 0, 0] += 1

    report = compare_kda_intermediates(
        assembler,
        local,
        layer_index=0,
        mx_module=FakeMX,
    )

    assert report["stage_count"] == 10
    assert report["exact_stage_count"] == 9
    assert report["first_difference"] == "decay"


def test_kda_intermediate_report_localizes_non_finite_values_without_aborting():
    names = kda_intermediate_names(0)
    spark = {
        name: np.arange(4, dtype=np.float32).reshape(2, 2)
        for name in names.values()
    }
    spark[names["key"]][0, 0] = np.nan
    archive = TensorArchive(
        spark,
        model_id=MODEL,
        model_revision=REVISION,
        prompt_sha256=prompt_digest([1, 2]),
        cached_tokens=2,
        mtp_cached_tokens=1,
        spark_first_token=7,
    )
    local = {
        stage: np.array(spark[name], copy=True)
        for stage, name in names.items()
    }
    for stage in ("key", "value", "decay", "beta", "state"):
        local[stage] = local[stage][None]
    local["key"][0, 0, 0] = 0.0

    report = compare_kda_intermediates(
        complete(archive),
        local,
        layer_index=0,
        mx_module=FakeMX,
    )

    assert report["stage_count"] == 10
    assert report["first_difference"] == "key"
    key = next(item for item in report["stages"] if item["stage"] == "key")
    assert key["exact"] is False
    assert key["contains_nonfinite"] is True
    assert key["numerical_metrics_available"] is False
    assert key["imported_nan_count"] == 1
    assert key["local_nan_count"] == 0
    assert key["exact_elements"] is None
    assert key["bitwise_exact_elements"] == key["element_count"] - 1


def test_cache_drafter_telemetry_reports_only_a_real_local_proposer():
    assert mac_decode._cache_drafter_telemetry([object(), MeasuredDraftSlot()]) == {
        "dflash_ms": 12.5,
        "dflash_proposals": 3,
    }
    assert mac_decode._cache_drafter_telemetry([object()]) is None


def test_engine_drafter_proposer_survives_finished_stream_cleanup():
    caller_cache = [object(), EmptyDraftSlot()]
    working_slot = MeasuredDraftSlot()
    working_cache = [object(), working_slot]
    stream = SimpleNamespace(stream_id="request")
    engine = SimpleNamespace(_live=[(stream, working_cache)])

    proposer = mac_decode._engine_drafter_proposer(
        engine, "request", caller_cache
    )
    engine._live.clear()
    working_slot.proposer = None

    assert mac_decode._proposer_telemetry(proposer) == {
        "dflash_ms": 12.5,
        "dflash_proposals": 3,
    }


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
        mtp_cached_tokens=7,
        spark_first_token=source.manifest.spark_first_token,
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


class ProgressiveMailbox:
    generation = 7
    max_reply = 4 + 4 + 64 * 1024 + 17

    def __init__(self, source):
        self.source = source
        self.job = None
        self.seal = None
        self.released = False
        self.requests = []

    def call(self, raw, _timeout):
        request = decode_progressive_request(raw)
        self.requests.append(request)
        if request.operation == "prepare":
            specs = tuple(
                ProgressiveTensorSpec(
                    ordinal=ordinal,
                    name=spec.name,
                    dtype=spec.dtype,
                    shape=spec.shape,
                    nbytes=spec.nbytes,
                )
                for ordinal, spec in enumerate(self.source.manifest.tensors)
            )
            plan = ProgressivePlan.build(
                job_id=request.job_id,
                generation=request.generation,
                model_id=request.model_id,
                model_revision=request.model_revision,
                prompt_sha256=request.prompt_sha256,
                cached_tokens=len(request.tokens),
                mtp_cached_tokens=len(request.tokens) - 1,
                tensors=specs,
            )
            self.job = ProgressiveJob(plan, generation=request.generation)
            for spec in plan.tensors:
                self.job.publish_piece(spec.name, 0, self.source.bytes_for(spec.name))
            self.seal = self.job.seal(
                spark_first_token=self.source.manifest.spark_first_token,
                generation=request.generation,
            )
            return pack_plan(plan)
        assert self.job is not None
        if request.operation == "frame":
            return self.job.frame(
                request.offset,
                request.length,
                timeout=1,
                generation=request.generation,
            )
        if request.operation == "seal":
            assert self.seal is not None
            return pack_seal(self.seal)
        if request.operation == "release":
            self.job.release(generation=request.generation)
            self.released = True
            return pack_progressive_ack(request, request.transfer_id)
        raise AssertionError(request)


def test_progressive_fetch_seals_imports_and_releases_the_cache():
    source = source_archive()
    mailbox = ProgressiveMailbox(source)
    telemetry = {}
    manifest, assembler, frame_lengths = fetch_progressive_cache(
        mailbox,
        [1, 2, 3, 4],
        model_id=MODEL,
        model_revision=REVISION,
        timeout=1,
        chunk_bytes=1024,
        telemetry=telemetry,
    )
    assert manifest.spark_first_token == 7
    assert manifest.protocol == 4
    assert assembler.sealed
    assert mailbox.released
    assert sum(frame_lengths) == sum(spec.nbytes for spec in manifest.tensors)
    assert telemetry["protocol"] == 4
    assert telemetry["effective_frame_bytes"] == 17
    assert [frame["length"] for frame in telemetry["frames"]] == list(frame_lengths)
    assert sum(frame["call_seconds"] for frame in telemetry["frames"]) == pytest.approx(
        telemetry["frame_call_seconds"]
    )
    assert sum(frame["assemble_seconds"] for frame in telemetry["frames"]) == pytest.approx(
        telemetry["frame_assemble_seconds"]
    )
    cache = import_cache(Runtime(), manifest, assembler, mx_module=FakeMX)
    assert cache[0].offset == cache[1].offset == 4
    assert cache[2].offset == 3


def test_progressive_fetch_honors_a_frame_limit_above_32_mib():
    source = source_archive()
    mailbox = ProgressiveMailbox(source)
    mailbox.max_reply = 4 + 4 + 64 * 1024 + 64 * 1024 * 1024
    telemetry = {}
    fetch_progressive_cache(
        mailbox,
        [1, 2, 3, 4],
        model_id=MODEL,
        model_revision=REVISION,
        timeout=1,
        chunk_bytes=64 * 1024 * 1024,
        telemetry=telemetry,
    )
    assert telemetry["effective_frame_bytes"] == 64 * 1024 * 1024


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
    telemetry = {}
    manifest, assembler, frame_lengths = fetch_cache(
        mailbox,
        [1, 2, 3, 4],
        model_id=MODEL,
        model_revision=REVISION,
        timeout=1,
        chunk_bytes=7,
        telemetry=telemetry,
    )
    assert manifest == source.manifest
    assert mailbox.released
    assert sum(frame_lengths) == sum(spec.nbytes for spec in manifest.tensors)
    assert telemetry["archive_bytes"] == sum(frame_lengths)
    assert telemetry["frame_payload_bytes"] == sum(frame_lengths)
    assert telemetry["frame_count"] == len(frame_lengths)
    assert telemetry["frame_call_seconds"] >= 0
    assert telemetry["frame_assemble_seconds"] >= 0
    assert telemetry["prepare_call_seconds"] >= 0
    assert telemetry["release_call_seconds"] >= 0
    assert telemetry["total_seconds"] >= telemetry["frame_call_seconds"]
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


def test_fixed_length_requests_ignore_eos_without_changing_normal_requests():
    tokenizer = SimpleNamespace(eos_token_ids=[2, 9])
    assert mac_decode._request_eos_ids(tokenizer, force_length=False) == frozenset({2, 9})
    assert mac_decode._request_eos_ids(tokenizer, force_length=True) == frozenset()


def test_first_token_difference_reports_content_and_length_drift():
    assert mac_decode._first_token_difference([1, 2, 3], [1, 9, 3]) == 1
    assert mac_decode._first_token_difference([1, 2], [1, 2, 3]) == 2
    assert mac_decode._first_token_difference([1, 2], [1, 2]) is None


def test_local_handoff_cache_mirrors_cuda_prefill_and_final_mtp_absorb(monkeypatch):
    observed = {"hidden": [], "absorbed": [], "evaluated": 0}

    class MX:
        uint32 = np.uint32

        @staticmethod
        def array(value, dtype=None):
            return np.asarray(value, dtype=dtype)

        @staticmethod
        def eval(*_values):
            observed["evaluated"] += 1

    class LocalRuntime:
        def make_cache(self):
            return [SimpleNamespace(keys=np.zeros((1, 1)), state=[])]

        def hidden(self, inputs, cache):
            del cache
            tokens = np.asarray(inputs).reshape(-1).tolist()
            observed["hidden"].append(tokens)
            self.rows = np.asarray([[token, token + 100] for token in tokens])
            return self.rows

        def draft_rows(self):
            return self.rows

        def absorb_draft_context(self, rows, tokens, cache):
            del cache
            observed["absorbed"].append((np.asarray(rows).tolist(), list(tokens)))

    common = ModuleType("tensorfold.engine.family_common")
    common.cache_arrays = lambda _cache: []
    engine = ModuleType("tensorfold.engine")
    engine.family_common = common
    tensorfold = ModuleType("tensorfold")
    tensorfold.engine = engine
    monkeypatch.setitem(sys.modules, "tensorfold", tensorfold)
    monkeypatch.setitem(sys.modules, "tensorfold.engine", engine)
    monkeypatch.setitem(sys.modules, "tensorfold.engine.family_common", common)

    cache = build_local_handoff_cache(
        LocalRuntime(), [10, 11, 12, 13, 14, 15],
        prefill_rows=2, mx_module=MX,
    )

    assert cache
    assert observed["hidden"] == [[10, 11], [12, 13], [14]]
    assert [tokens for _rows, tokens in observed["absorbed"]] == [
        [11, 12], [13, 14], [15]
    ]
    assert observed["absorbed"][-1][0] == [[14, 114]]
    assert observed["evaluated"] == 1


def test_full_prompt_continuation_primes_mtp_with_the_first_reply_token(monkeypatch):
    observed = {"hidden": [], "absorbed": [], "evaluated": 0}

    class MX:
        uint32 = np.uint32

        @staticmethod
        def array(value, dtype=None):
            return np.asarray(value, dtype=dtype)

        @staticmethod
        def eval(*_values):
            observed["evaluated"] += 1

    class LocalRuntime:
        def make_cache(self):
            return [SimpleNamespace(keys=np.zeros((1, 1)), state=[])]

        def hidden(self, inputs, cache):
            del cache
            tokens = np.asarray(inputs).reshape(-1).tolist()
            observed["hidden"].append(tokens)
            self.rows = np.asarray([[token, token + 100] for token in tokens])
            return self.rows

        def draft_rows(self):
            return self.rows

        def absorb_draft_context(self, rows, tokens, cache):
            del cache
            observed["absorbed"].append((np.asarray(rows).tolist(), list(tokens)))

    common = ModuleType("tensorfold.engine.family_common")
    common.cache_arrays = lambda _cache: []
    engine = ModuleType("tensorfold.engine")
    engine.family_common = common
    tensorfold = ModuleType("tensorfold")
    tensorfold.engine = engine
    monkeypatch.setitem(sys.modules, "tensorfold", tensorfold)
    monkeypatch.setitem(sys.modules, "tensorfold.engine", engine)
    monkeypatch.setitem(sys.modules, "tensorfold.engine.family_common", common)

    cache, final_hidden = build_local_full_prompt_continuation_state(
        LocalRuntime(), [10, 11, 12, 13, 14], 99,
        prefill_rows=2, mx_module=MX,
    )

    assert cache
    assert final_hidden.tolist() == [[14, 114]]
    assert observed["hidden"] == [[10, 11], [12, 13], [14]]
    assert [tokens for _rows, tokens in observed["absorbed"]] == [
        [11, 12], [13, 14], [99]
    ]
    assert observed["absorbed"][-1][0] == [[14, 114]]
    assert observed["evaluated"] == 1

    wrapped = build_local_full_prompt_continuation_cache(
        LocalRuntime(), [10, 11], 99,
        prefill_rows=2, mx_module=MX,
    )
    assert wrapped


def test_mac_selects_and_verifies_the_first_token_from_spark_state(monkeypatch):
    absorbed = []

    class MX:
        @staticmethod
        def argmax(value):
            return np.argmax(value)

        @staticmethod
        def eval(*_values):
            return None

    class FirstTokenRuntime:
        @staticmethod
        def head(_hidden):
            return np.asarray([[0.1, 0.2, 0.9, 0.3]])

        @staticmethod
        def absorb_draft_context(hidden, tokens, cache):
            absorbed.append((hidden, list(tokens), cache))

    common = ModuleType("tensorfold.engine.family_common")
    common.cache_arrays = lambda _cache: []
    engine = ModuleType("tensorfold.engine")
    engine.family_common = common
    tensorfold = ModuleType("tensorfold")
    tensorfold.engine = engine
    monkeypatch.setitem(sys.modules, "tensorfold", tensorfold)
    monkeypatch.setitem(sys.modules, "tensorfold.engine", engine)
    monkeypatch.setitem(sys.modules, "tensorfold.engine.family_common", common)

    hidden = np.asarray([[5.0, 6.0]])
    cache = [object()]
    assert mac_decode.prepare_first_token_continuation(
        FirstTokenRuntime(), cache, hidden, expected_token=2, mx_module=MX,
    ) == 2
    assert len(absorbed) == 1
    assert absorbed[0][0] is hidden
    assert absorbed[0][1] == [2]
    assert absorbed[0][2] is cache
    with pytest.raises(HandoffError, match="Spark and Mac first tokens differ"):
        mac_decode.prepare_first_token_continuation(
            FirstTokenRuntime(), cache, hidden, expected_token=1, mx_module=MX,
        )


def test_teacher_forced_trace_stops_at_first_cross_backend_prediction_split():
    class MX:
        uint32 = np.uint32
        float32 = np.float32

        @staticmethod
        def array(value, dtype=None):
            return np.asarray(value, dtype=dtype)

        @staticmethod
        def eval(*_values):
            return None

    class TraceRuntime:
        @staticmethod
        def hidden(inputs, cache):
            fed = int(np.asarray(inputs).reshape(-1)[0])
            slot = cache[0]
            slot.step += 1
            return np.asarray(
                [[float(fed), float(slot.variant), float(slot.step)]],
                dtype=np.float32,
            )

        @staticmethod
        def head(hidden):
            fed = int(hidden[0, 0])
            variant = int(hidden[0, 1])
            logits = np.full((1, 12), -5.0, dtype=np.float32)
            logits[0, fed + 1] = 5.0
            if variant == 1 and fed == 2:
                logits[0, 9] = 6.0
            return logits

    imported = [SimpleNamespace(step=0, variant=1)]
    local = [SimpleNamespace(step=0, variant=0)]
    report = trace_teacher_forced_divergence(
        TraceRuntime(), imported, local, [1, 2, 3, 4], mx_module=MX,
    )

    assert report["first_cross_backend_difference"] == 2
    assert report["first_imported_reference_difference"] == 2
    assert report["first_local_reference_difference"] is None
    assert report["traced_prediction_count"] == 2
    assert report["divergence"]["imported"]["token"] == 9
    assert report["divergence"]["local"]["token"] == 3


def test_mia_workload_separates_short_warmup_from_fixed_measured_rounds(
    monkeypatch, tmp_path,
):
    runtime = object()
    tokenizer = object()
    observed = {"load": [], "local": [], "handoff": [], "prepare": []}
    source = source_archive()
    reusable = mac_decode.ReusablePrefix(
        manifest=source.manifest,
        cache=[object()],
        frame_lengths=(7,),
        transfer_seconds=0.2,
        import_seconds=0.1,
        final_hidden=object(),
        handoff_telemetry={"prepare_call_seconds": 0.08},
    )

    def load(model_dir, *, mtp_drafts, drafter, drafter_bits):
        observed["load"].append((model_dir, mtp_drafts, drafter, drafter_bits))
        return runtime, tokenizer

    def local_decode(*_args, **kwargs):
        observed["local"].append(kwargs)
        return {"tokens": [7, 8, 9], "token_sha256": "same"}

    def handoff(*_args, **kwargs):
        observed["handoff"].append(kwargs)
        count = kwargs["max_new_tokens"]
        return {
            "completion_tokens": count,
            "aggregate_pipeline_tokens_per_second": 8.0,
            "result": {
                "tokens": list(range(count)),
                "token_sha256": "same",
                "completion_from_origin_seconds": 0.5,
                "first_token_from_origin_seconds": 0.1,
                "decode_tokens_per_second": 10.0,
            },
        }

    def prepare(*_args, **kwargs):
        observed["prepare"].append(kwargs)
        return reusable

    monkeypatch.setattr(mac_decode, "_load_runtime", load)
    monkeypatch.setattr(mac_decode, "render_user_prompt", lambda *_args: [1, 2, 3])
    monkeypatch.setattr(mac_decode, "run_decode", local_decode)
    monkeypatch.setattr(mac_decode, "run_handoff_round", handoff)
    monkeypatch.setattr(mac_decode, "prepare_reusable_prefix", prepare)
    monkeypatch.setattr(mac_decode, "_reclaim_mlx_memory", lambda: None)
    args = SimpleNamespace(
        mtp_drafts=1,
        drafter="/models/dflash2",
        drafter_bits=4,
        prompt="prose",
        compare_local=True,
        max_new_tokens=3,
        force_length=True,
        allow_local_drift=True,
        copy_drafts=False,
        copy_min_match=4,
        reuse_warm_prefix=True,
        warmup_new_tokens=2,
        mailbox="test",
        timeout=1,
        model_id=MODEL,
        model_revision=REVISION,
        chunk_mib=1,
        parallel=1,
        warmups=1,
        repeats=3,
    )

    result = mac_decode._run_workload(args, tmp_path)

    assert observed["load"] == [(tmp_path, 1, "/models/dflash2", 4)]
    assert len(observed["prepare"]) == 1
    assert observed["local"][0]["force_length"] is True
    assert observed["local"][0]["copy_drafts"] is False
    assert observed["local"][0]["copy_min_match"] == 4
    assert observed["handoff"][0]["max_new_tokens"] == 2
    assert observed["handoff"][0]["force_length"] is False
    assert observed["handoff"][0]["local_reference"] is None
    assert all(call["reusable_prefix"] is reusable for call in observed["handoff"])
    assert all(call["copy_drafts"] is False for call in observed["handoff"])
    assert all(call["copy_min_match"] == 4 for call in observed["handoff"])
    assert all(call["max_new_tokens"] == 3 for call in observed["handoff"][1:])
    assert all(call["force_length"] is True for call in observed["handoff"][1:])
    assert all(call["strict_local_match"] is False for call in observed["handoff"][1:])
    assert all(call["local_reference"]["tokens"] == [7, 8, 9] for call in observed["handoff"][1:])
    assert result["warmup_new_tokens"] == 2
    assert result["complete_request_seconds"]["median"] == 0.5
    assert result["ttft_seconds"]["median"] == 0.1
    assert result["decode_tokens_per_second"]["median"] == 10.0
    assert result["output_token_sha256"] == "same"
    assert result["reuse_warm_prefix"] is True
    assert result["prefix_setup"]["cache_wire_transfers"] == 1
    assert result["prefix_setup"]["cache_protocol"] == 2
    assert result["prefix_setup"]["handoff_telemetry"]["prepare_call_seconds"] == 0.08
    assert reusable.cache == []
    assert reusable.final_hidden is None


def test_cache_diagnostic_builds_local_reference_without_running_a_benchmark(
    monkeypatch, tmp_path,
):
    runtime = object()
    tokenizer = object()
    observed = {"local": [], "diagnostic": []}

    monkeypatch.setattr(
        mac_decode,
        "_load_runtime",
        lambda *_args, **_kwargs: (runtime, tokenizer),
    )
    monkeypatch.setattr(mac_decode, "render_user_prompt", lambda *_args: [1, 2, 3])

    def local_decode(*_args, **kwargs):
        observed["local"].append(kwargs)
        return {"tokens": [7, 8, 9], "token_sha256": "local"}

    def diagnostic(*args, **kwargs):
        observed["diagnostic"].append((args, kwargs))
        return {"event": "glm_mcdma_cache_state_diagnostic"}

    monkeypatch.setattr(mac_decode, "run_decode", local_decode)
    monkeypatch.setattr(mac_decode, "run_cache_state_diagnostic", diagnostic)
    monkeypatch.setattr(
        mac_decode,
        "run_handoff_round",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("diagnostic mode must not run a timed handoff round")
        ),
    )
    args = SimpleNamespace(
        mtp_drafts=3,
        drafter="/models/dflash2",
        drafter_bits=4,
        prompt="prose",
        compare_local=False,
        diagnose_cache_state=True,
        max_new_tokens=3,
        force_length=True,
        allow_local_drift=False,
        copy_drafts=True,
        copy_min_match=4,
        reuse_warm_prefix=False,
        progressive_handoff=True,
        warmup_new_tokens=None,
        mailbox="test",
        timeout=1,
        model_id=MODEL,
        model_revision=REVISION,
        chunk_mib=96,
        parallel=1,
        warmups=0,
        repeats=1,
    )

    result = mac_decode._run_workload(args, tmp_path)

    assert result["event"] == "glm_mcdma_cache_state_diagnostic"
    assert observed["local"][0]["force_length"] is True
    assert len(observed["diagnostic"]) == 1
    call_args, call_kwargs = observed["diagnostic"][0]
    assert call_args == (runtime, [1, 2, 3], {"tokens": [7, 8, 9], "token_sha256": "local"})
    assert call_kwargs["chunk_bytes"] == 96 * 1024 * 1024
    assert call_kwargs["progressive_handoff"] is True


def test_cache_transplant_plan_covers_families_draft_and_three_ranges():
    cache = [KDA(), KDA(), MLA(), KDA(), MLA(), MLA(), DraftSlot()]

    plan = mac_decode._cache_transplant_plan(cache)
    by_name = {item["name"]: tuple(item["local_indices"]) for item in plan}

    assert by_name["all_local"] == tuple(range(7))
    assert by_name["local_kda_only"] == (0, 1, 3)
    assert by_name["local_mla_only"] == (2, 4, 5)
    assert by_name["local_draft_only"] == (6,)
    assert by_name["local_kda_and_draft"] == (0, 1, 3, 6)
    assert by_name["local_mla_and_draft"] == (2, 4, 5, 6)
    assert by_name["local_layers_without_draft"] == (0, 1, 3, 2, 4, 5)
    assert by_name["local_kda_layers_0_0_only"] == (0,)
    assert by_name["all_local_except_kda_layers_0_0"] == (1, 2, 3, 4, 5, 6)
    assert by_name["local_mla_layers_5_5_only"] == (5,)
    assert by_name["all_local_except_mla_layers_5_5"] == (0, 1, 2, 3, 4, 6)


def test_cache_transplant_plan_rejects_unknown_layer():
    with pytest.raises(HandoffError, match="unsupported cache type"):
        mac_decode._cache_transplant_plan([object(), DraftSlot()])


def test_cache_transplant_trace_uses_fresh_hybrids_and_compact_evidence(monkeypatch):
    imported = [KDA(), MLA(), DraftSlot()]
    local = [KDA(), MLA(), DraftSlot()]
    for index, item in enumerate(imported):
        item.origin = f"imported-{index}"
    for index, item in enumerate(local):
        item.origin = f"local-{index}"
    observed = []
    reclaimed = []

    class LaneEngine:
        @staticmethod
        def copy_single_cache(cache):
            import copy

            return [copy.copy(item) for item in cache]

    lane = ModuleType("tensorfold.engine.lane_engine")
    lane.LaneEngine = LaneEngine
    engine = ModuleType("tensorfold.engine")
    engine.lane_engine = lane
    tensorfold = ModuleType("tensorfold")
    tensorfold.engine = engine
    monkeypatch.setitem(sys.modules, "tensorfold", tensorfold)
    monkeypatch.setitem(sys.modules, "tensorfold.engine", engine)
    monkeypatch.setitem(sys.modules, "tensorfold.engine.lane_engine", lane)

    def trace(_runtime, hybrid, reference, tokens, **_kwargs):
        hybrid_origins = [item.origin for item in hybrid]
        reference_origins = [item.origin for item in reference]
        observed.append((hybrid_origins, reference_origins, list(tokens)))
        exact = all(value.startswith("local-") for value in hybrid_origins)
        difference = None if exact else 4
        return {
            "traced_prediction_count": 5 if exact else 4,
            "first_imported_reference_difference": difference,
            "first_local_reference_difference": None,
            "first_cross_backend_difference": difference,
            "divergence": None if exact else {"generated_token_index": 4},
        }

    monkeypatch.setattr(mac_decode, "trace_teacher_forced_divergence", trace)
    monkeypatch.setattr(mac_decode, "_reclaim_mlx_memory", lambda: reclaimed.append(True))

    results = mac_decode.trace_cache_transplant_variants(
        object(), imported, local, [10, 11, 12, 13, 14, 15]
    )
    by_name = {item["name"]: item for item in results}

    assert by_name["all_local"]["first_cross_backend_difference"] is None
    assert by_name["local_kda_only"]["first_cross_backend_difference"] == 4
    assert by_name["local_kda_only"]["local_cache_indices"] == [0]
    assert by_name["local_kda_only"]["local_draft_cache"] is False
    assert by_name["local_draft_only"]["local_draft_cache"] is True
    assert all(reference == ["local-0", "local-1", "local-2"] for _, reference, _ in observed)
    assert all(tokens == [10, 11, 12, 13, 14, 15] for _, _, tokens in observed)
    assert len(reclaimed) == len(results)


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
            self.rounds = 0
            self.drafted = 0
            self.accepted = 0
            self.proposer = None

    class Proposer:
        def __init__(self, *, min_match):
            self.min_match = min_match

        def telemetry(self):
            return {"min_match": self.min_match}

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
                stream.rounds += 1
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
    lane.SuffixLookupProposer = Proposer
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
    assert observed == {
        "model_dir": tmp_path,
        "kwargs": {"mtp_drafts": 3, "drafter": "", "drafter_bits": 4},
    }

    assert mac_decode._load_runtime(
        tmp_path,
        mtp_drafts=2,
        drafter="/models/dflash2",
        drafter_bits=8,
    ) == loaded
    assert observed == {
        "model_dir": tmp_path,
        "kwargs": {"mtp_drafts": 2, "drafter": "/models/dflash2", "drafter_bits": 8},
    }

    with pytest.raises(ValueError, match="1..8"):
        mac_decode._load_runtime(tmp_path, mtp_drafts=0)


def test_copy_proposer_matches_server_policy(monkeypatch):
    observed = {}

    class Proposer:
        def __init__(self, *, min_match):
            observed["min_match"] = min_match

    lane = ModuleType("tensorfold.engine.lane_engine")
    lane.SuffixLookupProposer = Proposer
    engine = ModuleType("tensorfold.engine")
    engine.lane_engine = lane
    tensorfold = ModuleType("tensorfold")
    tensorfold.engine = engine
    monkeypatch.setitem(sys.modules, "tensorfold", tensorfold)
    monkeypatch.setitem(sys.modules, "tensorfold.engine", engine)
    monkeypatch.setitem(sys.modules, "tensorfold.engine.lane_engine", lane)

    assert isinstance(mac_decode._copy_proposer(enabled=True, min_match=4), Proposer)
    assert observed == {"min_match": 4}
    assert mac_decode._copy_proposer(enabled=False, min_match=4) is None
    with pytest.raises(ValueError, match="positive"):
        mac_decode._copy_proposer(enabled=True, min_match=0)


def test_decode_timing_starts_when_the_first_token_is_emitted(monkeypatch):
    lifecycle = _fake_decode_engine(monkeypatch)
    tokenizer = SimpleNamespace(eos_token_id=99, decode=lambda tokens, **_kwargs: repr(tokens))
    result = run_decode(
        Runtime(), tokenizer, [10, 11, 12], max_new_tokens=3,
        cache=[object(), MeasuredDraftSlot()], cached_tokens=2,
    )
    assert result["tokens"] == [1, 2, 3]
    assert result["first_token_seconds"] > 0
    assert result["generation_seconds"] > 0
    assert result["subsequent_tokens_per_second"] > 0
    assert result["end_to_end_tokens_per_second"] > 0
    assert result["copy_drafts"] is True
    assert result["drafter_telemetry"] == {
        "dflash_ms": 12.5,
        "dflash_proposals": 3,
    }
    assert lifecycle == {"release_rounds": 1, "reset": 1, "reclaim": 1}


def test_batch_decode_reports_real_aggregate_throughput(monkeypatch):
    lifecycle = _fake_decode_engine(monkeypatch)
    tokenizer = SimpleNamespace(eos_token_id=99, decode=lambda tokens, **_kwargs: repr(tokens))
    jobs = [
        DecodeJob(
            f"request-{index}", (10, 11, 12), 3,
            [object(), MeasuredDraftSlot()], 2,
        )
        for index in range(2)
    ]
    result = run_decode_batch(Runtime(), tokenizer, jobs)
    assert result["parallel"] == 2
    assert result["tokens"] == 6
    assert result["decode_tokens"] == 4
    assert result["first_token_seconds"] > 0
    assert result["aggregate_decode_tokens_per_second"] > 0
    assert result["copy_drafts"] is True
    assert [item["tokens"] for item in result["results"]] == [[1, 2, 3], [1, 2, 3]]
    assert [item["drafter_telemetry"] for item in result["results"]] == [
        {"dflash_ms": 12.5, "dflash_proposals": 3},
        {"dflash_ms": 12.5, "dflash_proposals": 3},
    ]
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
    monkeypatch.setattr(mac_decode, "import_prefix", lambda *_args: ([object()], object()))
    monkeypatch.setattr(mac_decode, "prepare_first_token_continuation", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(mac_decode, "_reclaim_mlx_memory", lambda: None)
    monkeypatch.setattr(mac_decode, "run_decode_batch", lambda _runtime, _tokenizer, jobs, **_kwargs: {
        "results": [
            {
                "tokens": [2, 3],
                "token_sha256": "continued",
                "total_seconds": 0.01,
                "first_token_from_origin_seconds": 0.005,
                "completion_from_origin_seconds": 0.01,
            }
            for _job in jobs
        ],
        "total_seconds": 0.01,
    })

    result = run_handoff_round(
        Runtime(), SimpleNamespace(decode=lambda tokens, **_kwargs: repr(tokens)), [1, 2, 3, 4],
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


def test_handoff_reuses_retained_prefix_without_another_wire_transfer(monkeypatch):
    source = source_archive()
    retained_cache = [object()]
    reusable = mac_decode.ReusablePrefix(
        manifest=source.manifest,
        cache=retained_cache,
        frame_lengths=(sum(spec.nbytes for spec in source.manifest.tensors),),
        transfer_seconds=0.2,
        import_seconds=0.1,
        final_hidden=object(),
    )
    calls = {"copy": 0, "decode": 0}

    class LaneEngine:
        @staticmethod
        def copy_single_cache(cache):
            calls["copy"] += 1
            assert cache is retained_cache
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

    class NoMailbox:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("a retained prefix must not open MCDMA again")

    def decode(_runtime, _tokenizer, _prompt_ids, **_kwargs):
        calls["decode"] += 1
        assert _kwargs["copy_drafts"] is False
        assert _kwargs["copy_min_match"] == 7
        return {
            "tokens": [2, 3],
            "token_sha256": "continued",
            "total_seconds": 0.01,
            "first_token_from_origin_seconds": 0.005,
            "completion_from_origin_seconds": 0.01,
        }

    monkeypatch.setattr(mac_decode, "ClientMailbox", NoMailbox)
    monkeypatch.setattr(mac_decode, "prepare_first_token_continuation", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        mac_decode,
        "import_cache",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("a retained prefix must not be imported again")
        ),
    )
    monkeypatch.setattr(mac_decode, "run_decode", decode)
    monkeypatch.setattr(mac_decode, "_reclaim_mlx_memory", lambda: None)

    result = run_handoff_round(
        Runtime(), SimpleNamespace(decode=lambda tokens, **_kwargs: repr(tokens)), [1, 2, 3, 4],
        parallel=1, max_new_tokens=3, mailbox_name="test", timeout=1,
        model_id=MODEL, model_revision=REVISION,
        local_reference={"tokens": [1, 2, 3], "token_sha256": "same"},
        reusable_prefix=reusable,
        copy_drafts=False,
        copy_min_match=7,
    )

    assert calls == {"copy": 1, "decode": 1}
    assert retained_cache == reusable.cache
    assert result["cache_wire_transfers"] == 0
    assert result["cache_reuse_copies"] == 1
    assert result["cache_bytes_total"] == 0
    assert result["cache_frame_count"] == 0
    assert result["warm_prefix_reused"] is True
    assert result["exact_token_match"] is True
    assert result["copy_drafts"] is False
    assert result["copy_min_match"] == 7


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
