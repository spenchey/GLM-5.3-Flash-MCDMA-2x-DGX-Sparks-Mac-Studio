import json
import platform
import time
from types import SimpleNamespace

import pytest

from experiments.three_machine import concurrent_capacity
from experiments.three_machine.cache_handoff import HandoffError, prompt_digest


TOKENS = [1, 2, 3]
TEXT = "one two three"
PROMPT_HASH = "9" * 64


def native_result(tokens=TOKENS, text=TEXT, *, first=0.01, total=0.02):
    return {
        "tokens": len(tokens),
        "token_ids": list(tokens),
        "token_sha256": prompt_digest(tokens),
        "text": text,
        "text_sha256": concurrent_capacity._text_digest(text),
        "first_token_seconds": first,
        "seconds": total,
        "finish_reason": "length",
    }


def test_sse_parser_records_first_token_and_exact_ids():
    lines = [
        b'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n',
        b'data: {"choices":[{"delta":{"content":"one two three"},"finish_reason":null}]}\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"completion_tokens":3},"tensorfold":{"token_ids":[1,2,3]}}\n',
        b"data: [DONE]\n",
    ]
    result = concurrent_capacity._parse_sse_lines(lines, time.perf_counter() - 0.1)
    assert result["token_ids"] == TOKENS
    assert result["text"] == TEXT
    assert 0 < result["first_token_seconds"] <= result["seconds"]


def test_sse_parser_fails_closed_without_done_or_token_ids():
    with pytest.raises(HandoffError, match=r"without \[DONE\]"):
        concurrent_capacity._parse_sse_lines([], time.perf_counter())
    lines = [
        b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"length"}],"usage":{"completion_tokens":1}}\n',
        b"data: [DONE]\n",
    ]
    with pytest.raises(HandoffError, match="no token IDs"):
        concurrent_capacity._parse_sse_lines(lines, time.perf_counter())


def test_same_width_native_round_has_client_ttft_and_tail_metrics():
    result = concurrent_capacity.run_native_round(
        prompt="fixed",
        native_url="http://native",
        max_new_tokens=3,
        timeout=1,
        local_reference={"tokens": TOKENS, "text": TEXT, "prompt_sha256": PROMPT_HASH},
        run=1,
        parallel=4,
        native_post=lambda *_args: native_result(),
    )
    assert result["arm"] == "A"
    assert result["allocation"] == {"spark": 4, "mac": 0}
    assert result["parallel"] == 4
    assert result["completion_tokens"] == 12
    assert 0.01 <= result["max_ttft_seconds"] < 0.1
    assert 0.02 <= result["p95_completion_seconds"] < 0.1
    assert len(result["lanes"]) == 4


def test_spark_and_mac_lanes_are_counted_together_and_mac_ttft_includes_handoff():
    def native_post(*_args):
        return native_result()

    def mac_run(*_args, **_kwargs):
        items = [
            {
                "tokens": list(TOKENS),
                "token_sha256": prompt_digest(TOKENS),
                "text": TEXT,
                "first_token_seconds": 0.2,
                "first_token_from_origin_seconds": 1.7,
                "completion_from_origin_seconds": 2.5,
                "generation_seconds": 0.8,
            },
            {
                "tokens": list(TOKENS),
                "token_sha256": prompt_digest(TOKENS),
                "text": TEXT,
                "first_token_seconds": 0.25,
                "first_token_from_origin_seconds": 1.75,
                "completion_from_origin_seconds": 2.5,
                "generation_seconds": 0.75,
            },
        ]
        return {
            "exact_token_match": True,
            "result": items[0],
            "batch": {"results": items},
            "cache_wire_transfers": 1,
            "cache_reuse_copies": 1,
            "transfer_id": "transfer-1",
            "prompt_sha256": PROMPT_HASH,
            "cache_bytes_total": 10,
            "cache_frame_count": 1,
            "cache_frame_lengths": [10],
            "transfer_seconds": 1.0,
            "import_seconds": 0.5,
            "reclaim_seconds": 0.1,
            "pipeline_seconds": 2.6,
        }

    result = concurrent_capacity.run_round(
        object(),
        object(),
        [9, 10],
        prompt="fixed",
        native_url="http://native",
        mailbox_name="test",
        max_new_tokens=3,
        timeout=1,
        model_id="model",
        model_revision="revision",
        local_reference={"tokens": TOKENS, "text": TEXT, "prompt_sha256": PROMPT_HASH},
        run=1,
        spark_parallel=4,
        mac_parallel=2,
        native_post=native_post,
        mac_run=mac_run,
    )
    assert result["arm"] == "B"
    assert result["allocation"] == {"spark": 4, "mac": 2}
    assert result["parallel"] == 6
    assert result["completion_tokens"] == 18
    assert result["aggregate_pipeline_tokens_per_second"] > 0
    mac_lanes = [lane for lane in result["lanes"] if lane["kind"] == "mac"]
    assert [lane["first_token_seconds"] for lane in mac_lanes] == [1.7, 1.75]
    assert result["max_ttft_seconds"] == pytest.approx(1.75)
    assert result["mac"] == {
        "cache_wire_transfers": 1,
        "cache_reuse_copies": 1,
        "transfer_id": "transfer-1",
        "prompt_sha256": PROMPT_HASH,
        "cache_bytes_total": 10,
        "cache_frame_count": 1,
        "cache_frame_lengths": [10],
        "transfer_seconds": 1.0,
        "import_seconds": 0.5,
        "reclaim_seconds": 0.1,
        "pipeline_seconds": 2.6,
    }


def test_candidate_supports_all_mac_allocation_at_width_one():
    item = {
        "tokens": list(TOKENS),
        "token_sha256": prompt_digest(TOKENS),
        "text": TEXT,
        "first_token_seconds": 0.1,
        "first_token_from_origin_seconds": 0.4,
        "completion_from_origin_seconds": 0.7,
        "total_seconds": 0.4,
        "generation_seconds": 0.3,
    }
    result = concurrent_capacity.run_round(
        object(), object(), [9, 10],
        prompt="fixed", native_url="http://native", mailbox_name="test",
        max_new_tokens=3, timeout=1, model_id="model", model_revision="revision",
        local_reference={"tokens": TOKENS, "text": TEXT, "prompt_sha256": PROMPT_HASH}, run=1,
        spark_parallel=0, mac_parallel=1,
        native_post=lambda *_args: native_result(),
        mac_run=lambda *_args, **_kwargs: {
            "result": item, "exact_token_match": True,
            "cache_wire_transfers": 1, "cache_reuse_copies": 0,
            "transfer_id": "transfer-1", "prompt_sha256": PROMPT_HASH,
            "cache_bytes_total": 10, "cache_frame_count": 1,
            "cache_frame_lengths": [10],
            "transfer_seconds": 0.2, "import_seconds": 0.1,
        },
    )
    assert result["allocation"] == {"spark": 0, "mac": 1}
    assert result["max_ttft_seconds"] == pytest.approx(0.4)
    assert result["max_completion_seconds"] == pytest.approx(0.7)


def test_native_round_rejects_byte_mismatch_even_when_tokens_match():
    with pytest.raises(HandoffError, match="bytes differ"):
        concurrent_capacity.run_native_round(
            prompt="fixed", native_url="http://native", max_new_tokens=3,
            timeout=1, local_reference={"tokens": TOKENS, "text": TEXT, "prompt_sha256": PROMPT_HASH},
            run=1, parallel=1,
            native_post=lambda *_args: native_result(text="different"),
        )


def test_script_worker_id_shape_is_accepted_by_exact_pid_record(tmp_path):
    worker_id = "paired-capacity-p16-s8-m8-t256-20261003t010203z-12345"
    path = tmp_path / "worker.json"
    with concurrent_capacity._worker_pidfile(path, worker_id) as record:
        assert json.loads(path.read_text()) == record
        assert record["worker_id"] == worker_id
    assert not path.exists()


def test_runtime_evidence_binds_imported_module_path_and_version(tmp_path, monkeypatch):
    source = tmp_path / "tensorfold-source"
    module_path = source / "tensorfold" / "__init__.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text('__version__ = "0.6.3"\n')
    (source / ".tensorfold-metal-stage.json").write_text(json.dumps({
        "upstream_tag": "v0.6.3",
        "upstream_commit": "9" * 40,
    }))
    python_env = f"{platform.python_version()},mlx=1,mlx-lm=2,numpy=3"
    args = SimpleNamespace(
        tensorfold_source=source,
        expected_tensorfold_tree_sha256="a" * 64,
        expected_python_sha256="b" * 64,
        expected_python_env=python_env,
    )
    monkeypatch.setattr(
        concurrent_capacity.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "a" * 64 + "\n",
    )
    monkeypatch.setattr(concurrent_capacity, "_sha256_file", lambda _path: "b" * 64)
    versions = {"mlx": "1", "mlx-lm": "2", "numpy": "3"}
    monkeypatch.setattr(
        concurrent_capacity.importlib.metadata,
        "version",
        lambda name: versions[name],
    )
    module = SimpleNamespace(__file__=str(module_path), __version__="0.6.3")
    result = concurrent_capacity._runtime_evidence(args, module)
    assert result["tensorfold_module_path"] == str(module_path.resolve())
    assert result["tensorfold_version"] == "0.6.3"

    with pytest.raises(HandoffError, match="outside the frozen source"):
        concurrent_capacity._runtime_evidence(
            args,
            SimpleNamespace(__file__=str(tmp_path / "other.py"), __version__="0.6.3"),
        )
    with pytest.raises(HandoffError, match="version differs"):
        concurrent_capacity._runtime_evidence(
            args,
            SimpleNamespace(__file__=str(module_path), __version__="0.6.2"),
        )
