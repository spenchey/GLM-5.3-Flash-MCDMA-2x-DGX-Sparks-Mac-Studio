import importlib.util
from pathlib import Path

from experiments.three_machine.cache_handoff import prompt_digest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_native_portable", ROOT / "scripts" / "benchmark-native-portable.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def response(tokens=(1, 2, 3)):
    return {
        "choices": [{"finish_reason": "length", "token_ids": list(tokens)}],
        "usage": {"prompt_tokens": 10, "completion_tokens": len(tokens)},
        "tensorfold": {"token_ids": list(tokens)},
    }


def test_parallel_native_batch_reports_aggregate_pipeline_rate(monkeypatch):
    monkeypatch.setattr(benchmark, "post", lambda *_args, **_kwargs: response())
    result = benchmark.run_batch(
        "http://native/v1/chat/completions", {}, parallel=4, timeout=1, run=1
    )
    assert result["parallel"] == 4
    assert result["completion_tokens"] == 12
    assert result["aggregate_pipeline_tokens_per_second"] > 0
    assert result["token_sha256"] == prompt_digest([1, 2, 3])


def test_parallel_native_batch_rejects_divergent_tokens(monkeypatch):
    calls = iter((response(), response(), response((1, 2, 4))))
    monkeypatch.setattr(benchmark, "post", lambda *_args, **_kwargs: next(calls))
    try:
        benchmark.run_batch(
            "http://native/v1/chat/completions", {}, parallel=3, timeout=1, run=1
        )
    except RuntimeError as error:
        assert "different tokens" in str(error)
    else:
        raise AssertionError("divergent native tokens were accepted")
