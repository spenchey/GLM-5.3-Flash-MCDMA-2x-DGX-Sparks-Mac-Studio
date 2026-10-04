import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark-mac-single-stream-ceiling.py"
SPEC = importlib.util.spec_from_file_location("benchmark_mac_single_stream_ceiling", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def run(tokens, *, total, ttft, decode, rate, text="same"):
    return {
        "tokens": tokens,
        "token_sha256": "a" * 64,
        "text": text,
        "total_seconds": total,
        "first_token_seconds": ttft,
        "generation_seconds": decode,
        "decode_tokens_per_second": rate,
    }


def test_summary_preserves_primary_and_decode_boundaries():
    result = module.summarize_runs([
        run([1, 2, 3], total=5.0, ttft=1.0, decode=4.0, rate=0.5),
        run([1, 2, 3], total=4.0, ttft=0.8, decode=3.2, rate=0.625),
        run([1, 2, 3], total=6.0, ttft=1.2, decode=4.8, rate=0.416),
    ], prompt_ids=[9, 8], max_new_tokens=3)
    assert result["prompt_tokens"] == 2
    assert result["completion_tokens"] == 3
    assert result["total_seconds"]["median"] == 5.0
    assert result["ttft_seconds"]["median"] == 1.0
    assert result["decode_tokens_per_second"]["max"] == 0.625


def test_summary_refuses_short_or_nondeterministic_rounds():
    with pytest.raises(RuntimeError, match="fixed output length"):
        module.summarize_runs([
            run([1, 2], total=1, ttft=.1, decode=.9, rate=1),
        ], prompt_ids=[9], max_new_tokens=3)
    with pytest.raises(RuntimeError, match="different output"):
        module.summarize_runs([
            run([1, 2, 3], total=1, ttft=.1, decode=.9, rate=1),
            run([1, 2, 3], total=1, ttft=.1, decode=.9, rate=1, text="different"),
        ], prompt_ids=[9], max_new_tokens=3)


def test_runtime_call_profiler_measures_and_restores_methods():
    class Runtime:
        def hidden(self, value):
            return value + 1

        def head(self, value):
            return value * 2

    runtime = Runtime()
    profiler = module.RuntimeCallProfiler(runtime)
    assert runtime.hidden(2) == 3
    assert runtime.head(3) == 6
    first = profiler.snapshot()
    assert first["hidden"]["calls"] == 1
    assert first["head"]["calls"] == 1
    profiler.reset()
    assert profiler.snapshot() == {}
    profiler.stop()
    assert runtime.hidden(4) == 5
    assert runtime.head(5) == 10


def test_layer_call_profiler_aggregates_and_restores_components():
    class Attention:
        def __call__(self, value):
            return value + 1

    class MLP:
        def __call__(self, value):
            return value * 2

    class Layer:
        def __init__(self):
            self.attn = Attention()
            self.mlp = MLP()

    class Model:
        def __init__(self):
            self.layers = [Layer(), Layer()]

        def embed_tokens(self, value):
            return value

        def boundary(self, value):
            return value

        def final_norm(self, value):
            return value

    class Runtime:
        def __init__(self):
            self.model = Model()

    runtime = Runtime()
    original_attention = runtime.model.layers[0].attn
    profiler = module.LayerCallProfiler(runtime)
    assert runtime.model.layers[0].attn(2) == 3
    assert runtime.model.layers[1].attn(3) == 4
    assert runtime.model.layers[0].mlp(3) == 6
    assert runtime.model.boundary(4) == 4
    result = profiler.snapshot()
    assert result["attn.Attention"]["calls"] == 2
    assert result["mlp.MLP"]["calls"] == 1
    assert result["boundary"]["calls"] == 1
    profiler.reset()
    assert profiler.snapshot() == {}
    profiler.stop()
    assert runtime.model.layers[0].attn is original_attention
