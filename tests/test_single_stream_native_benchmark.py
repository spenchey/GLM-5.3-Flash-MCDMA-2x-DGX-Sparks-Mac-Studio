from pathlib import Path
import importlib.util

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark-single-stream-native.py"
SPEC = importlib.util.spec_from_file_location("benchmark_single_stream_native", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
normalize = MODULE.normalize


def test_normalize_preserves_client_and_server_boundaries():
    result = normalize({
        "seconds": 5.0,
        "first_token_seconds": 0.5,
        "tokens": 256,
        "token_ids": list(range(256)),
        "token_sha256": "a" * 64,
        "text_sha256": "b" * 64,
        "finish_reason": "length",
        "client_chunk_seconds": [0.5, 0.6, 0.8],
        "runtime": {"prefill_s": 0.2, "decode_s": 4.0},
    })
    assert result["total_seconds"] == 5.0
    assert result["ttft_seconds"] == 0.5
    assert result["decode_tokens_per_second"] == pytest.approx(63.75)
    assert result["client_chunk_gaps_seconds"] == pytest.approx([0.1, 0.2])


def test_normalize_can_derive_decode_time_from_server_rate():
    result = normalize({
        "seconds": 5.0,
        "first_token_seconds": 0.5,
        "tokens": 11,
        "token_ids": list(range(11)),
        "token_sha256": "a" * 64,
        "text_sha256": "b" * 64,
        "finish_reason": "length",
        "client_chunk_seconds": [0.5],
        "runtime": {"prefill_seconds": 0.2, "tokens_per_second": 5.0},
    })
    assert result["decode_seconds"] == pytest.approx(2.0)
    assert result["decode_tokens_per_second"] == pytest.approx(5.0)
