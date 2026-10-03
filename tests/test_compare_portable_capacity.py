import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_portable_capacity", ROOT / "scripts" / "compare-portable-capacity.py"
)
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(comparison)


def evidence(native_rate=70.0, hybrid_rate=75.0):
    native = {
        "kind": "portable-native-two-spark-concurrency",
        "parallel": 4,
        "max_tokens": 128,
        "token_sha256": "a" * 64,
        "aggregate_pipeline_tokens_per_second": {"median": native_rate},
    }
    hybrid = {
        "event": "glm_mcdma_benchmark",
        "parallel": 4,
        "max_new_tokens": 128,
        "local_reference": {"token_sha256": "a" * 64},
        "warmups": [{"exact_token_match": True}],
        "runs": [{"exact_token_match": True}],
        "aggregate_pipeline_tokens_per_second": {"median": hybrid_rate},
    }
    return native, hybrid


def test_comparison_requires_and_reports_three_percent_gain():
    native, hybrid = evidence()
    result = comparison.compare(native, hybrid, 0.03)
    assert result["pass"] is True
    assert result["gain_fraction"] == pytest.approx(75 / 70 - 1)


def test_comparison_rejects_a_slower_hybrid():
    native, hybrid = evidence(hybrid_rate=71.0)
    with pytest.raises(RuntimeError, match='"pass": false'):
        comparison.compare(native, hybrid, 0.03)


def test_comparison_rejects_different_output_tokens():
    native, hybrid = evidence()
    hybrid["local_reference"]["token_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="output tokens differ"):
        comparison.compare(native, hybrid, 0.03)
