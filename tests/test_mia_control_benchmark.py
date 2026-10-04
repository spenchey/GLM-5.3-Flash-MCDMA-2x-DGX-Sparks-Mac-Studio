from pathlib import Path
import importlib.util

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark-mia-control.py"
SPEC = importlib.util.spec_from_file_location("benchmark_mia_control", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _line(value):
    return ("data: " + __import__("json").dumps(value) + "\n\n").encode()


def test_parse_sse_uses_usage_and_first_to_last_content_window():
    times = iter([10.2, 10.5, 11.0])
    result = MODULE.parse_sse(
        [
            _line({"choices": [{"delta": {"role": "assistant"}}]}),
            _line({"choices": [{"delta": {"content": "one"}}]}),
            _line({"choices": [{"delta": {"content": " two"}, "finish_reason": "length"}]}),
            _line({"usage": {"prompt_tokens": 8, "completion_tokens": 4}, "choices": []}),
            b"data: [DONE]\n\n",
        ],
        10.0,
        clock=lambda: next(times),
    )
    assert result["ttft_seconds"] == pytest.approx(0.2)
    assert result["decode_seconds"] == pytest.approx(0.3)
    assert result["decode_tokens_per_second"] == pytest.approx(10.0)
    assert result["total_seconds"] == pytest.approx(1.0)
    assert result["completion_tokens"] == 4
    assert result["prompt_tokens"] == 8
    assert result["finish_reason"] == "length"
    assert result["text"] == "one two"


def test_request_body_matches_sparkdash_c1_protocol():
    body = MODULE.request_body("model", "prompt", 400, force_length=True)
    assert body["temperature"] == 0
    assert body["top_p"] == 1
    assert body["max_tokens"] == 400
    assert body["min_tokens"] == 400
    assert body["ignore_eos"] is True
    assert body["stop"] == []
    assert body["stream_options"] == {"include_usage": True}
    assert body["chat_template_kwargs"]["enable_thinking"] is False


def test_protocol_prompts_are_exact_current_sparkdash_c1_prompts():
    assert MODULE.PROMPTS["structured"] == (
        "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
    )
    assert MODULE.PROMPTS["prose"] == (
        "Write a detailed step-by-step explanation of how a hash map works, "
        "including collision handling, resizing, and time complexity. Be thorough."
    )
    assert MODULE.PROMPTS["code"].startswith("binary_search\n")
