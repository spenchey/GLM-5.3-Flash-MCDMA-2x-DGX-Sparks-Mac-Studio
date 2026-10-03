import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load_script("benchmark-three-machine.py")
quality = load_script("quality-three-machine.py")
capture = load_script("capture-native-references.py")
native_benchmark = load_script("benchmark-native-two-spark.py")


def response(*, zero_metal: bool = False, completion_tokens: int = 3):
    metal = 0.0 if zero_metal else 0.01
    steps = [
        {"prompt": True, "rows": 63, "accepted_tokens": 0,
         "metal_s": metal, "total_s": 0.2},
        {"prompt": True, "rows": 2, "accepted_tokens": 1,
         "metal_s": metal, "total_s": 0.2},
        {"prompt": False, "rows": 3, "accepted_tokens": 1,
         "verified_drafts": 2, "accepted_drafts": 0,
         "metal_s": metal, "total_s": 0.03},
        {"prompt": False, "rows": 3, "accepted_tokens": 1,
         "verified_drafts": 2, "accepted_drafts": 0,
         "metal_s": metal, "total_s": 0.04},
    ]
    return {
        "choices": [{"finish_reason": "length", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 65, "completion_tokens": completion_tokens},
        "three_machine_metrics": {
            "generated_token_ids": [1, 2, 3],
            "steps": steps,
            "decode_rounds": 2,
            "verified_drafts": 4,
            "accepted_drafts": 0,
            "metal_s": metal * len(steps),
            "elapsed_s": 0.4,
            "request_to_first_token_s": 0.12,
            "request_to_completion_s": 0.495,
        },
    }


def test_benchmark_percentile_is_nearest_rank():
    assert benchmark.percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    assert benchmark.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0


def test_benchmark_derives_steps_and_transport_calls_from_usage():
    result = benchmark.normalize(response(), 0.5, 1, 3, True)
    assert result["expected_mcdma_calls"] == 10
    assert result["completion_tokens"] == 3
    assert result["decode_step_s"]["median"] == pytest.approx(0.035)
    assert result["decode_tokens_per_s"] == pytest.approx(2 / 0.07)


def test_benchmark_wall_tolerance_has_fifty_millisecond_floor():
    result = benchmark.normalize(response(), 0.54, 1, 3, True)
    assert result["client_server_wall_delta_s"] == pytest.approx(0.045)
    with pytest.raises(RuntimeError, match="50 ms or 2%"):
        benchmark.normalize(response(), 0.56, 1, 3, True)


def test_benchmark_rejects_unmeasured_request_stall():
    stalled = response()
    stalled["three_machine_metrics"]["request_to_completion_s"] = 0.9
    with pytest.raises(RuntimeError, match="not accounted for"):
        benchmark.normalize(stalled, 0.9, 1, 3, True)


def test_benchmark_fails_closed_on_missing_metal_or_token_evidence():
    with pytest.raises(RuntimeError, match="Mac Metal participation"):
        benchmark.normalize(response(zero_metal=True), 0.5, 1, 3, True)
    malformed = response()
    malformed["three_machine_metrics"]["generated_token_ids"] = [1, 2]
    with pytest.raises(RuntimeError, match="token IDs"):
        benchmark.normalize(malformed, 0.5, 1, 3, True)


def test_benchmark_requires_prompt_token_parity():
    with pytest.raises(RuntimeError, match="prompt-token count"):
        benchmark.normalize(response(), 0.5, 1, 3, True, 64)


def test_consumers_require_the_pinned_reference_hash(tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text("{}\n")
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    assert benchmark.require_reference_hash(reference, digest) == digest
    assert quality.require_reference_hash(reference, digest) == digest
    for consumer in (benchmark, quality):
        with pytest.raises(SystemExit, match="pinned value"):
            consumer.require_reference_hash(reference, "0" * 64)
        with pytest.raises(SystemExit, match="must be a full"):
            consumer.require_reference_hash(reference, None)


def test_quality_requires_prompt_token_parity():
    assert quality.require_prompt_tokens("E1", 17, 17) == 17
    with pytest.raises(RuntimeError, match="prompt-token count"):
        quality.require_prompt_tokens("E1", 16, 17)


def test_capture_refuses_overwrite_without_force(tmp_path):
    output = tmp_path / "references.json"
    output.write_text("keep")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        capture.ensure_output_path(output, False)
    capture.ensure_output_path(output, True)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        capture.write_output(output, {"new": True}, False)
    assert output.read_text() == "keep"


def test_capture_requires_attested_native_source_identity():
    image_id = "sha256:" + "a" * 64
    health = {
        "status": "ok",
        "model": "model",
    }
    container = {
        "name": "glm53", "container_id": "b" * 64, "image_id": image_id,
        "running": True, "oom_killed": False, "exit_code": 0,
    }
    attestation = {
        "schema": 1, "kind": "native-two-spark", "model": "model",
        "image_id": image_id, "drafter": "mtp", "recipe_commit": "c" * 40,
        "recipe_tree": "d" * 40, "recipe_config_sha256": "e" * 64,
        "head": container, "worker": {**container, "container_id": "f" * 64},
    }
    assert capture.health_source(health, "model", attestation, image_id) == {
        "kind": "native-two-spark",
        "model": "model",
        "image_id": image_id,
        "drafter": "mtp",
        "recipe_commit": "c" * 40,
        "recipe_tree": "d" * 40,
        "recipe_config_sha256": "e" * 64,
        "containers": {"head": container, "worker": {**container, "container_id": "f" * 64}},
    }
    broken = dict(attestation)
    broken["image_id"] = "sha256:" + "0" * 64
    with pytest.raises(RuntimeError, match="pinned image"):
        capture.health_source(health, "model", broken, image_id)


def test_capture_accepts_current_tensorfold_health_with_container_attestation():
    image_id = "sha256:" + "a" * 64
    container = {
        "name": "glm53", "container_id": "b" * 64, "image_id": image_id,
        "running": True, "oom_killed": False, "exit_code": 0,
    }
    attestation = {
        "schema": 1, "kind": "native-two-spark", "model": "model",
        "image_id": image_id, "drafter": "mtp", "recipe_commit": "c" * 64,
        "recipe_tree": "d" * 64, "recipe_config_sha256": "e" * 64,
        "head": container, "worker": {**container, "container_id": "f" * 64},
    }
    source = capture.health_source(
        {"ok": True, "backend": "tensorfold", "busy": False},
        "model", attestation, image_id,
    )
    assert source["model"] == "model"
    assert native_benchmark.health_ready({"ok": True, "backend": "tensorfold"})


def test_native_capture_proves_mtp_off_and_prompt_count(monkeypatch):
    prompt = "hello"
    base = {
        "model": "model",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}, "token_ids": [1]}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 1},
        "tensorfold": {"drafts": False},
    }
    monkeypatch.setattr(capture, "post", lambda *_args, **_kwargs: base)
    assert capture.capture("http://native", "model", prompt, 1, 1)["prompt_tokens"] == 7
    base["tensorfold"] = {"drafts": True}
    with pytest.raises(RuntimeError, match="MTP drafts were disabled"):
        capture.capture("http://native", "model", prompt, 1, 1)


def test_quality_extracts_fenced_python_and_checks_symbols():
    text = "```python\ndef required():\n    return 1\n```"
    assert quality.code_body(text).startswith("def required")
    quality.validate_code(text, ["required"])


def test_quality_fails_closed_on_bad_code_or_missing_symbol():
    with pytest.raises(ValueError, match="unclosed code fence"):
        quality.code_body("```python\ndef broken():\n")
    with pytest.raises(SyntaxError):
        quality.validate_code("def broken(:", ["broken"])
    with pytest.raises(ValueError, match="missing required symbol"):
        quality.validate_code("def other():\n    pass", ["required"])


@pytest.mark.parametrize(
    ("validator", "expected", "accepted", "rejected"),
    [
        ("python_symbols", ["required"], "def required():\n    pass", "def other():\n    pass"),
        ("exact_json", {"items": [1, 2]}, '```json\n{"items":[1,2]}\n```', '{"items":[2,1]}'),
        ("integer", 42, "42", "The answer is 42"),
        ("contains_all", ["alpha", "omega"], "alpha then omega", "alpha only"),
        ("min_words", 4, "one two three four", "one two three"),
    ],
)
def test_quality_per_entry_validators(validator, expected, accepted, rejected):
    task = {"validator": validator, "expected": expected}
    assert quality.validation_spec(task) == (validator, expected)
    quality.validate_response(accepted, validator, expected)
    with pytest.raises(ValueError):
        quality.validate_response(rejected, validator, expected)


def test_quality_legacy_expected_symbols_normalizes_to_python_validator():
    assert quality.validation_spec({"expected_symbols": ["required"]}) == (
        "python_symbols", ["required"]
    )
