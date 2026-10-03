from http import HTTPStatus
from pathlib import Path
import time

import pytest

from experiments.three_machine import api_server
from experiments.three_machine.api_server import InferenceServer, MODEL_ID, Service, _validate_request_options


def valid_payload(**updates):
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "temperature": 0,
        "enable_thinking": False,
        "stream": False,
    }
    payload.update(updates)
    return payload


def test_request_options_accept_only_the_documented_surface():
    _validate_request_options(valid_payload())
    with pytest.raises(ValueError, match="stream must be false"):
        _validate_request_options(valid_payload(stream=True))
    with pytest.raises(ValueError, match="model must be"):
        _validate_request_options(valid_payload(model="another-model"))
    with pytest.raises(ValueError, match=r"unsupported request field\(s\): tools"):
        _validate_request_options(valid_payload(tools=[]))
    with pytest.raises(ValueError, match="enable_thinking must be a boolean"):
        _validate_request_options(valid_payload(enable_thinking="false"))


def test_orderly_stop_rejects_new_work_and_ends_the_server_loop():
    service = Service.__new__(Service)
    service.stopping = True
    with pytest.raises(RuntimeError, match="draining"):
        service.complete(valid_payload())

    server = InferenceServer.__new__(InferenceServer)
    server.stop_requested = True
    server._BaseServer__shutdown_request = False
    server.service_actions()
    assert server._BaseServer__shutdown_request is True


class FakeTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return "prompt"

    def encode(self, *_args, **_kwargs):
        return [1, 2]


def fake_service():
    service = Service.__new__(Service)
    service.model_dir = Path("/model")
    service.mailbox = "test"
    service.timeout = 1
    service.stage = type("Stage", (), {"context_capacity": 2051})()
    service.tokenizer = FakeTokenizer()
    service.requests = 0
    service.failures = 0
    service.failed = False
    service.stopping = False
    service.generation = 7
    service.observed_generation = 7
    return service


def test_api_uses_the_actual_dense_context_capacity(monkeypatch):
    monkeypatch.setattr(api_server, "link_generation", lambda _name: 7)
    service = fake_service()
    with pytest.raises(ValueError, match="between 1 and 2051"):
        service.complete(valid_payload(max_tokens=2052))
    with pytest.raises(ValueError, match="needs 2052 positions"):
        service.complete(valid_payload(max_tokens=2051))


def test_inference_failure_latches_health_unavailable(monkeypatch):
    service = fake_service()
    calls = []
    monkeypatch.setattr(api_server, "link_generation", lambda _name: 7)

    def fail(*_args, **_kwargs):
        calls.append(True)
        raise RuntimeError("rank failed")

    monkeypatch.setattr(api_server, "infer", fail)
    with pytest.raises(RuntimeError, match="rank failed"):
        service.complete(valid_payload())
    assert service.failed is True
    assert service.failures == 1
    status, health = service.health()
    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert health["status"] == "failed"
    assert health["failed_requests"] == 1
    with pytest.raises(RuntimeError, match="failed closed"):
        service.complete(valid_payload())
    assert len(calls) == 1


def test_link_generation_change_latches_health_unavailable(monkeypatch):
    service = fake_service()
    monkeypatch.setattr(api_server, "link_generation", lambda _name: 8)

    status, health = service.health()

    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert health["status"] == "failed"
    assert health["link_generation"] == 7
    assert health["observed_link_generation"] == 8
    assert health["failed_requests"] == 1


def test_failed_service_does_not_send_a_second_shutdown_command(monkeypatch):
    service = fake_service()
    calls = []
    monkeypatch.setattr(api_server, "control", lambda *_args: calls.append(True))

    service.failed = True
    service.shutdown_model()

    assert calls == []


def test_healthy_service_sends_orderly_shutdown_command(monkeypatch):
    service = fake_service()
    calls = []
    monkeypatch.setattr(api_server, "control", lambda *args: calls.append(args))

    service.shutdown_model()

    assert calls == [("test", "shutdown", 1)]


def test_api_measures_render_tokenize_and_request_to_first_token(monkeypatch):
    service = fake_service()
    monkeypatch.setattr(api_server, "link_generation", lambda _name: 7)

    def succeed(*_args, **_kwargs):
        return {
            "request_id": 11,
            "generation": 7,
            "prompt_tokens": 2,
            "generated_tokens": [42],
            "text": "answer",
            "finish_reason": "length",
            "elapsed_s": 0.2,
            "tokens_per_s": 5.0,
            "first_token_s": 0.1,
            "tokenize_s": 0.0,
            "mailbox_open_s": 0.0,
            "reset_s": 0.0,
            "metal_s": 0.01,
            "spark_mcdma_roundtrip_s": 0.08,
            "ack_s": 0.01,
            "decode_rounds": 0,
            "verified_drafts": 0,
            "accepted_drafts": 0,
            "tokens_per_decode_round": 0.0,
            "draft_acceptance_rate": 0.0,
            "committed": 2,
            "context_capacity": 2051,
            "steps": [{"metal_s": 0.01}],
        }

    monkeypatch.setattr(api_server, "infer", succeed)
    response = service.complete(
        valid_payload(max_tokens=1),
        request_received_s=time.perf_counter() - 0.5,
    )
    metrics = response["three_machine_metrics"]

    assert metrics["api_render_s"] >= 0
    assert metrics["api_tokenize_s"] >= 0
    assert metrics["request_to_first_token_s"] >= 0.5
    assert metrics["request_to_completion_s"] >= 0.5
    assert metrics["verified_drafts"] == 0
