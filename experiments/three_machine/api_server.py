"""Small persistent OpenAI-compatible endpoint for the three-machine GLM path."""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import sys
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from experiments.three_machine.control import run as control
from experiments.three_machine.mac_drafter import MacMTPStage, run as infer
from experiments.three_machine.mac_stage import _tokenizer, validate_token_budget
from experiments.three_machine.mailbox import link_generation


MODEL_ID = "GLM-5.3-Flash-TensorFold-MCDMA-3Machine"
SUPPORTED_REQUEST_FIELDS = {
    "model", "messages", "max_tokens", "temperature", "enable_thinking", "stream"
}


def _memory_metrics() -> dict[str, int | None]:
    """Return process peak and live MLX allocation evidence without mutating it."""

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak *= 1024
    result: dict[str, int | None] = {"process_peak_rss_bytes": peak}
    try:
        import mlx.core as mx

        result.update({
            "mlx_active_bytes": int(mx.get_active_memory()),
            "mlx_cache_bytes": int(mx.get_cache_memory()),
            "mlx_peak_bytes": int(mx.get_peak_memory()),
        })
    except (ImportError, AttributeError, RuntimeError):
        result.update({
            "mlx_active_bytes": None,
            "mlx_cache_bytes": None,
            "mlx_peak_bytes": None,
        })
    return result


def _validate_request_options(payload: dict[str, Any]) -> None:
    """Reject fields whose OpenAI semantics this small endpoint does not implement."""

    unsupported = sorted(set(payload) - SUPPORTED_REQUEST_FIELDS)
    if unsupported:
        raise ValueError(f"unsupported request field(s): {', '.join(unsupported)}")
    if payload.get("model", MODEL_ID) != MODEL_ID:
        raise ValueError(f"model must be {MODEL_ID}")
    if "stream" in payload and payload["stream"] is not False:
        raise ValueError("stream must be false; streaming is not supported")
    if "enable_thinking" in payload and type(payload["enable_thinking"]) is not bool:
        raise ValueError("enable_thinking must be a boolean")


def _prompt(tokenizer: Any, payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    if any(not isinstance(item, dict) or not isinstance(item.get("role"), str)
           or not isinstance(item.get("content"), str) for item in messages):
        raise ValueError("each message must contain string role and content fields")
    thinking = bool(payload.get("enable_thinking", False))
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=thinking)


class Service:
    def __init__(self, model_dir: Path, mailbox: str, timeout: float) -> None:
        self.model_dir = model_dir
        self.mailbox = mailbox
        self.timeout = timeout
        self.stage = MacMTPStage(
            model_dir,
            drafts=int(os.environ.get("THREE_MACHINE_MTP_DRAFTS", "3")),
            confidence=float(os.environ.get("THREE_MACHINE_MTP_CONFIDENCE", "0.35")),
        )
        self.tokenizer = _tokenizer(model_dir)
        self.requests = 0
        self.failures = 0
        self.failed = False
        self.stopping = False
        self.generation = link_generation(mailbox)
        self.observed_generation: int | None = self.generation

    def _link_ready(self) -> bool:
        if self.failed:
            return False
        try:
            self.observed_generation = link_generation(self.mailbox)
        except Exception:
            self.observed_generation = None
        if self.observed_generation != self.generation:
            self.failed = True
            self.failures += 1
            return False
        return True

    def complete(self, payload: dict[str, Any], *, request_received_s: float | None = None) -> dict[str, Any]:
        received_s = time.perf_counter() if request_received_s is None else request_received_s
        if self.stopping:
            raise RuntimeError("the service is draining for an orderly shutdown")
        _validate_request_options(payload)
        temperature = float(payload.get("temperature", 0) or 0)
        if temperature != 0:
            raise ValueError("this verified deployment currently supports greedy temperature=0 only")
        max_tokens = payload.get("max_tokens", 256)
        if type(max_tokens) is not int:
            raise ValueError("max_tokens must be a whole number")
        if not 1 <= max_tokens <= self.stage.context_capacity:
            raise ValueError(
                f"max_tokens must be between 1 and {self.stage.context_capacity}"
            )
        if self.failed:
            raise RuntimeError("the inference stream failed closed; run the verified recovery command")
        if not self._link_ready():
            raise RuntimeError("the MCDMA link generation changed; run the verified recovery command")
        began = time.perf_counter()
        prompt = _prompt(self.tokenizer, payload)
        api_render_s = time.perf_counter() - began
        began = time.perf_counter()
        prompt_tokens = list(self.tokenizer.encode(prompt, add_special_tokens=False))
        api_tokenize_s = time.perf_counter() - began
        validate_token_budget(prompt_tokens, max_tokens, self.stage.context_capacity)
        try:
            infer_started_s = time.perf_counter()
            result = infer(self.model_dir, prompt, max_tokens, self.mailbox,
                           self.timeout, False, stage=self.stage, tokenizer=self.tokenizer,
                           prompt_tokens=prompt_tokens)
            if result["generation"] != self.generation:
                raise RuntimeError("inference used a different MCDMA link generation")
        except Exception:
            self.failed = True
            self.failures += 1
            raise
        self.requests += 1
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result["text"]},
                         "finish_reason": result["finish_reason"]}],
            "usage": {"prompt_tokens": result["prompt_tokens"],
                      "completion_tokens": len(result["generated_tokens"]),
                      "total_tokens": result["prompt_tokens"] + len(result["generated_tokens"])},
            "three_machine_metrics": {key: result[key] for key in (
                "request_id", "elapsed_s", "tokens_per_s", "first_token_s", "tokenize_s", "mailbox_open_s",
                "reset_s", "metal_s", "spark_mcdma_roundtrip_s", "ack_s", "committed",
                "decode_rounds", "verified_drafts", "accepted_drafts", "tokens_per_decode_round",
                "draft_acceptance_rate", "context_capacity", "generation", "steps")},
        }
        response["three_machine_metrics"]["generated_token_ids"] = result["generated_tokens"]
        response["three_machine_metrics"].update({
            "api_render_s": api_render_s,
            "api_tokenize_s": api_tokenize_s,
            "request_to_first_token_s": (
                infer_started_s - received_s + float(result["first_token_s"])
            ),
            "request_to_completion_s": time.perf_counter() - received_s,
        })
        return response

    def health(self) -> tuple[int, dict[str, Any]]:
        if not self.failed and not self.stopping:
            self._link_ready()
        unavailable = self.failed or self.stopping
        status = HTTPStatus.SERVICE_UNAVAILABLE if unavailable else HTTPStatus.OK
        state = "failed" if self.failed else "draining" if self.stopping else "ready"
        return status, {
            "status": state,
            "model": MODEL_ID,
            "completed_requests": self.requests,
            "failed_requests": self.failures,
            "link_generation": self.generation,
            "observed_link_generation": self.observed_generation,
            "context_capacity": self.stage.context_capacity,
            "architecture": "complete-main-on-two-sparks-mtp-on-mac",
            "deployment_id": os.environ.get("THREE_MACHINE_DEPLOYMENT_ID", "unknown"),
            "commit": os.environ.get("THREE_MACHINE_COMMIT", "unknown"),
            "project_manifest": os.environ.get("THREE_MACHINE_PROJECT_MANIFEST", "unknown"),
            "config_sha256": os.environ.get("THREE_MACHINE_CONFIG_SHA256", "unknown"),
            "image_id": os.environ.get("THREE_MACHINE_RUNTIME_IMAGE_ID", "unknown"),
            "performance_knobs": {
                key: os.environ.get(key, "unset") for key in (
                    "TF_GLM_EXL3_LOADS", "TF_GLM_L2PF", "TF_GLM_KDA_CHUNKED",
                    "THREE_MACHINE_MTP_DRAFTS", "THREE_MACHINE_MTP_CONFIDENCE",
                    "THREE_MACHINE_PHASE_TIMING",
                )
            },
            "memory": _memory_metrics(),
        }

    def shutdown_model(self) -> None:
        if self.failed:
            return
        control(self.mailbox, "shutdown", self.timeout)


class Handler(BaseHTTPRequestHandler):
    server_version = "TensorFoldMCDMA/1"

    @property
    def service(self) -> Service:
        return self.server.service  # type: ignore[attr-defined]

    def _json(self, status: int, value: Any) -> None:
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            status, payload = self.service.health()
            self._json(status, payload)
        elif self.path == "/v1/models":
            self._json(HTTPStatus.OK, {"object": "list", "data": [{"id": MODEL_ID, "object": "model",
                                                                     "owned_by": "tensorfold-mcdma"}]})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        request_received_s = time.perf_counter()
        if self.path != "/v1/chat/completions":
            self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
            return
        if self.service.stopping:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE,
                       {"error": {"message": "the service is draining for an orderly shutdown"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 1_048_576:
                raise ValueError("request body must be between 1 byte and 1 MiB")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            self._json(
                HTTPStatus.OK,
                self.service.complete(payload, request_received_s=request_received_s),
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                       {"error": {"message": f"{type(exc).__name__}: {exc}"}})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[api] {self.address_string()} {format % args}", flush=True)


class InferenceServer(HTTPServer):
    """Finish an in-flight request before honoring an orderly stop signal."""

    stop_requested = False

    def service_actions(self) -> None:
        if self.stop_requested:
            # BaseServer checks this private flag at the top of its next loop.
            # Calling shutdown() from the signal-handling main thread deadlocks.
            self._BaseServer__shutdown_request = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mailbox", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    service = Service(args.model, args.mailbox, args.timeout)
    server = InferenceServer((args.host, args.port), Handler)
    server.service = service  # type: ignore[attr-defined]

    def stop(_signum: int, _frame: Any) -> None:
        service.stopping = True
        server.stop_requested = True

    signal.signal(signal.SIGTERM, stop)
    print(json.dumps({"event": "ready", "host": args.host, "port": args.port, "model": MODEL_ID}), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.shutdown_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
