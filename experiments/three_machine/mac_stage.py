"""TensorFold Metal embedding/layer-0 client for the three-machine GLM split."""

from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path
from typing import Any

from experiments.three_machine import protocol
from experiments.three_machine.mailbox import ClientMailbox

DEFAULT_CONTEXT_CAPACITY = 2051


def dense_context_capacity(index_topk: int, index_kpool: int) -> int:
    """Return the pinned CUDA engine's dense-only context capacity."""

    capacity = int(index_topk) + int(index_kpool) - 1
    if capacity <= 0:
        raise ValueError("model configuration has no usable dense context")
    return capacity


def validate_token_budget(
        prompt_tokens: list[int], max_tokens: int,
        capacity: int = DEFAULT_CONTEXT_CAPACITY) -> None:
    """Reject a request before any model state changes when it cannot fit."""

    if not prompt_tokens:
        raise ValueError("prompt encoded to zero tokens")
    # The first generated token is selected by the final prompt activation;
    # each additional generated token consumes one decode position.
    required = len(prompt_tokens) + max_tokens - 1
    if required > capacity:
        raise ValueError(
            f"prompt plus requested output needs {required} positions; this deployment has {capacity}"
        )


class MetalStage:
    """The real GLM embedding and original layer 0, with transactional cache commit."""

    def __init__(self, model_dir: Path) -> None:
        import mlx.core as mx

        from tensorfold.families.glm5_next.config import Config
        from tensorfold.families.glm5_next.linear import Dense, Q
        from tensorfold.families.glm5_next.model import GLM5
        from tensorfold.families.glm5_next.weights import Weights, load_layer, set_activation

        raw = json.loads((model_dir / "config.json").read_text())
        set_activation(raw)
        self.cfg = Config.from_dict(raw)
        if int(self.cfg.num_hidden_layers) != 45 or int(self.cfg.hc_mult) != 4:
            raise RuntimeError("stage checkpoint is not the pinned 45-layer, four-stream GLM-5.3 model")
        self.context_capacity = dense_context_capacity(self.cfg.index_topk, self.cfg.index_kpool)
        weights = Weights(model_dir, mtp_layer=int(self.cfg.num_hidden_layers))
        self.embed = weights.linear("embed_tokens")
        if not isinstance(self.embed, (Dense, Q)):
            raise RuntimeError("unsupported embedding representation")
        self.layer = load_layer(weights, 0, self.cfg)
        # GLM5 supplies the exact boundary/hyper-connection implementation.  Its
        # final norm and head are deliberately unused by this stage.
        dummy_norm = mx.ones((int(self.cfg.hidden_size),), dtype=mx.bfloat16)
        self.model = GLM5(self.cfg, self.embed, [self.layer], dummy_norm, self.embed)
        self.reset()

    def reset(self) -> None:
        self.cache = self.model.make_cache()
        self.pending: tuple[int, bool] | None = None

    def _embedding(self, ids: Any) -> Any:
        from tensorfold.families.glm5_next import config as C
        from tensorfold.families.glm5_next.linear import Dense

        if isinstance(self.embed, Dense):
            return self.embed.weight[ids].astype(C.act())
        return self.model.embed_tokens(ids)

    def forward(self, tokens: list[int]) -> bytes:
        import mlx.core as mx
        import numpy as np

        from tensorfold.families.glm5_next import config as C

        if self.pending is not None:
            raise protocol.ProtocolError("Metal already has an uncommitted activation")
        ids = mx.array(tokens).reshape(-1).astype(mx.uint32)
        rows = int(ids.shape[0])
        if not 1 <= rows <= protocol.MAX_ROWS:
            raise protocol.ProtocolError("Metal windows must contain 1..64 tokens")
        decode = rows <= int(C.DECODE_ROWS)
        h = self._embedding(ids)
        x = mx.contiguous(mx.broadcast_to(h[:, None, :], (rows, int(self.cfg.hc_mult), int(h.shape[-1]))))
        layer = self.layer
        x, normed, post, comb = self.model.boundary(x, None, layer.attn_hc, layer.in_norm, decode)
        pending = (layer.attn(normed, [self.cache[0]], (rows,), decode), post, comb)
        x, normed, post, comb = self.model.boundary(x, pending, layer.ffn_hc, layer.post_norm, decode)
        pending = (layer.mlp(normed, decode), post, comb)
        x = mx.contiguous(self.model.boundary(x, pending, None, None, decode)[0].reshape(rows, -1).astype(mx.bfloat16))
        mx.eval(x)
        self.pending = (rows, decode)
        return np.array(x.view(mx.uint16)).tobytes(order="C")

    def commit(self, keep: int | None = None) -> None:
        if self.pending is None:
            raise protocol.ProtocolError("Metal has no pending activation")
        rows, _decode = self.pending
        keep = rows if keep is None else int(keep)
        if not 1 <= keep <= rows:
            raise protocol.ProtocolError("Metal keep must be within the pending activation rows")
        # Forward writes every speculative row eagerly. Roll rejected rows back
        # before the next window; keeping the whole window needs no replay.
        if keep < rows:
            self.model.keep_rows(self.cache, rows, keep)
        self.pending = None


def _tokenizer(model_dir: Path):
    from tensorfold.families.glm5_next.prompts import GlmTokenizer
    from tensorfold.families.tokenizer import load_tokenizer

    raw = json.loads((model_dir / "config.json").read_text())
    text = raw.get("text_config") or raw
    eos = text.get("eos_token_id")
    return GlmTokenizer(load_tokenizer(model_dir, eos_token_ids=eos))


def _expect_ack(raw: bytes, expected: protocol.Frame) -> None:
    reply = protocol.unpack(raw, now_ns=time.time_ns())
    if reply.kind is protocol.Kind.ERROR:
        raise RuntimeError(reply.payload.decode("utf-8", errors="replace"))
    if (reply.kind, reply.request_id, reply.step, reply.committed, reply.generation) != (
            protocol.Kind.ACK, expected.request_id, expected.step, expected.committed, expected.generation):
        raise protocol.ProtocolError("server acknowledgement identity differs from the client commit")


def _window_reply(raw: bytes, *, frame: protocol.Frame) -> tuple[list[int], list[int], int]:
    reply = protocol.unpack(raw, now_ns=time.time_ns())
    if reply.kind is protocol.Kind.ERROR:
        raise RuntimeError(reply.payload.decode("utf-8", errors="replace"))
    return protocol.checked_window_result(
        reply,
        request_id=frame.request_id,
        step=frame.step,
        generation=frame.generation,
        base_committed=frame.committed,
        input_rows=frame.rows,
        prompt=bool(frame.flags & protocol.Flags.PROMPT),
    )


def run(model_dir: Path, prompt: str, max_tokens: int, mailbox_name: str, timeout: float,
        shutdown: bool, *, stage: MetalStage | None = None, tokenizer=None,
        prompt_tokens: list[int] | None = None) -> dict:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    stage = stage or MetalStage(model_dir)
    tokenizer = tokenizer or _tokenizer(model_dir)
    tokenize_started = time.perf_counter()
    if prompt_tokens is None:
        prompt_tokens = list(tokenizer.encode(prompt, add_special_tokens=False))
    tokenize_seconds = time.perf_counter() - tokenize_started
    validate_token_budget(prompt_tokens, max_tokens, stage.context_capacity)
    request_id = secrets.randbits(63) or 1
    step = committed = 0
    generated: list[int] = []
    drafts: list[int] = []
    metal_seconds = 0.0
    activation_roundtrip_seconds = 0.0
    acknowledgement_seconds = 0.0
    mailbox_open_seconds = 0.0
    reset_seconds = 0.0
    steps: list[dict[str, float | int | bool]] = []
    decode_rounds = 0
    verified_drafts = 0
    accepted_drafts = 0
    first_token_seconds = 0.0
    started = time.perf_counter()
    began = time.perf_counter()
    mailbox_instance = ClientMailbox(mailbox_name, ready_timeout_s=timeout)
    mailbox_open_seconds = time.perf_counter() - began
    with mailbox_instance as mailbox:
        try:
            # A CLI invocation cannot retain its Metal cache after exit, so it
            # always starts from and leaves the two CUDA ranks at position 0.
            reset = protocol.control(kind=protocol.Kind.RESET, request_id=request_id, step=0, committed=0,
                                     generation=mailbox.generation, timeout_s=timeout)
            began = time.perf_counter()
            _expect_ack(mailbox.call(protocol.pack(reset), timeout), reset)
            reset_seconds += time.perf_counter() - began
            stage.reset()
            chunks = [prompt_tokens[i:i + protocol.WINDOW_MAX_ROWS]
                      for i in range(0, len(prompt_tokens), protocol.WINDOW_MAX_ROWS)]
            for chunk_index, chunk in enumerate(chunks):
                final_prompt = chunk_index + 1 == len(chunks)
                step_record: dict[str, float | int | bool] = {
                    "step": step, "prompt": True, "rows": len(chunk), "position_start": committed,
                }
                began = time.perf_counter()
                payload = stage.forward(chunk)
                step_record["metal_s"] = time.perf_counter() - began
                metal_seconds += float(step_record["metal_s"])
                frame = protocol.window_activation(
                    request_id=request_id, step=step, committed=committed,
                    generation=mailbox.generation, tokens=chunk, payload=payload,
                    prompt=True, final=final_prompt, timeout_s=timeout,
                )
                began = time.perf_counter()
                answer_raw = mailbox.call(protocol.pack(frame), timeout)
                step_record["spark_mcdma_roundtrip_s"] = time.perf_counter() - began
                activation_roundtrip_seconds += float(step_record["spark_mcdma_roundtrip_s"])
                accepted, next_drafts, keep = _window_reply(answer_raw, frame=frame)
                answer = protocol.unpack(answer_raw, now_ns=time.time_ns())
                if not final_prompt and next_drafts:
                    raise protocol.ProtocolError("an intermediate prompt window returned decode drafts")
                ack = protocol.control(kind=protocol.Kind.ACK, request_id=request_id, step=step,
                                       committed=answer.committed, generation=mailbox.generation,
                                       timeout_s=timeout)
                began = time.perf_counter()
                _expect_ack(mailbox.call(protocol.pack(ack), timeout), ack)
                step_record["ack_s"] = time.perf_counter() - began
                acknowledgement_seconds += float(step_record["ack_s"])
                stage.commit(keep)
                step_record["position_end"] = answer.committed
                step_record["accepted_tokens"] = len(accepted)
                step_record["next_drafts"] = len(next_drafts)
                step_record["total_s"] = (
                    float(step_record["metal_s"])
                    + float(step_record["spark_mcdma_roundtrip_s"])
                    + float(step_record["ack_s"])
                )
                steps.append(step_record)
                step += 1
                committed = answer.committed
                if final_prompt:
                    generated.extend(accepted)
                    drafts = next_drafts
            first_token_seconds = time.perf_counter() - started
            eos = getattr(tokenizer, "eos_token_ids", None) or getattr(tokenizer, "eos_token_id", None) or []
            eos = {int(x) for x in (eos if isinstance(eos, (list, tuple, set)) else [eos]) if x is not None}
            while len(generated) < max_tokens and generated[-1] not in eos:
                remaining = max_tokens - len(generated)
                window = [generated[-1], *drafts[:max(0, remaining - 1)]]
                step_record = {"step": step, "prompt": False, "rows": len(window),
                               "position_start": committed}
                began = time.perf_counter()
                payload = stage.forward(window)
                step_record["metal_s"] = time.perf_counter() - began
                metal_seconds += float(step_record["metal_s"])
                frame = protocol.window_activation(
                    request_id=request_id, step=step, committed=committed,
                    generation=mailbox.generation, tokens=window, payload=payload,
                    prompt=False, timeout_s=timeout,
                )
                began = time.perf_counter()
                answer_raw = mailbox.call(protocol.pack(frame), timeout)
                step_record["spark_mcdma_roundtrip_s"] = time.perf_counter() - began
                activation_roundtrip_seconds += float(step_record["spark_mcdma_roundtrip_s"])
                accepted, next_drafts, keep = _window_reply(answer_raw, frame=frame)
                answer = protocol.unpack(answer_raw, now_ns=time.time_ns())
                ack = protocol.control(kind=protocol.Kind.ACK, request_id=request_id, step=step,
                                       committed=answer.committed, generation=mailbox.generation,
                                       timeout_s=timeout)
                began = time.perf_counter()
                _expect_ack(mailbox.call(protocol.pack(ack), timeout), ack)
                step_record["ack_s"] = time.perf_counter() - began
                acknowledgement_seconds += float(step_record["ack_s"])
                stage.commit(keep)
                generated.extend(accepted)
                drafts = next_drafts
                decode_rounds += 1
                verified_drafts += len(window) - 1
                accepted_drafts += keep - 1
                step_record["position_end"] = answer.committed
                step_record["accepted_tokens"] = len(accepted)
                step_record["verified_drafts"] = len(window) - 1
                step_record["accepted_drafts"] = keep - 1
                step_record["next_drafts"] = len(next_drafts)
                step_record["total_s"] = (
                    float(step_record["metal_s"])
                    + float(step_record["spark_mcdma_roundtrip_s"])
                    + float(step_record["ack_s"])
                )
                steps.append(step_record)
                step += 1
                committed = answer.committed
            if shutdown:
                stop = protocol.control(kind=protocol.Kind.SHUTDOWN, request_id=request_id, step=step,
                                        committed=committed, generation=mailbox.generation, timeout_s=timeout)
                _expect_ack(mailbox.call(protocol.pack(stop), timeout), stop)
            else:
                reset = protocol.control(kind=protocol.Kind.RESET, request_id=request_id, step=step,
                                         committed=committed, generation=mailbox.generation, timeout_s=timeout)
                reset_ack = protocol.control(kind=protocol.Kind.ACK, request_id=request_id, step=0, committed=0,
                                             generation=mailbox.generation, timeout_s=timeout)
                began = time.perf_counter()
                _expect_ack(mailbox.call(protocol.pack(reset), timeout), reset_ack)
                reset_seconds += time.perf_counter() - began
                stage.reset()
        except Exception:
            stage.reset()
            try:
                reset = protocol.control(kind=protocol.Kind.RESET, request_id=request_id, step=step,
                                         committed=committed, generation=mailbox.generation, timeout_s=timeout)
                mailbox.call(protocol.pack(reset), timeout)
            except Exception:
                pass
            raise
    elapsed = time.perf_counter() - started
    return {"request_id": request_id, "generation": mailbox_instance.generation,
            "prompt_tokens": len(prompt_tokens), "generated_tokens": generated,
            "text": tokenizer.decode(generated, skip_special_tokens=True), "committed": committed,
            "elapsed_s": elapsed,
            "tokens_per_s": len(generated) / elapsed if elapsed else 0.0,
            "first_token_s": first_token_seconds, "tokenize_s": tokenize_seconds,
            "mailbox_open_s": mailbox_open_seconds, "reset_s": reset_seconds,
            "metal_s": metal_seconds, "spark_mcdma_roundtrip_s": activation_roundtrip_seconds,
            "ack_s": acknowledgement_seconds, "steps": steps,
            "decode_rounds": decode_rounds,
            "verified_drafts": verified_drafts,
            "accepted_drafts": accepted_drafts,
            "tokens_per_decode_round": (
                max(0, len(generated) - 1) / decode_rounds if decode_rounds else 0.0
            ),
            "draft_acceptance_rate": (
                accepted_drafts / verified_drafts if verified_drafts else 0.0
            ),
            "context_capacity": stage.context_capacity,
            "finish_reason": "stop" if generated[-1] in eos else "length"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--mailbox", default="tfglm53")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--shutdown", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.model, args.prompt, args.max_tokens, args.mailbox, args.timeout, args.shutdown),
                     ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
