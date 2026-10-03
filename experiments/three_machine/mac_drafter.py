"""Mac Studio MTP drafter for the exact two-Spark TensorFold main model."""

from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path
from typing import Any

from experiments.three_machine import protocol
from experiments.three_machine.mac_stage import _expect_ack, _tokenizer, validate_token_budget
from experiments.three_machine.mailbox import ClientMailbox


class MacMTPStage:
    """The checkpoint's real MTP prediction layer, embedding, and shared head."""

    def __init__(self, model_dir: Path, drafts: int = 3, confidence: float = 0.35) -> None:
        import mlx.core as mx

        from tensorfold.families.glm5_next import config as C
        from tensorfold.families.glm5_next.config import Config
        from tensorfold.families.glm5_next.linear import Dense, Q
        from tensorfold.families.glm5_next.model import GLM5
        from tensorfold.families.glm5_next.mtp import load as load_mtp
        from tensorfold.families.glm5_next.weights import Weights, _materialize, set_activation

        if not 1 <= drafts <= 8:
            raise ValueError("MTP draft count must be in 1..8")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("MTP confidence must be in 0..1")
        raw = json.loads((model_dir / "config.json").read_text())
        set_activation(raw)
        self.cfg = Config.from_dict(raw)
        if int(self.cfg.num_hidden_layers) != 45 or int(self.cfg.hc_mult) != 4:
            raise RuntimeError("MTP checkpoint is not the pinned 45-layer, four-stream GLM-5.3 model")
        self.context_capacity = int(self.cfg.index_topk) + int(self.cfg.index_kpool) - 1
        self.draft_count = int(drafts)
        self.confidence = float(confidence)
        weights = Weights(model_dir, mtp_layer=int(self.cfg.num_hidden_layers))
        self.embed = weights.linear("embed_tokens")
        self.lm_head = weights.linear("lm_head")
        if not isinstance(self.embed, (Dense, Q)) or not isinstance(self.lm_head, (Dense, Q)):
            raise RuntimeError("unsupported MTP embedding or shared-head representation")
        dummy_norm = mx.ones((int(self.cfg.hidden_size),), dtype=mx.bfloat16)
        self.model = GLM5(self.cfg, self.embed, [], dummy_norm, self.lm_head)
        self.model.weights = weights
        if isinstance(self.embed, Dense):
            # The EXL3 checkpoint keeps these shared matrices as dense BF16.
            self.model.embed_tokens = lambda ids: self.embed.weight[ids.reshape(-1)].astype(C.act())
        _materialize(self.embed, self.lm_head)
        self.mtp = load_mtp(self.model)
        self.reset()

    def reset(self) -> None:
        self.cache = self.mtp.make_cache()
        self.drafted = 0
        self.prompt_tail = None

    @staticmethod
    def _hidden(raw: bytes, rows: int):
        import mlx.core as mx
        import numpy as np

        if len(raw) != rows * protocol.ROW_BYTES:
            raise protocol.ProtocolError("hidden byte count differs from its declared rows")
        words = np.frombuffer(raw, dtype=np.uint16).reshape(rows, protocol.WIDTH)
        value = mx.array(words).view(mx.bfloat16)
        mx.eval(value)
        return value

    def _trim_drafts(self) -> None:
        if self.drafted:
            self.cache.trim(self.drafted)
            self.drafted = 0

    def _step(self, hidden: Any, tokens: list[int]):
        import mlx.core as mx

        if len(tokens) != int(hidden.shape[0]) or not tokens:
            raise protocol.ProtocolError("MTP hidden rows and following tokens must be nonempty and equal")
        ids = mx.array(tokens, dtype=mx.uint32)
        output = self.mtp(
            self.model,
            hidden,
            ids,
            [self.cache],
            (len(tokens),),
            len(tokens) <= 16,
        )
        logits = self.mtp.logits(self.model, output[-1:])
        mx.eval(output, logits)
        return output, logits

    @staticmethod
    def _draw(logits: Any, need_probability: bool) -> tuple[int, float]:
        import mlx.core as mx

        values = logits[0].astype(mx.float32)
        token_array = mx.argmax(values)
        mx.eval(token_array)
        token = int(token_array.item())
        if need_probability:
            top = mx.max(values)
            probability = mx.exp(values[token] - top) / mx.sum(mx.exp(values - top))
            mx.eval(probability)
            return token, float(probability.item())
        return token, 1.0

    def absorb_and_draft(self, hidden: Any, next_tokens: list[int], count: int) -> list[int]:
        """Absorb verified pairs, trim rejected prior drafts, then propose a new chain."""

        if not 0 <= count <= self.draft_count:
            raise ValueError("requested draft count exceeds the configured Mac MTP depth")
        self._trim_drafts()
        output, logits = self._step(hidden, next_tokens)
        if count == 0:
            return []
        drafts: list[int] = []
        chain = 1.0
        state = output[-1:]
        for index in range(count):
            token, probability = self._draw(logits, self.confidence > 0)
            if self.confidence > 0 and index > 0 and chain * probability < self.confidence:
                break
            drafts.append(token)
            if self.confidence > 0:
                chain *= probability
                if chain < self.confidence:
                    break
            if index + 1 < count:
                state, logits = self._step(state, [token])
                state = state[-1:]
                self.drafted += 1
        return drafts

    def absorb_prompt(self, tokens: list[int], hidden_raw: bytes, accepted: list[int], count: int) -> list[int]:
        import mlx.core as mx

        hidden = self._hidden(hidden_raw, len(tokens))
        pair_hidden: list[Any] = []
        pair_tokens: list[int] = []
        if self.prompt_tail is not None:
            pair_hidden.append(self.prompt_tail)
            pair_tokens.append(tokens[0])
        if len(tokens) > 1:
            pair_hidden.append(hidden[:-1])
            pair_tokens.extend(tokens[1:])
        if pair_tokens:
            self._trim_drafts()
            self._step(mx.concatenate(pair_hidden), pair_tokens)
        self.prompt_tail = hidden[-1:]
        if accepted:
            if len(accepted) != 1:
                raise protocol.ProtocolError("final prompt must contain one exact Spark sample")
            return self.absorb_and_draft(self.prompt_tail, accepted, count)
        if count:
            raise protocol.ProtocolError("only the final prompt may request MTP drafts")
        return []

    def absorb_decode(self, hidden_raw: bytes, accepted: list[int], count: int) -> list[int]:
        if not accepted:
            raise protocol.ProtocolError("decode must accept at least one exact Spark token")
        hidden = self._hidden(hidden_raw, len(accepted))
        return self.absorb_and_draft(hidden, accepted, count)


def _hidden_reply(raw: bytes, *, frame: protocol.Frame) -> tuple[list[int], bytes, int]:
    reply = protocol.unpack(raw, now_ns=time.time_ns())
    if reply.kind is protocol.Kind.ERROR:
        raise RuntimeError(reply.payload.decode("utf-8", errors="replace"))
    return protocol.checked_verified_hidden(
        reply,
        request_id=frame.request_id,
        step=frame.step,
        generation=frame.generation,
        base_committed=frame.committed,
        input_rows=frame.rows,
        prompt=bool(frame.flags & protocol.Flags.PROMPT),
        final=bool(frame.flags & protocol.Flags.FINAL),
    )


def run(model_dir: Path, prompt: str, max_tokens: int, mailbox_name: str, timeout: float,
        shutdown: bool, *, stage: MacMTPStage | None = None, tokenizer=None,
        prompt_tokens: list[int] | None = None, drafts: int = 3,
        confidence: float = 0.35) -> dict:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    stage = stage or MacMTPStage(model_dir, drafts=drafts, confidence=confidence)
    tokenizer = tokenizer or _tokenizer(model_dir)
    tokenize_started = time.perf_counter()
    if prompt_tokens is None:
        prompt_tokens = list(tokenizer.encode(prompt, add_special_tokens=False))
    tokenize_seconds = time.perf_counter() - tokenize_started
    validate_token_budget(prompt_tokens, max_tokens, stage.context_capacity)
    request_id = secrets.randbits(63) or 1
    step = committed = 0
    generated: list[int] = []
    proposals: list[int] = []
    metal_seconds = 0.0
    roundtrip_seconds = 0.0
    acknowledgement_seconds = 0.0
    reset_seconds = 0.0
    decode_rounds = verified_drafts = accepted_drafts = 0
    steps: list[dict[str, float | int | bool]] = []
    started = time.perf_counter()
    began = time.perf_counter()
    mailbox_instance = ClientMailbox(mailbox_name, ready_timeout_s=timeout)
    mailbox_open_seconds = time.perf_counter() - began
    with mailbox_instance as mailbox:
        try:
            reset = protocol.control(
                kind=protocol.Kind.RESET,
                request_id=request_id,
                step=0,
                committed=0,
                generation=mailbox.generation,
                timeout_s=timeout,
            )
            began = time.perf_counter()
            _expect_ack(mailbox.call(protocol.pack(reset), timeout), reset)
            reset_seconds += time.perf_counter() - began
            stage.reset()
            chunks = [prompt_tokens[index:index + protocol.WINDOW_MAX_ROWS]
                      for index in range(0, len(prompt_tokens), protocol.WINDOW_MAX_ROWS)]
            for chunk_index, chunk in enumerate(chunks):
                final_prompt = chunk_index + 1 == len(chunks)
                record: dict[str, float | int | bool] = {
                    "step": step,
                    "prompt": True,
                    "rows": len(chunk),
                    "position_start": committed,
                }
                frame = protocol.token_window(
                    request_id=request_id,
                    step=step,
                    committed=committed,
                    generation=mailbox.generation,
                    tokens=chunk,
                    prompt=True,
                    final=final_prompt,
                    timeout_s=timeout,
                )
                began = time.perf_counter()
                answer_raw = mailbox.call(protocol.pack(frame), timeout)
                record["spark_mcdma_roundtrip_s"] = time.perf_counter() - began
                roundtrip_seconds += float(record["spark_mcdma_roundtrip_s"])
                accepted, hidden, keep = _hidden_reply(answer_raw, frame=frame)
                answer = protocol.unpack(answer_raw, now_ns=time.time_ns())
                began = time.perf_counter()
                want = min(stage.draft_count, max(0, max_tokens - len(accepted))) if final_prompt else 0
                proposals = stage.absorb_prompt(chunk, hidden, accepted, want)
                record["metal_s"] = time.perf_counter() - began
                metal_seconds += float(record["metal_s"])
                ack = protocol.control(
                    kind=protocol.Kind.ACK,
                    request_id=request_id,
                    step=step,
                    committed=answer.committed,
                    generation=mailbox.generation,
                    timeout_s=timeout,
                )
                began = time.perf_counter()
                _expect_ack(mailbox.call(protocol.pack(ack), timeout), ack)
                record["ack_s"] = time.perf_counter() - began
                acknowledgement_seconds += float(record["ack_s"])
                record.update({
                    "position_end": answer.committed,
                    "accepted_tokens": len(accepted),
                    "next_drafts": len(proposals),
                    "total_s": float(record["spark_mcdma_roundtrip_s"])
                    + float(record["metal_s"]) + float(record["ack_s"]),
                })
                steps.append(record)
                committed = answer.committed
                step += 1
                if final_prompt:
                    generated.extend(accepted)
            first_token_seconds = time.perf_counter() - started
            eos_value = getattr(tokenizer, "eos_token_ids", None) or getattr(tokenizer, "eos_token_id", None) or []
            eos = {int(value) for value in (
                eos_value if isinstance(eos_value, (list, tuple, set)) else [eos_value]
            ) if value is not None}
            while len(generated) < max_tokens and generated[-1] not in eos:
                remaining = max_tokens - len(generated)
                window = [generated[-1], *proposals[:max(0, remaining - 1)]]
                record = {"step": step, "prompt": False, "rows": len(window), "position_start": committed}
                frame = protocol.token_window(
                    request_id=request_id,
                    step=step,
                    committed=committed,
                    generation=mailbox.generation,
                    tokens=window,
                    prompt=False,
                    timeout_s=timeout,
                )
                began = time.perf_counter()
                answer_raw = mailbox.call(protocol.pack(frame), timeout)
                record["spark_mcdma_roundtrip_s"] = time.perf_counter() - began
                roundtrip_seconds += float(record["spark_mcdma_roundtrip_s"])
                accepted, hidden, keep = _hidden_reply(answer_raw, frame=frame)
                answer = protocol.unpack(answer_raw, now_ns=time.time_ns())
                began = time.perf_counter()
                future = max(0, max_tokens - len(generated) - len(accepted))
                proposals = stage.absorb_decode(hidden, accepted, min(stage.draft_count, future))
                record["metal_s"] = time.perf_counter() - began
                metal_seconds += float(record["metal_s"])
                ack = protocol.control(
                    kind=protocol.Kind.ACK,
                    request_id=request_id,
                    step=step,
                    committed=answer.committed,
                    generation=mailbox.generation,
                    timeout_s=timeout,
                )
                began = time.perf_counter()
                _expect_ack(mailbox.call(protocol.pack(ack), timeout), ack)
                record["ack_s"] = time.perf_counter() - began
                acknowledgement_seconds += float(record["ack_s"])
                generated.extend(accepted)
                decode_rounds += 1
                verified_drafts += len(window) - 1
                accepted_drafts += keep - 1
                record.update({
                    "position_end": answer.committed,
                    "accepted_tokens": len(accepted),
                    "verified_drafts": len(window) - 1,
                    "accepted_drafts": keep - 1,
                    "next_drafts": len(proposals),
                    "total_s": float(record["spark_mcdma_roundtrip_s"])
                    + float(record["metal_s"]) + float(record["ack_s"]),
                })
                steps.append(record)
                committed = answer.committed
                step += 1
            if shutdown:
                stop = protocol.control(
                    kind=protocol.Kind.SHUTDOWN,
                    request_id=request_id,
                    step=step,
                    committed=committed,
                    generation=mailbox.generation,
                    timeout_s=timeout,
                )
                _expect_ack(mailbox.call(protocol.pack(stop), timeout), stop)
            else:
                reset = protocol.control(
                    kind=protocol.Kind.RESET,
                    request_id=request_id,
                    step=step,
                    committed=committed,
                    generation=mailbox.generation,
                    timeout_s=timeout,
                )
                expected = protocol.control(
                    kind=protocol.Kind.ACK,
                    request_id=request_id,
                    step=0,
                    committed=0,
                    generation=mailbox.generation,
                    timeout_s=timeout,
                )
                began = time.perf_counter()
                _expect_ack(mailbox.call(protocol.pack(reset), timeout), expected)
                reset_seconds += time.perf_counter() - began
                stage.reset()
        except Exception:
            stage.reset()
            try:
                reset = protocol.control(
                    kind=protocol.Kind.RESET,
                    request_id=request_id,
                    step=step,
                    committed=committed,
                    generation=mailbox.generation,
                    timeout_s=timeout,
                )
                mailbox.call(protocol.pack(reset), timeout)
            except Exception:
                pass
            raise
    elapsed = time.perf_counter() - started
    return {
        "request_id": request_id,
        "generation": mailbox_instance.generation,
        "prompt_tokens": len(prompt_tokens),
        "generated_tokens": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "committed": committed,
        "elapsed_s": elapsed,
        "tokens_per_s": len(generated) / elapsed if elapsed else 0.0,
        "first_token_s": first_token_seconds,
        "tokenize_s": tokenize_seconds,
        "mailbox_open_s": mailbox_open_seconds,
        "reset_s": reset_seconds,
        "metal_s": metal_seconds,
        "spark_mcdma_roundtrip_s": roundtrip_seconds,
        "ack_s": acknowledgement_seconds,
        "steps": steps,
        "decode_rounds": decode_rounds,
        "verified_drafts": verified_drafts,
        "accepted_drafts": accepted_drafts,
        "tokens_per_decode_round": max(0, len(generated) - 1) / decode_rounds if decode_rounds else 0.0,
        "draft_acceptance_rate": accepted_drafts / verified_drafts if verified_drafts else 0.0,
        "context_capacity": stage.context_capacity,
        "finish_reason": "stop" if generated[-1] in eos else "length",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--mailbox", default="tfglm53")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--drafts", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--shutdown", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(
        args.model,
        args.prompt,
        args.max_tokens,
        args.mailbox,
        args.timeout,
        args.shutdown,
        drafts=args.drafts,
        confidence=args.confidence,
    ), ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
