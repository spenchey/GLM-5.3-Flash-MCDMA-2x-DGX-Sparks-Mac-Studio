"""Experimental Mac selector between GLM's MTP and DFlash2 drafters.

The target model still verifies every proposed token.  Only one draft head runs
per round.  A small measured bandit, matching TensorFold CUDA's policy shape,
chooses the head by committed tokens per elapsed round millisecond.

This module is an isolated benchmark runtime, not a serving default.  The
measured policy was rejected by ``MAC-DRAFT-SELECTOR-001`` and is retained only
so that the negative result can be reproduced or studied.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Sequence


def _other(arm: str) -> str:
    return "m" if arm == "f" else "f"


@dataclass
class AdaptiveDraftChoice:
    """Choose DFlash (``f``) or MTP (``m``) from measured useful work."""

    first: str = "f"
    explore: int = 2
    every: int = 8
    margin: float = 0.03
    window: int = 6
    recheck: int = 3
    rounds: list[tuple[str, int, float]] = field(default_factory=list)
    choice: str = field(init=False)
    run: int = 0
    since: dict[str, int] = field(default_factory=lambda: {"m": 0, "f": 0})

    def __post_init__(self) -> None:
        if self.first not in ("m", "f"):
            raise ValueError("first drafter must be 'm' or 'f'")
        if self.explore < 1 or self.every < 0 or self.window < 1 or self.recheck < 0:
            raise ValueError("selector counts are out of range")
        if self.margin < 0:
            raise ValueError("selector margin must be nonnegative")
        self.choice = self.first

    def rate(self, arm: str) -> float | None:
        rows = [row for row in self.rounds[self.since[arm]:] if row[0] == arm][-self.window:]
        elapsed = sum(row[2] for row in rows)
        return sum(row[1] for row in rows) / elapsed if rows and elapsed > 0 else None

    def pick(self) -> str:
        n = len(self.rounds)
        if n < self.explore:
            return self.first
        if n < 2 * self.explore:
            return _other(self.first)
        current = self.choice
        current_rate, other_rate = self.rate(current), self.rate(_other(current))
        required = 1.0 if n == 2 * self.explore else 1.0 + self.margin
        if other_rate is not None and (current_rate is None or other_rate > current_rate * required):
            self.since[current] = n
            self.choice = current = _other(current)
            self.run = max(0, self.every - self.recheck) if self.recheck else 0
        if self.every and self.run >= self.every:
            self.run = 0
            return _other(current)
        self.run += 1
        return current

    def record(self, arm: str, committed: int, milliseconds: float) -> None:
        if arm not in ("m", "f") or committed < 1 or milliseconds <= 0:
            raise ValueError("invalid selector observation")
        self.rounds.append((arm, int(committed), float(milliseconds)))


def _classes() -> tuple[type[Any], type[Any]]:
    """Build the MLX-dependent slot and runtime classes lazily on the Studio."""

    import mlx.core as mx

    from tensorfold.families.glm5_next.runtime import GLMDFlash, MTPCache
    from tensorfold.families.qwen3_5.dflash_head import DraftSlot

    class AdaptiveDraftSlot(DraftSlot):
        """Both draft contexts, an MTP catch-up queue, and one selector."""

        def __init__(self, drafter: Any, chains: bool = False, **choice: Any) -> None:
            super().__init__(drafter, chains)
            self.mtp_cache = MTPCache()
            self.mtp_ready = False
            self.mtp_rows: list[Any] = []
            self.mtp_tokens: list[Any] = []
            self.selector = AdaptiveDraftChoice(**choice)
            self.last_arm: str | None = None
            self.last_started: float | None = None
            self.last_depth = 0

        @property
        def state(self) -> list[Any]:
            return [*super().state, *self.mtp_cache.state]

        @property
        def nbytes(self) -> int:
            queued = [*self.mtp_rows, *self.mtp_tokens]
            return (super().nbytes
                    + sum(int(value.nbytes) for value in self.mtp_cache.state)
                    + sum(int(value.nbytes) for value in queued))

        def __copy__(self) -> "AdaptiveDraftSlot":
            base = super().__copy__()
            slot = AdaptiveDraftSlot(
                self.drafter,
                self.chains,
                first=self.selector.first,
                explore=self.selector.explore,
                every=self.selector.every,
                margin=self.selector.margin,
                window=self.selector.window,
                recheck=self.selector.recheck,
            )
            slot.proposer = base.proposer
            for name, value in vars(self.mtp_cache).items():
                if isinstance(value, mx.array):
                    value = mx.contiguous(value)
                elif isinstance(value, list):
                    value = list(value)
                else:
                    value = copy.copy(value)
                setattr(slot.mtp_cache, name, value)
            slot.mtp_ready = bool(self.mtp_ready)
            slot.mtp_rows = [mx.contiguous(value) for value in self.mtp_rows]
            slot.mtp_tokens = [mx.contiguous(value) for value in self.mtp_tokens]
            slot.selector.rounds = list(self.selector.rounds)
            slot.selector.choice = self.selector.choice
            slot.selector.run = int(self.selector.run)
            slot.selector.since = dict(self.selector.since)
            return slot

    class GLMAdaptiveDraft(GLMDFlash):
        """DFlash and MTP with one measured draft head selected per round."""

        def __init__(self, model: Any, drafter: Any, mtp_head: Any, *, drafts: int,
                     check: bool = True, **choice: Any) -> None:
            super().__init__(model, drafter, drafts=drafts, check=check)
            self.adaptive_mtp = mtp_head
            self.choice = dict(choice)
            self.selection_records: list[dict[str, Any]] = []

        def reset_selection(self) -> None:
            self.selection_records = []

        def make_cache(self) -> list[Any]:
            return [*self.model.make_cache(), AdaptiveDraftSlot(
                self.head_drafts.drafter, self.head_drafts.chains, **self.choice
            )]

        def adopt_cache(self, cache: list[Any]) -> list[Any]:
            if not (cache and isinstance(cache[-1], AdaptiveDraftSlot)):
                cache.append(AdaptiveDraftSlot(
                    self.head_drafts.drafter, self.head_drafts.chains, **self.choice
                ))
            return cache

        @staticmethod
        def _slot(cache: list[Any]) -> AdaptiveDraftSlot:
            slot = cache[-1]
            if not isinstance(slot, AdaptiveDraftSlot):
                raise TypeError("GLM adaptive cache is missing its combined draft slot")
            return slot

        def _mtp_absorb(self, slot: AdaptiveDraftSlot, rows: Any, tokens: Any) -> Any:
            values = tokens if isinstance(tokens, mx.array) else mx.array(tokens)
            values = values.reshape(-1).astype(mx.uint32)
            count = int(values.shape[0])
            self._trim_chained(slot.mtp_cache)
            return self.adaptive_mtp(
                self.model,
                rows.reshape(-1, rows.shape[-1])[:count],
                values,
                [slot.mtp_cache],
                (count,),
                count <= self.fused_rows,
            )

        def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any],
                                 start: int = 0) -> None:
            super().absorb_draft_context(hidden, next_tokens, cache, start)
            values = next_tokens if isinstance(next_tokens, mx.array) else mx.array(next_tokens)
            values = values.reshape(-1).astype(mx.uint32)
            count = int(values.shape[0])
            if count:
                rows = hidden.reshape(-1, hidden.shape[-1])
                slot = self._slot(cache)
                self._mtp_absorb(slot, rows[int(start):int(start) + count], values)
                slot.mtp_ready = True

        def _queue_mtp(self, slot: AdaptiveDraftSlot, rows: Any, tokens: Any) -> None:
            row_copy = mx.contiguous(rows)
            token_copy = mx.contiguous(tokens.reshape(-1).astype(mx.uint32))
            mx.async_eval(row_copy, token_copy)
            slot.mtp_rows.append(row_copy)
            slot.mtp_tokens.append(token_copy)

        def speculate(self, cache: list[Any], tokens: Any, position: int, sampling: Any,
                      start: int = 0, last_only: bool = False,
                      rows: Sequence[int] | None = None) -> Any:
            result = super().speculate(
                cache, tokens, position, sampling, start=start, last_only=last_only, rows=rows
            )
            slot = self._slot(cache)
            if not slot.mtp_ready:
                return result
            follow = tokens if isinstance(tokens, mx.array) else mx.array(tokens)
            follow = follow.reshape(-1).astype(mx.uint32)
            _, _, first = self._last[id(cache)]
            kept = ([int(row) for row in rows] if rows is not None else
                    list(range(first + int(start), first + int(start) + int(follow.shape[0]))))
            if kept:
                index = mx.array(kept, dtype=mx.int32)
                self._queue_mtp(slot, mx.take(self._rows, index, axis=0), follow)
            return result

        def _mtp_tree(self, slot: AdaptiveDraftSlot, position: int, sampling: Any,
                      count: int) -> Any:
            if not slot.mtp_rows or count <= 0:
                return []
            rows = mx.concatenate(slot.mtp_rows, axis=0)
            tokens = mx.concatenate(slot.mtp_tokens, axis=0)
            slot.mtp_rows, slot.mtp_tokens = [], []
            state = self._mtp_absorb(slot, rows, tokens)[-1:]
            from tensorfold.engine.gpu_sampling import sample as gpu_sample

            head = gpu_sample(
                self.adaptive_mtp.logits(self.model, state), sampling, [int(position)]
            ).reshape(1).astype(mx.uint32)
            chain = [head]
            for step in range(1, int(count)):
                state = self.adaptive_mtp(
                    self.model, state, chain[-1], [slot.mtp_cache], (1,), True
                )
                slot.mtp_cache.drafted += 1
                chain.append(gpu_sample(
                    self.adaptive_mtp.logits(self.model, state),
                    sampling,
                    [int(position) + step],
                ).reshape(1).astype(mx.uint32))
            drafts = mx.concatenate(chain)
            mx.async_eval(drafts)
            return drafts

        def _record_previous(self, slot: AdaptiveDraftSlot, keep: int) -> None:
            if slot.last_arm is None or slot.last_started is None:
                return
            milliseconds = (time.perf_counter() - slot.last_started) * 1e3
            slot.selector.record(slot.last_arm, int(keep), milliseconds)
            self.selection_records.append({
                "arm": slot.last_arm,
                "committed": int(keep),
                "depth": int(slot.last_depth),
                "milliseconds": milliseconds,
            })
            slot.last_arm = None
            slot.last_started = None

        def settle(self, cache: list[Any], keep: int, first: Any, position: int,
                   sampling: Any, count: int) -> Any:
            slot = self._slot(cache)
            self._record_previous(slot, keep)
            if count <= 0:
                return []
            arm = slot.selector.pick() if slot.mtp_ready else "f"
            slot.last_arm = arm
            slot.last_depth = int(count)
            slot.last_started = time.perf_counter()
            if arm == "m":
                drafts = self._mtp_tree(slot, position, sampling, count)
                if len(drafts):
                    return drafts
                arm = slot.last_arm = "f"
            return self.head_drafts.tree(cache, position, sampling, count)

        def draft_probabilities(self, cache: list[Any]) -> list[float] | None:
            slot = self._slot(cache)
            return None if slot.last_arm == "m" else self.head_drafts.probabilities(cache)

    return AdaptiveDraftSlot, GLMAdaptiveDraft


def load_adaptive_runtime(model_dir: Path, drafter_dir: Path, *, drafts: int = 2,
                          drafter_bits: int = 4, check: bool = True,
                          **choice: Any) -> tuple[Any, Any]:
    """Load the target and both draft heads into the isolated adaptive runtime."""

    import mlx.core as mx

    from tensorfold.drafters.dflash_drafter import DFlashDrafter
    from tensorfold.families.glm5_next import mtp as mtp_module
    from tensorfold.families.glm5_next import weights as glm_weights

    if mx.metal.is_available():
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        limit = int(info.get("max_recommended_working_set_size", 0))
        if limit:
            mx.set_wired_limit(limit)
    model, tokenizer = glm_weights.load(Path(model_dir))
    mtp_head = mtp_module.load(model)
    drafter = DFlashDrafter(model, str(drafter_dir), bits=int(drafter_bits))
    width = min(int(drafter.block_size) - 1, 7)
    drafts = min(int(drafts), width)
    if drafts < 1:
        raise ValueError("adaptive drafting needs at least one draft row")
    model.weights = None
    _slot, runtime_class = _classes()
    runtime = runtime_class(
        model, drafter, mtp_head, drafts=drafts, check=check, **choice
    )
    print(
        f"[glm5] adaptive DFlash2/MTP selector, drafts up to {runtime.drafts}",
        flush=True,
    )
    return runtime, tokenizer


__all__ = ["AdaptiveDraftChoice", "load_adaptive_runtime"]
