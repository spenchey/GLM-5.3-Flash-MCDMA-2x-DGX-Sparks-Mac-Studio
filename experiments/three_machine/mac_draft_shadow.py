"""Measure GLM's MTP proposal beside the active Mac DFlash2 proposal.

This is an experiment runtime, not a serving default.  DFlash2 still supplies
every proposal that TensorFold verifies.  The checkpoint's MTP head consumes
the same committed target rows, predicts one next token, and records whether
that token would have helped on a round where DFlash2's first token missed.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Sequence


class ShadowDraftSlot:
    """Factory namespace replaced with the real DraftSlot subclass after MLX imports."""


def _classes() -> tuple[type[Any], type[Any]]:
    """Build the MLX-dependent slot and runtime classes lazily on the Studio."""

    import mlx.core as mx

    from tensorfold.families.glm5_next.runtime import GLMDFlash, MTPCache
    from tensorfold.families.qwen3_5.dflash_head import DraftSlot

    class _ShadowDraftSlot(DraftSlot):
        """DFlash state plus the checkpoint MTP cache and one pending prediction."""

        def __init__(self, drafter: Any, chains: bool = False) -> None:
            super().__init__(drafter, chains)
            self.mtp_cache = MTPCache()
            self.mtp_ready = False
            self.mtp_pending: int | None = None
            self.mtp_pending_position: int | None = None

        @property
        def state(self) -> list[Any]:
            return [*super().state, *self.mtp_cache.state]

        @property
        def nbytes(self) -> int:
            return super().nbytes + sum(int(value.nbytes) for value in self.mtp_cache.state)

        def __copy__(self) -> "_ShadowDraftSlot":
            # DraftSlot deliberately copies only an unread prompt context.  Keep
            # that contract, then copy the nested MTP cache the lane copier
            # cannot otherwise see.
            base = super().__copy__()
            slot = _ShadowDraftSlot(self.drafter, self.chains)
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
            return slot

    class GLMDFlashMTPShadow(GLMDFlash):
        """DFlash2 decode with a non-authoritative one-token MTP prediction."""

        def __init__(self, model: Any, drafter: Any, mtp_head: Any, *, drafts: int,
                     check: bool = True) -> None:
            super().__init__(model, drafter, drafts=drafts, check=check)
            self.shadow_mtp = mtp_head
            self.shadow_totals: dict[str, int] = {}
            self.reset_shadow()

        def reset_shadow(self) -> None:
            self.shadow_totals = {
                "observations": 0,
                "both_hit": 0,
                "mtp_only": 0,
                "dflash_only": 0,
                "neither": 0,
                "mtp_predictions": 0,
            }

        def make_cache(self) -> list[Any]:
            return [*self.model.make_cache(), _ShadowDraftSlot(self.head_drafts.drafter,
                                                                self.head_drafts.chains)]

        def adopt_cache(self, cache: list[Any]) -> list[Any]:
            if not (cache and isinstance(cache[-1], _ShadowDraftSlot)):
                cache.append(_ShadowDraftSlot(self.head_drafts.drafter, self.head_drafts.chains))
            return cache

        @staticmethod
        def _slot(cache: list[Any]) -> _ShadowDraftSlot:
            slot = cache[-1]
            if not isinstance(slot, _ShadowDraftSlot):
                raise TypeError("GLM draft shadow cache is missing its combined slot")
            return slot

        def _mtp_absorb(self, cache: list[Any], rows: Any, tokens: Any) -> Any:
            slot = self._slot(cache)
            values = tokens if isinstance(tokens, mx.array) else mx.array(tokens)
            values = values.reshape(-1).astype(mx.uint32)
            count = int(values.shape[0])
            return self.shadow_mtp(
                self.model,
                rows.reshape(-1, rows.shape[-1])[:count],
                values,
                [slot.mtp_cache],
                (count,),
                count <= self.fused_rows,
            )

        def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any],
                                 start: int = 0) -> None:
            # Preserve the real DFlash prompt context, then build the shadow MTP
            # context from the same final-normalized rows and following tokens.
            super().absorb_draft_context(hidden, next_tokens, cache, start)
            values = next_tokens if isinstance(next_tokens, mx.array) else mx.array(next_tokens)
            values = values.reshape(-1).astype(mx.uint32)
            count = int(values.shape[0])
            if count:
                rows = hidden.reshape(-1, hidden.shape[-1])
                self._mtp_absorb(cache, rows[int(start):int(start) + count], values)
                self._slot(cache).mtp_ready = True

        def _record_previous(self, slot: _ShadowDraftSlot, follow: Sequence[int],
                             kept_rows: Sequence[int]) -> None:
            if slot.mtp_pending is None or not follow:
                return
            mtp_hit = int(slot.mtp_pending) == int(follow[0])
            dflash_hit = len(kept_rows) > 1
            self.shadow_totals["observations"] += 1
            if mtp_hit and dflash_hit:
                key = "both_hit"
            elif mtp_hit:
                key = "mtp_only"
            elif dflash_hit:
                key = "dflash_only"
            else:
                key = "neither"
            self.shadow_totals[key] += 1

        def speculate(self, cache: list[Any], tokens: Any, position: int, sampling: Any,
                      start: int = 0, last_only: bool = False,
                      rows: Sequence[int] | None = None) -> Any:
            # DFlash remains authoritative.  Its return value goes to settle()
            # exactly as it does in the production experiment runtime.
            result = super().speculate(
                cache, tokens, position, sampling, start=start,
                last_only=last_only, rows=rows,
            )
            follow = [int(token) for token in (
                tokens.reshape(-1).tolist() if hasattr(tokens, "tolist") else tokens
            )]
            _, _, first = self._last[id(cache)]
            kept = ([int(row) for row in rows] if rows is not None else
                    list(range(first + int(start), first + int(start) + len(follow))))
            slot = self._slot(cache)
            self._record_previous(slot, follow, kept)
            slot.mtp_pending = None
            slot.mtp_pending_position = None
            if not slot.mtp_ready or not follow or not kept:
                return result
            index = mx.array(kept, dtype=mx.int32)
            target_rows = mx.take(self._rows, index, axis=0)
            out = self._mtp_absorb(cache, target_rows, follow)
            from tensorfold.engine.gpu_sampling import sample as gpu_sample

            prediction_position = int(position) + 1 + len(follow)
            pending = gpu_sample(
                self.shadow_mtp.logits(self.model, out[-1:]),
                sampling,
                [prediction_position],
            )
            mx.eval(pending)
            slot.mtp_pending = int(pending.item())
            slot.mtp_pending_position = prediction_position
            self.shadow_totals["mtp_predictions"] += 1
            return result

    return _ShadowDraftSlot, GLMDFlashMTPShadow


def load_shadow_runtime(model_dir: Path, drafter_dir: Path, *, drafts: int = 2,
                        drafter_bits: int = 4, check: bool = True) -> tuple[Any, Any]:
    """Load one GLM target with both draft heads, leaving DFlash authoritative."""

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
        raise ValueError("draft shadow needs at least one DFlash row")
    model.weights = None
    _slot, runtime_class = _classes()
    runtime = runtime_class(model, drafter, mtp_head, drafts=drafts, check=check)
    print(
        f"[glm5] DFlash2 block with one-token MTP shadow, drafts up to {runtime.drafts}",
        flush=True,
    )
    return runtime, tokenizer


__all__ = ["load_shadow_runtime"]
