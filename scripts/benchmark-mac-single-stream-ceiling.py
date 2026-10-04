#!/usr/bin/env python3
"""Measure the Mac's one-request GLM decode ceiling without transport cost."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from experiments.three_machine.cache_handoff import prompt_digest
from experiments.three_machine.mac_decode import (
    _load_runtime,
    _verify_snapshot_path,
    render_user_prompt,
    run_decode,
)


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty metric")
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


class RuntimeCallProfiler:
    """Measure host time spent constructing each lazy TensorFold phase."""

    METHOD_NAMES = (
        "hidden",
        "head",
        "speculate",
        "prepare_settle",
        "settle",
        "keep_rows",
        "unspeculate",
    )

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.originals: dict[str, Any] = {}
        self.samples: dict[str, list[float]] = {}
        for name in self.METHOD_NAMES:
            original = getattr(runtime, name, None)
            if original is None:
                continue
            self.originals[name] = original

            def measured(*args: Any, _name: str = name, _original: Any = original,
                         **kwargs: Any) -> Any:
                started = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.samples.setdefault(_name, []).append(time.perf_counter() - started)

            setattr(runtime, name, measured)

    def reset(self) -> None:
        self.samples = {}

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "calls": len(values),
                "total_ms": sum(values) * 1e3,
                "mean_ms": statistics.mean(values) * 1e3,
                "max_ms": max(values) * 1e3,
            }
            for name, values in sorted(self.samples.items())
            if values
        }

    def stop(self) -> None:
        for name, original in self.originals.items():
            setattr(self.runtime, name, original)


class _ProfiledCallable:
    def __init__(self, target: Any, label: str, profiler: "LayerCallProfiler") -> None:
        self.target = target
        self.label = label
        self.profiler = profiler

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self.target(*args, **kwargs)
        finally:
            self.profiler.record(self.label, time.perf_counter() - started)


class LayerCallProfiler:
    """Break the GLM host-build phase into attention, MLP, and boundary work."""

    MODEL_METHODS = ("embed_tokens", "boundary", "final_norm")

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.model = runtime.model
        self.stats: dict[str, dict[str, float | int]] = {}
        self.model_originals: dict[str, Any] = {}
        self.layer_originals: list[tuple[Any, str, Any]] = []
        for name in self.MODEL_METHODS:
            original = getattr(self.model, name, None)
            if original is None:
                continue
            self.model_originals[name] = original

            def measured(*args: Any, _name: str = name, _original: Any = original,
                         **kwargs: Any) -> Any:
                started = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.record(_name, time.perf_counter() - started)

            setattr(self.model, name, measured)
        for layer in self.model.layers:
            for name in ("attn", "mlp"):
                original = getattr(layer, name)
                label = f"{name}.{type(original).__name__}"
                self.layer_originals.append((layer, name, original))
                setattr(layer, name, _ProfiledCallable(original, label, self))

    def record(self, label: str, elapsed: float) -> None:
        stat = self.stats.setdefault(label, {"calls": 0, "total_ms": 0.0, "max_ms": 0.0})
        stat["calls"] = int(stat["calls"]) + 1
        elapsed_ms = elapsed * 1e3
        stat["total_ms"] = float(stat["total_ms"]) + elapsed_ms
        stat["max_ms"] = max(float(stat["max_ms"]), elapsed_ms)

    def reset(self) -> None:
        self.stats = {}

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                **values,
                "mean_ms": float(values["total_ms"]) / int(values["calls"]),
            }
            for name, values in sorted(self.stats.items())
            if int(values["calls"])
        }

    def stop(self) -> None:
        for layer, name, original in self.layer_originals:
            setattr(layer, name, original)
        for name, original in self.model_originals.items():
            setattr(self.model, name, original)


def summarize_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one measured run is required")
    if any(len(run.get("tokens", [])) != max_new_tokens for run in runs):
        raise RuntimeError("Mac ceiling round stopped before the fixed output length")
    identities = {
        (str(run["token_sha256"]), _digest_text(str(run["text"]))) for run in runs
    }
    if len(identities) != 1:
        raise RuntimeError("identical Mac ceiling rounds returned different output")
    token_hash, text_hash = next(iter(identities))
    return {
        "prompt_tokens": len(prompt_ids),
        "prompt_token_sha256": prompt_digest(prompt_ids),
        "completion_tokens": max_new_tokens,
        "token_sha256": token_hash,
        "text_sha256": text_hash,
        "total_seconds": _summary([float(run["total_seconds"]) for run in runs]),
        "ttft_seconds": _summary([float(run["first_token_seconds"]) for run in runs]),
        "decode_seconds": _summary([float(run["generation_seconds"]) for run in runs]),
        "decode_tokens_per_second": _summary(
            [float(run["decode_tokens_per_second"]) for run in runs]
        ),
        "runs": [dict(run) for run in runs],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mtp-drafts", type=int, choices=range(1, 9), default=3)
    parser.add_argument(
        "--drafter", type=Path,
        help="optional DFlash2 snapshot used by the actual Mac decode runtime",
    )
    parser.add_argument(
        "--drafter-revision",
        help="content-addressed DFlash2 revision; required with --drafter",
    )
    parser.add_argument("--drafter-bits", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--copy-drafts", action=argparse.BooleanOptionalAction, default=True,
        help="use TensorFold's normal server-side suffix copy drafts",
    )
    parser.add_argument("--copy-min-match", type=int, default=4)
    parser.add_argument(
        "--component-profile", action="store_true",
        help="record host-side lazy-graph construction time by runtime method",
    )
    parser.add_argument(
        "--layer-profile", action="store_true",
        help="split the hidden-pass host time into GLM layer components",
    )
    parser.add_argument(
        "--eval-every", type=int,
        help="override TensorFold's GLM partial-launch interval (0 disables partial launches)",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 2 or args.warmups < 1 or args.repeats < 3:
        parser.error("max-new-tokens >=2, warmups >=1, and repeats >=3 are required")
    if args.eval_every is not None and args.eval_every < 0:
        parser.error("eval-every must be nonnegative")
    if bool(args.drafter) != bool(args.drafter_revision):
        parser.error("--drafter and --drafter-revision must be provided together")

    model_dir = _verify_snapshot_path(args.model, args.model_revision)
    drafter_dir = (
        _verify_snapshot_path(args.drafter, args.drafter_revision)
        if args.drafter is not None
        else None
    )
    import tensorfold
    from tensorfold.families.glm5_next import config as glm_config

    if args.eval_every is not None:
        glm_config.EVAL_EVERY = int(args.eval_every)
    eval_every = int(glm_config.EVAL_EVERY)

    load_started = time.perf_counter()
    runtime, tokenizer = _load_runtime(
        model_dir,
        mtp_drafts=args.mtp_drafts,
        drafter=str(drafter_dir or ""),
        drafter_bits=args.drafter_bits,
    )
    load_seconds = time.perf_counter() - load_started
    prompt_ids = render_user_prompt(tokenizer, args.prompt)
    profiler = RuntimeCallProfiler(runtime) if args.component_profile else None
    layer_profiler = LayerCallProfiler(runtime) if args.layer_profile else None
    try:
        warmups = [
            run_decode(
                runtime,
                tokenizer,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                stream_id=f"mac-ceiling-warmup-{index}",
                copy_drafts=args.copy_drafts,
                copy_min_match=args.copy_min_match,
            )
            for index in range(args.warmups)
        ]
        if profiler is not None:
            profiler.reset()
        if layer_profiler is not None:
            layer_profiler.reset()
        measured = [
            run_decode(
                runtime,
                tokenizer,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                stream_id=f"mac-ceiling-{index}",
                copy_drafts=args.copy_drafts,
                copy_min_match=args.copy_min_match,
            )
            for index in range(args.repeats)
        ]
        component_profile = profiler.snapshot() if profiler is not None else None
        layer_profile = layer_profiler.snapshot() if layer_profiler is not None else None
    finally:
        if layer_profiler is not None:
            layer_profiler.stop()
        if profiler is not None:
            profiler.stop()
    result = summarize_runs(
        measured,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
    )
    result.update({
        "schema": 1,
        "event": "glm_mac_single_stream_ceiling",
        "model_revision": args.model_revision,
        "tensorfold_version": tensorfold.__version__,
        "load_seconds": load_seconds,
        "warmup_count": args.warmups,
        "repeat_count": args.repeats,
        "mtp_drafts": args.mtp_drafts,
        "drafter": str(drafter_dir) if drafter_dir is not None else None,
        "drafter_revision": args.drafter_revision,
        "drafter_bits": args.drafter_bits if drafter_dir is not None else None,
        "copy_drafts": args.copy_drafts,
        "copy_min_match": args.copy_min_match,
        "eval_every": eval_every,
        "warmups": warmups,
    })
    if component_profile is not None:
        result["component_profile"] = component_profile
    if layer_profile is not None:
        result["layer_profile"] = layer_profile
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
