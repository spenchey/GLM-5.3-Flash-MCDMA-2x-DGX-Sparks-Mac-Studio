#!/usr/bin/env python3
"""Measure DFlash draft-window confidence scaling on one loaded Mac runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.three_machine.mac_decode import (
    _load_runtime,
    _verify_snapshot_path,
    render_user_prompt,
    run_decode,
)


DEFAULT_PROMPT = (
    "Write a detailed step-by-step explanation of how a hash map works, "
    "including collision handling, resizing, and time complexity. Be thorough."
)


def _floats(value: str) -> list[float]:
    try:
        values = [float(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be comma-separated numbers") from exc
    if not values or any(number <= 0 for number in values):
        raise argparse.ArgumentTypeError("values must be positive")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must not repeat")
    return values


def _median(runs: Sequence[dict[str, Any]], field: str) -> float:
    return statistics.median(float(run[field]) for run in runs)


def _balanced(values: Sequence[float], repeats: int) -> list[list[float]]:
    out: list[list[float]] = []
    for index in range(repeats):
        order = list(values if index % 2 == 0 else reversed(values))
        shift = (index // 2) % len(order)
        out.append(order[shift:] + order[:shift])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--drafter", type=Path, required=True)
    parser.add_argument("--drafter-revision", required=True)
    parser.add_argument("--drafter-bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--edge", type=float, default=1.4)
    parser.add_argument("--taus", type=_floats, default=_floats("0.5,0.75,1,1.25,1.5,2"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.edge < 0 or args.max_new_tokens < 2 or args.repeats < 3:
        parser.error("edge >=0, max-new-tokens >=2, and repeats >=3 are required")

    model_dir = _verify_snapshot_path(args.model, args.model_revision)
    drafter_dir = _verify_snapshot_path(args.drafter, args.drafter_revision)
    from tensorfold.families.qwen3_5.dflash_head import DraftSlot

    policy = {"tau": 1.5}
    original_get = DraftSlot.get

    def tuned_get(slot: DraftSlot, sampling: Any) -> Any:
        proposer = original_get(slot, sampling)
        proposer.tree_edge = float(args.edge)
        proposer.tree_tau = float(policy["tau"])
        proposer.async_layers = (0, 1, 2, 3, 4)
        return proposer

    DraftSlot.get = tuned_get
    try:
        runtime, tokenizer = _load_runtime(
            model_dir,
            mtp_drafts=2,
            drafter=str(drafter_dir),
            drafter_bits=args.drafter_bits,
        )
        prompt_ids = render_user_prompt(tokenizer, args.prompt)
        run_decode(
            runtime,
            tokenizer,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            stream_id="dflash-tau-warmup",
            copy_drafts=False,
            force_length=True,
        )
        runs: dict[float, list[dict[str, Any]]] = {tau: [] for tau in args.taus}
        expected_hash = None
        for repeat, order in enumerate(_balanced(args.taus, args.repeats)):
            for tau in order:
                policy["tau"] = float(tau)
                run = run_decode(
                    runtime,
                    tokenizer,
                    prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    stream_id=f"dflash-tau-{tau:g}-run-{repeat}",
                    copy_drafts=False,
                    force_length=True,
                )
                token_hash = str(run["token_sha256"])
                expected_hash = token_hash if expected_hash is None else expected_hash
                if token_hash != expected_hash:
                    raise RuntimeError("DFlash tau settings returned different greedy outputs")
                runs[tau].append(run)
                print(
                    f"TAU {tau:g} ROUND {repeat} TOTAL {run['total_seconds']:.6f} "
                    f"DECODE {run['decode_tokens_per_second']:.3f} "
                    f"ROUNDS {run['decode_rounds']} DRAFTED {run['drafted_tokens']} "
                    f"ACCEPTED {run['accepted_draft_tokens']}",
                    flush=True,
                )
    finally:
        DraftSlot.get = original_get

    summary = []
    for tau in args.taus:
        values = runs[tau]
        summary.append({
            "tau": tau,
            "median_total_seconds": _median(values, "total_seconds"),
            "median_ttft_seconds": _median(values, "first_token_seconds"),
            "median_decode_tokens_per_second": _median(values, "decode_tokens_per_second"),
            "median_rounds": _median(values, "decode_rounds"),
            "median_drafted_tokens": _median(values, "drafted_tokens"),
            "median_accepted_tokens": _median(values, "accepted_draft_tokens"),
            "runs": values,
        })
    print(json.dumps({
        "event": "glm_mac_dflash_tau_sweep",
        "edge": args.edge,
        "async_layers": [0, 1, 2, 3, 4],
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": args.max_new_tokens,
        "model_revision": args.model_revision,
        "drafter_revision": args.drafter_revision,
        "drafter_bits": args.drafter_bits,
        "repeats": args.repeats,
        "token_sha256": expected_hash,
        "summary": summary,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
