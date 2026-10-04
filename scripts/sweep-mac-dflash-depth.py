#!/usr/bin/env python3
"""Find the fastest verified DFlash depth before paying for MCDMA trials."""

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


def _depths(value: str) -> list[int]:
    depths = [int(item) for item in value.split(",") if item]
    if not depths or any(depth < 1 or depth > 7 for depth in depths):
        raise argparse.ArgumentTypeError("depths must be comma-separated integers in 1..7")
    if len(depths) != len(set(depths)):
        raise argparse.ArgumentTypeError("depths must not repeat")
    return depths


def _median(runs: Sequence[dict[str, Any]], field: str) -> float:
    return statistics.median(float(run[field]) for run in runs)


def _configure_depth(runtime: Any, depth: int) -> None:
    runtime.drafts = int(depth)
    runtime.head_drafts.nodes = int(depth)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--drafter-bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--depths", type=_depths, default=_depths("1,2,3,4,5,6,7"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.max_new_tokens < 2 or args.warmups < 0 or args.repeats < 3:
        parser.error("max-new-tokens >=2, warmups >=0, and repeats >=3 are required")

    model_dir = _verify_snapshot_path(args.model, args.model_revision)
    runtime, tokenizer = _load_runtime(
        model_dir,
        mtp_drafts=max(args.depths),
        drafter=args.drafter,
        drafter_bits=args.drafter_bits,
    )
    prompt_ids = render_user_prompt(tokenizer, args.prompt)
    results = []
    expected_hash = None
    for depth in args.depths:
        _configure_depth(runtime, depth)
        for index in range(args.warmups):
            run_decode(
                runtime,
                tokenizer,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                stream_id=f"dflash-depth-{depth}-warmup-{index}",
                copy_drafts=False,
                force_length=True,
            )
        runs = [
            run_decode(
                runtime,
                tokenizer,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                stream_id=f"dflash-depth-{depth}-run-{index}",
                copy_drafts=False,
                force_length=True,
            )
            for index in range(args.repeats)
        ]
        hashes = {str(run["token_sha256"]) for run in runs}
        if len(hashes) != 1:
            raise RuntimeError(f"DFlash depth {depth} returned different greedy outputs")
        token_hash = next(iter(hashes))
        expected_hash = token_hash if expected_hash is None else expected_hash
        if token_hash != expected_hash:
            raise RuntimeError("DFlash depths returned different greedy outputs")
        results.append({
            "depth": depth,
            "median_complete_seconds": _median(runs, "total_seconds"),
            "median_ttft_seconds": _median(runs, "first_token_seconds"),
            "median_decode_tokens_per_second": _median(runs, "decode_tokens_per_second"),
            "median_rounds": _median(runs, "decode_rounds"),
            "median_drafted_tokens": _median(runs, "drafted_tokens"),
            "median_accepted_tokens": _median(runs, "accepted_draft_tokens"),
            "token_sha256": token_hash,
            "runs": runs,
        })

    winner = min(results, key=lambda item: item["median_complete_seconds"])
    print(json.dumps({
        "event": "glm_mac_dflash_depth_sweep",
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": args.max_new_tokens,
        "drafter": args.drafter,
        "drafter_bits": args.drafter_bits,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "winner_depth": winner["depth"],
        "winner_complete_seconds": winner["median_complete_seconds"],
        "results": results,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
