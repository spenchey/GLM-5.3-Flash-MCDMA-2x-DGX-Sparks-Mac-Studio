#!/usr/bin/env python3
"""Reproduce the rejected MAC-DRAFT-SELECTOR-001 Mac experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any

from experiments.three_machine.mac_decode import (
    _verify_snapshot_path,
    render_user_prompt,
    run_decode,
)
from experiments.three_machine.mac_draft_selector import load_adaptive_runtime


DEFAULT_PROMPT = (
    "Write a detailed step-by-step explanation of how a hash map works, "
    "including collision handling, resizing, and time complexity. Be thorough."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--drafter", type=Path, required=True)
    parser.add_argument("--drafter-revision", required=True)
    parser.add_argument("--drafter-bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--drafts", type=int, choices=range(1, 8), default=2)
    parser.add_argument("--first", choices=("f", "m"), default="f")
    parser.add_argument("--explore", type=int, default=2)
    parser.add_argument("--probe-every", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--recheck", type=int, default=3)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--expected-token-sha256", default="")
    args = parser.parse_args()
    if args.max_new_tokens < 2 or args.repeats < 3:
        parser.error("max-new-tokens >=2 and repeats >=3 are required")

    model_dir = _verify_snapshot_path(args.model, args.model_revision)
    drafter_dir = _verify_snapshot_path(args.drafter, args.drafter_revision)
    from tensorfold.families.qwen3_5.dflash_head import DraftSlot

    original_get = DraftSlot.get

    def retained_policy(slot: DraftSlot, sampling: Any) -> Any:
        proposer = original_get(slot, sampling)
        proposer.tree_edge = 1.4
        proposer.tree_tau = 1.5
        proposer.async_layers = (0, 1, 2, 3, 4)
        return proposer

    DraftSlot.get = retained_policy
    try:
        runtime, tokenizer = load_adaptive_runtime(
            model_dir,
            drafter_dir,
            drafts=args.drafts,
            drafter_bits=args.drafter_bits,
            first=args.first,
            explore=args.explore,
            every=args.probe_every,
            margin=args.margin,
            window=args.window,
            recheck=args.recheck,
        )
        prompt_ids = render_user_prompt(tokenizer, args.prompt)
        run_decode(
            runtime,
            tokenizer,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            stream_id="draft-selector-warmup",
            copy_drafts=False,
            force_length=True,
        )
        runs = []
        for repeat in range(args.repeats):
            runtime.reset_selection()
            run = run_decode(
                runtime,
                tokenizer,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                stream_id=f"draft-selector-{repeat}",
                copy_drafts=False,
                force_length=True,
            )
            expected = args.expected_token_sha256
            if expected and run["token_sha256"] != expected:
                raise RuntimeError(
                    "adaptive selector changed the frozen greedy output "
                    f"({run['token_sha256']} != {expected})"
                )
            records = list(runtime.selection_records)
            arms = "".join(record["arm"] for record in records)
            counts = Counter(arms)
            by_arm = {
                arm: {
                    "rounds": counts[arm],
                    "committed": sum(record["committed"] for record in records if record["arm"] == arm),
                    "milliseconds": sum(record["milliseconds"] for record in records if record["arm"] == arm),
                }
                for arm in ("f", "m")
            }
            record = {
                "repeat": repeat,
                "token_sha256": run["token_sha256"],
                "total_seconds": run["total_seconds"],
                "decode_tokens_per_second": run["decode_tokens_per_second"],
                "decode_rounds": run["decode_rounds"],
                "accepted_draft_tokens": run["accepted_draft_tokens"],
                "arms": arms,
                "by_arm": by_arm,
            }
            runs.append(record)
            print("SELECTOR " + json.dumps(record, sort_keys=True), flush=True)
    finally:
        DraftSlot.get = original_get

    identities = {str(run["token_sha256"]) for run in runs}
    if len(identities) != 1:
        raise RuntimeError("adaptive selector runs returned different greedy outputs")
    print(json.dumps({
        "event": "glm_mac_adaptive_draft_selector",
        "model_revision": args.model_revision,
        "drafter_revision": args.drafter_revision,
        "drafter_bits": args.drafter_bits,
        "drafts": args.drafts,
        "first": args.first,
        "explore": args.explore,
        "probe_every": args.probe_every,
        "margin": args.margin,
        "window": args.window,
        "recheck": args.recheck,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "token_sha256": next(iter(identities)),
        "median_total_seconds": statistics.median(run["total_seconds"] for run in runs),
        "median_decode_tokens_per_second": statistics.median(
            run["decode_tokens_per_second"] for run in runs
        ),
        "runs": runs,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
