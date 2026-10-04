#!/usr/bin/env python3
"""Measure where GLM's built-in MTP head predicts a DFlash2 miss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from experiments.three_machine.mac_decode import (
    _verify_snapshot_path,
    render_user_prompt,
    run_decode,
)
from experiments.three_machine.mac_draft_shadow import load_shadow_runtime


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
        runtime, tokenizer = load_shadow_runtime(
            model_dir,
            drafter_dir,
            drafts=args.drafts,
            drafter_bits=args.drafter_bits,
        )
        prompt_ids = render_user_prompt(tokenizer, args.prompt)
        run_decode(
            runtime,
            tokenizer,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            stream_id="draft-shadow-warmup",
            copy_drafts=False,
            force_length=True,
        )
        runs = []
        for repeat in range(args.repeats):
            runtime.reset_shadow()
            run = run_decode(
                runtime,
                tokenizer,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                stream_id=f"draft-shadow-{repeat}",
                copy_drafts=False,
                force_length=True,
            )
            expected = args.expected_token_sha256
            if expected and run["token_sha256"] != expected:
                raise RuntimeError(
                    "draft shadow changed the frozen greedy output "
                    f"({run['token_sha256']} != {expected})"
                )
            record = {
                "repeat": repeat,
                "token_sha256": run["token_sha256"],
                "total_seconds": run["total_seconds"],
                "decode_tokens_per_second": run["decode_tokens_per_second"],
                "decode_rounds": run["decode_rounds"],
                "accepted_draft_tokens": run["accepted_draft_tokens"],
                "shadow": dict(runtime.shadow_totals),
            }
            runs.append(record)
            print("SHADOW " + json.dumps(record, sort_keys=True), flush=True)
    finally:
        DraftSlot.get = original_get

    identities = {str(run["token_sha256"]) for run in runs}
    if len(identities) != 1:
        raise RuntimeError("draft shadow runs returned different greedy outputs")
    fields = ("observations", "both_hit", "mtp_only", "dflash_only", "neither", "mtp_predictions")
    medians = {
        field: statistics.median(int(run["shadow"][field]) for run in runs)
        for field in fields
    }
    print(json.dumps({
        "event": "glm_mac_draft_complementarity",
        "model_revision": args.model_revision,
        "drafter_revision": args.drafter_revision,
        "drafter_bits": args.drafter_bits,
        "drafts": args.drafts,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "token_sha256": next(iter(identities)),
        "median_shadow": medians,
        "runs": runs,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
