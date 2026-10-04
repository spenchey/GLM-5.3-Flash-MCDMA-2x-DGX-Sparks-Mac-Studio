#!/usr/bin/env python3
"""Measure the upper bound from verifying a Spark-drafted opening block.

This does not claim a deployable distributed path.  It deliberately gives the
Mac the known-correct greedy opening tokens, verifies them in one target-model
call, and then requires the remaining decode to reproduce the reference token
stream exactly.  If this optimistic bound cannot clear the acceptance target,
building a real Spark producer cannot clear it either.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.three_machine.cache_handoff import prompt_digest
from experiments.three_machine.mac_decode import (
    _first_token_difference,
    _load_runtime,
    _reclaim_mlx_memory,
    _verify_snapshot_path,
    build_local_full_prompt_continuation_cache,
    render_user_prompt,
    run_decode,
)


DEFAULT_PROMPT = (
    "Write a detailed step-by-step explanation of how a hash map works, "
    "including collision handling, resizing, and time complexity. Be thorough."
)


def _positive_ints(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from exc
    if not values or any(number < 1 for number in values):
        raise argparse.ArgumentTypeError("values must be positive")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must not repeat")
    return values


def _balanced(values: Sequence[int], repeats: int) -> list[list[int]]:
    out: list[list[int]] = []
    for index in range(repeats):
        order = list(values if index % 2 == 0 else reversed(values))
        shift = (index // 2) % len(order)
        out.append(order[shift:] + order[:shift])
    return out


def _median(runs: Sequence[dict[str, Any]], field: str) -> float | None:
    values = [float(run[field]) for run in runs if field in run]
    return statistics.median(values) if values else None


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--drafter", type=Path, required=True)
    parser.add_argument("--drafter-revision", required=True)
    parser.add_argument("--drafter-bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--blocks", type=_positive_ints, default=_positive_ints("3,8,16,24,32,48,64"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prefill-rows", type=int, default=128)
    parser.add_argument("--edge", type=float, default=1.4)
    parser.add_argument("--tau", type=float, default=1.5)
    args = parser.parse_args()
    if args.max_new_tokens < 3 or args.repeats < 3 or args.prefill_rows < 1:
        parser.error("max-new-tokens >=3, repeats >=3, and prefill-rows >=1 are required")
    if args.edge < 0 or args.tau <= 0:
        parser.error("edge must be nonnegative and tau must be positive")
    if any(block >= args.max_new_tokens - 1 for block in args.blocks):
        parser.error("each block must leave at least one continuation token")

    model_dir = _verify_snapshot_path(args.model, args.model_revision)
    drafter_dir = _verify_snapshot_path(args.drafter, args.drafter_revision)

    from tensorfold.families.qwen3_5.dflash_head import DraftSlot

    original_get = DraftSlot.get

    def tuned_get(slot: DraftSlot, sampling: Any) -> Any:
        proposer = original_get(slot, sampling)
        proposer.tree_edge = float(args.edge)
        proposer.tree_tau = float(args.tau)
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
        reference = run_decode(
            runtime,
            tokenizer,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            stream_id="oracle-block-reference",
            copy_drafts=False,
            force_length=True,
        )
        reference_tokens = [int(token) for token in reference["tokens"]]
        reference_hash = str(reference["token_sha256"])
        runs: dict[int, list[dict[str, Any]]] = {block: [] for block in args.blocks}

        import mlx.core as mx
        from tensorfold.engine.family_common import cache_arrays

        for repeat, order in enumerate(_balanced(args.blocks, args.repeats)):
            for block in order:
                cache = build_local_full_prompt_continuation_cache(
                    runtime,
                    prompt_ids,
                    reference_tokens[0],
                    prefill_rows=args.prefill_rows,
                    mx_module=mx,
                )
                started = time.perf_counter()
                verify_inputs = mx.array(
                    [reference_tokens[:block]], dtype=mx.uint32
                )
                hidden = runtime.hidden(verify_inputs, cache)
                logits = runtime.head(hidden)
                predicted = mx.argmax(logits, axis=-1).reshape(-1)
                mx.eval(predicted)
                predicted_tokens = [int(token) for token in predicted.tolist()]
                expected_next = reference_tokens[1:block + 1]
                verify_seconds = time.perf_counter() - started
                verified = predicted_tokens == expected_next
                record: dict[str, Any] = {
                    "repeat": repeat,
                    "block": block,
                    "verified": verified,
                    "verify_seconds": verify_seconds,
                    "verify_first_difference": _first_token_difference(
                        expected_next, predicted_tokens
                    ),
                }
                if verified:
                    runtime.absorb_draft_context(
                        runtime.draft_rows(), expected_next, cache
                    )
                    mx.eval(*cache_arrays(cache))
                    verified_at = time.perf_counter()
                    pending = reference_tokens[block]
                    continued = run_decode(
                        runtime,
                        tokenizer,
                        [*prompt_ids, *reference_tokens[:block], pending],
                        max_new_tokens=args.max_new_tokens - block - 1,
                        cache=cache,
                        cached_tokens=len(prompt_ids) + block,
                        stream_id=f"oracle-block-{block}-run-{repeat}",
                        copy_drafts=False,
                        force_length=True,
                    )
                    combined = [
                        *reference_tokens[:block + 1],
                        *[int(token) for token in continued["tokens"]],
                    ]
                    finished = time.perf_counter()
                    record.update({
                        "prime_seconds": verified_at - started,
                        "continuation_seconds": finished - verified_at,
                        "verify_and_continue_seconds": finished - started,
                        "exact_output": combined == reference_tokens,
                        "output_first_difference": _first_token_difference(
                            reference_tokens, combined
                        ),
                        "token_sha256": prompt_digest(combined),
                        "text_sha256": _digest_text(
                            tokenizer.decode(combined, skip_special_tokens=False)
                        ),
                        "continued": continued,
                    })
                else:
                    cache.clear()
                    _reclaim_mlx_memory()
                runs[block].append(record)
                print(
                    f"BLOCK {block} ROUND {repeat} VERIFIED {int(verified)} "
                    f"EXACT {int(bool(record.get('exact_output')))} "
                    f"VERIFY {verify_seconds:.6f} "
                    f"TOTAL {float(record.get('verify_and_continue_seconds', 0.0)):.6f}",
                    flush=True,
                )
    finally:
        DraftSlot.get = original_get

    summary = []
    for block in args.blocks:
        values = runs[block]
        valid = [run for run in values if run.get("verified") and run.get("exact_output")]
        summary.append({
            "block": block,
            "valid_runs": len(valid),
            "repeat_count": len(values),
            "median_verify_seconds": _median(values, "verify_seconds"),
            "median_prime_seconds": _median(valid, "prime_seconds"),
            "median_continuation_seconds": _median(valid, "continuation_seconds"),
            "median_verify_and_continue_seconds": _median(
                valid, "verify_and_continue_seconds"
            ),
            "runs": values,
        })
    print(json.dumps({
        "event": "glm_mac_oracle_handoff_block_upper_bound",
        "qualification": (
            "optimistic Mac-side upper bound; excludes Spark draft production, "
            "MCDMA transfer, and first-token verification"
        ),
        "model_revision": args.model_revision,
        "drafter_revision": args.drafter_revision,
        "drafter_bits": args.drafter_bits,
        "edge": args.edge,
        "tau": args.tau,
        "async_layers": [0, 1, 2, 3, 4],
        "prompt_tokens": len(prompt_ids),
        "prompt_token_sha256": prompt_digest(prompt_ids),
        "completion_tokens": args.max_new_tokens,
        "reference_token_sha256": reference_hash,
        "reference_total_seconds": reference["total_seconds"],
        "reference_decode_tokens_per_second": reference["decode_tokens_per_second"],
        "repeats": args.repeats,
        "summary": summary,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
