#!/usr/bin/env python3
"""Compare normal Mac prefill with the exact Spark handoff boundary."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from experiments.three_machine.mac_decode import (
    _first_token_difference,
    _load_runtime,
    _reclaim_mlx_memory,
    _verify_snapshot_path,
    build_local_full_prompt_continuation_cache,
    build_local_handoff_cache,
    render_user_prompt,
    run_decode,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--mtp-drafts", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--prefill-rows", type=int, default=128)
    args = parser.parse_args()
    if args.max_new_tokens < 2 or args.prefill_rows < 1:
        parser.error("max-new-tokens must be at least 2 and prefill-rows must be positive")

    model_dir = _verify_snapshot_path(args.model, args.model_revision)
    with redirect_stdout(sys.stderr):
        runtime, tokenizer = _load_runtime(model_dir, mtp_drafts=args.mtp_drafts)
        prompt_ids = render_user_prompt(tokenizer, args.prompt)
        normal = run_decode(
            runtime,
            tokenizer,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            stream_id="normal-local-prefill",
            force_length=True,
        )
        mirrored_cache = build_local_handoff_cache(
            runtime,
            prompt_ids,
            prefill_rows=args.prefill_rows,
        )
        mirrored = run_decode(
            runtime,
            tokenizer,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            cache=mirrored_cache,
            cached_tokens=len(prompt_ids) - 1,
            stream_id="mirrored-handoff-prefill",
            force_length=True,
        )
        first_token = int(normal["tokens"][0])
        continuation_cache = build_local_full_prompt_continuation_cache(
            runtime,
            prompt_ids,
            first_token,
            prefill_rows=args.prefill_rows,
        )
        continued = run_decode(
            runtime,
            tokenizer,
            [*prompt_ids, first_token],
            max_new_tokens=args.max_new_tokens - 1,
            cache=continuation_cache,
            cached_tokens=len(prompt_ids),
            stream_id="full-prompt-first-token-continuation",
            force_length=True,
        )
        continued_tokens = [first_token, *continued["tokens"]]
        mirrored_cache.clear()
        continuation_cache.clear()
        _reclaim_mlx_memory()

    print(json.dumps({
        "event": "glm_local_handoff_boundary_diagnostic",
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": args.max_new_tokens,
        "mtp_drafts": args.mtp_drafts,
        "prefill_rows": args.prefill_rows,
        "exact_token_match": normal["tokens"] == mirrored["tokens"],
        "first_token_difference": _first_token_difference(
            normal["tokens"], mirrored["tokens"]
        ),
        "full_prompt_continuation": {
            "exact_token_match": normal["tokens"] == continued_tokens,
            "first_token_difference": _first_token_difference(
                normal["tokens"], continued_tokens
            ),
            "first_token": first_token,
            "continued": continued,
        },
        "normal": normal,
        "mirrored": mirrored,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
