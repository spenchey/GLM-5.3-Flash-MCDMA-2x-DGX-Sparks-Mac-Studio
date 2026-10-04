#!/usr/bin/env python3
"""Run the Mac ceiling benchmark with an isolated DFlash chain policy.

This is an experiment wrapper, not a serving-path default.  It changes only
the score used to select and budget DFlash chain nodes, then delegates to the
normal exact-output benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys
from typing import Any


def _layers(value: str) -> tuple[int, ...]:
    if value == "none":
        return ()
    try:
        layers = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("async layers must be comma-separated integers") from exc
    if any(layer < -1 for layer in layers):
        raise argparse.ArgumentTypeError("async layers must be -1 or nonnegative")
    return layers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge", type=float, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--async-layers", type=_layers, default=(0, 2))
    args, benchmark_args = parser.parse_known_args()
    if args.edge < 0:
        parser.error("edge must be nonnegative")
    if args.tau <= 0:
        parser.error("tau must be positive")

    from tensorfold.families.qwen3_5.dflash_head import DraftSlot

    original_get = DraftSlot.get

    def tuned_get(slot: DraftSlot, sampling: Any) -> Any:
        proposer = original_get(slot, sampling)
        proposer.tree_edge = float(args.edge)
        proposer.tree_tau = float(args.tau)
        proposer.async_layers = tuple(args.async_layers)
        return proposer

    DraftSlot.get = tuned_get
    benchmark = Path(__file__).with_name("benchmark-mac-single-stream-ceiling.py")
    sys.argv = [str(benchmark), *benchmark_args]
    try:
        runpy.run_path(str(benchmark), run_name="__main__")
    finally:
        DraftSlot.get = original_get
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
