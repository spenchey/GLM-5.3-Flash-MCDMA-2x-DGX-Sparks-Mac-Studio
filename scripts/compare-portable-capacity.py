#!/usr/bin/env python3
"""Fail unless the full three-machine pipeline beats the same two-Spark model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(native: dict, hybrid: dict, minimum_gain: float) -> dict:
    if native.get("kind") != "portable-native-two-spark-concurrency":
        raise ValueError("native evidence has the wrong kind")
    if hybrid.get("event") != "glm_mcdma_benchmark":
        raise ValueError("hybrid evidence has the wrong kind")
    if native.get("parallel") != hybrid.get("parallel"):
        raise ValueError("native and hybrid concurrency differ")
    if native.get("max_tokens") != hybrid.get("max_new_tokens"):
        raise ValueError("native and hybrid reply lengths differ")
    local = hybrid.get("local_reference") or {}
    if native.get("token_sha256") != local.get("token_sha256"):
        raise ValueError("native and hybrid output tokens differ")
    rounds = list(hybrid.get("warmups") or []) + list(hybrid.get("runs") or [])
    if not rounds or any(item.get("exact_token_match") is not True for item in rounds):
        raise ValueError("hybrid evidence does not prove exact output in every round")
    native_rate = float(native["aggregate_pipeline_tokens_per_second"]["median"])
    hybrid_rate = float(hybrid["aggregate_pipeline_tokens_per_second"]["median"])
    if native_rate <= 0 or hybrid_rate <= 0:
        raise ValueError("benchmark rates must be positive")
    gain = hybrid_rate / native_rate - 1.0
    result = {
        "schema": 1,
        "kind": "portable-serving-capacity-comparison",
        "parallel": native["parallel"],
        "max_tokens": native["max_tokens"],
        "token_sha256": native["token_sha256"],
        "native_two_spark_tokens_per_second": native_rate,
        "three_machine_tokens_per_second": hybrid_rate,
        "gain_fraction": gain,
        "minimum_gain_fraction": minimum_gain,
        "pass": gain >= minimum_gain,
    }
    if not result["pass"]:
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--minimum-gain", type=float, default=0.03)
    args = parser.parse_args()
    if args.minimum_gain < 0:
        parser.error("--minimum-gain must be nonnegative")
    result = compare(
        json.loads(args.native.read_text()),
        json.loads(args.hybrid.read_text()),
        args.minimum_gain,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
