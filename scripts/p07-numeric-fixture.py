#!/usr/bin/env python3
"""Deterministic, isolated P07 weight/projection fixture.

This is analysis code, not TensorFold runtime code.  It compares one affine
MLX matrix with the corresponding BF16 matrix and can measure the resident
MLX tensors for one layer.  It never writes model files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import struct
import time
from pathlib import Path

import numpy as np


def header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return size, json.loads(handle.read(size))


def weight_map(checkpoint: Path) -> dict[str, str]:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    return index["weight_map"]


def bf16_tensor(checkpoint: Path, name: str) -> np.ndarray:
    mapping = weight_map(checkpoint)
    shard = checkpoint / mapping[name]
    header_size, entries = header(shard)
    entry = entries[name]
    if entry["dtype"] != "BF16":
        raise ValueError(f"{name} is {entry['dtype']}, not BF16")
    start, end = entry["data_offsets"]
    with shard.open("rb") as handle:
        handle.seek(8 + header_size + start)
        raw = handle.read(end - start)
    words = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return words.view(np.float32).reshape(entry["shape"])


def fixture(array: np.ndarray) -> dict:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    indices = np.array([(i * 104729 + 17) % flat.size for i in range(512)])
    samples = flat[indices]
    x = (((np.arange(array.shape[1], dtype=np.int64) * 17 + 3) % 101) - 50).astype(
        np.float32
    ) / np.float32(50.0)
    projection = x @ np.asarray(array, dtype=np.float32).T
    return {
        "float32_sha256": hashlib.sha256(flat.tobytes()).hexdigest(),
        "sample_indices": indices.tolist(),
        "sample_values": samples.tolist(),
        "projection": projection.tolist(),
        "projection_sha256": hashlib.sha256(projection.tobytes()).hexdigest(),
    }


def mlx_affine(checkpoint: Path, base: str, layer: int, include_embedding: bool) -> dict:
    import mlx.core as mx

    mapping = weight_map(checkpoint)
    names = [f"{base}.weight", f"{base}.scales", f"{base}.biases"]
    shards: dict[str, dict] = {}
    arrays = {}
    for name in names:
        shard_name = mapping[name]
        if shard_name not in shards:
            shards[shard_name] = mx.load(str(checkpoint / shard_name))
        arrays[name] = shards[shard_name][name]
    dequantized = mx.dequantize(
        arrays[names[0]], arrays[names[1]], arrays[names[2]], group_size=64, bits=4
    )
    mx.eval(dequantized)

    layer_prefix = f"model.language_model.layers.{layer}."
    layer_arrays = []
    selected_names = []
    for name, shard_name in mapping.items():
        if not name.startswith(layer_prefix) and not (
            include_embedding and name.startswith("model.language_model.embed_tokens.")
        ):
            continue
        if shard_name not in shards:
            shards[shard_name] = mx.load(str(checkpoint / shard_name))
        layer_arrays.append(shards[shard_name][name])
        selected_names.append(name)
    before = mx.get_active_memory()
    started = time.perf_counter()
    mx.eval(*layer_arrays)
    load_seconds = time.perf_counter() - started
    after = mx.get_active_memory()

    x = mx.array(
        ((((np.arange(4096, dtype=np.int64) * 17 + 3) % 101) - 50).astype(np.float32))
        / np.float32(50.0)
    )
    started = time.perf_counter()
    y = mx.quantized_matmul(
        x,
        arrays[names[0]],
        arrays[names[1]],
        arrays[names[2]],
        group_size=64,
        bits=4,
    )
    mx.eval(y)
    forward_seconds = time.perf_counter() - started
    result = fixture(np.asarray(dequantized.astype(mx.float32)))
    result.update(
        {
            "format": "mlx-affine-4bit",
            "mlx_version": mx.__version__,
            "layer": layer,
            "layer_tensor_count": len(layer_arrays),
            "includes_embedding": include_embedding,
            "selected_tensor_names": sorted(selected_names),
            "layer_active_bytes_before_eval": before,
            "layer_active_bytes_after_eval": after,
            "layer_eval_seconds": load_seconds,
            "quantized_matmul_seconds": forward_seconds,
            "quantized_matmul_shape": list(y.shape),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--format", choices=("mlx-affine", "bf16"), required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--include-embedding", action="store_true")
    args = parser.parse_args()
    if args.format == "mlx-affine":
        result = mlx_affine(
            args.checkpoint, args.base, args.layer, args.include_embedding
        )
    else:
        result = fixture(bf16_tensor(args.checkpoint, f"{args.base}.weight"))
        result["format"] = "bf16"
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
