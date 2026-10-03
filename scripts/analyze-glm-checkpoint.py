#!/usr/bin/env python3
"""Read-only safetensors inventory for a pinned GLM checkpoint.

The helper intentionally reads only the JSON headers.  It does not import a
model runtime, mmap tensor bodies, or mutate the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import Counter
from pathlib import Path


LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


def read_header(path: Path) -> dict:
    with path.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"short safetensors header: {path}")
        size = struct.unpack("<Q", size_bytes)[0]
        return json.loads(handle.read(size))


def inventory(checkpoint: Path) -> dict[str, dict]:
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    tensors: dict[str, dict] = {}
    for shard_name in sorted(set(weight_map.values())):
        header = read_header(checkpoint / shard_name)
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"duplicate tensor: {name}")
            if weight_map.get(name) != shard_name:
                raise ValueError(f"index/header mismatch: {name}")
            offsets = metadata["data_offsets"]
            tensors[name] = {
                "dtype": metadata["dtype"],
                "shape": metadata["shape"],
                "bytes": offsets[1] - offsets[0],
                "shard": shard_name,
            }
    missing = sorted(set(weight_map) - set(tensors))
    unexpected = sorted(set(tensors) - set(weight_map))
    if missing or unexpected:
        raise ValueError(
            f"index/header mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--names", action="store_true")
    args = parser.parse_args()

    tensors = inventory(args.checkpoint)
    selected = tensors
    if args.layer is not None:
        prefix = f"model.language_model.layers.{args.layer}."
        selected = {name: data for name, data in tensors.items() if name.startswith(prefix)}

    layer_counts: Counter[int] = Counter()
    layer_bytes: Counter[int] = Counter()
    for name, data in tensors.items():
        match = LAYER_RE.match(name)
        if match:
            layer = int(match.group(1))
            layer_counts[layer] += 1
            layer_bytes[layer] += data["bytes"]

    result = {
        "checkpoint": str(args.checkpoint),
        "tensor_count": len(tensors),
        "dtype_counts": dict(sorted(Counter(d["dtype"] for d in tensors.values()).items())),
        "layer_tensor_counts": dict(sorted(layer_counts.items())),
        "layer_tensor_bytes": dict(sorted(layer_bytes.items())),
        "selected_tensor_count": len(selected),
        "selected_tensor_bytes": sum(data["bytes"] for data in selected.values()),
        "selected_shards": sorted({data["shard"] for data in selected.values()}),
    }
    if args.names:
        result["tensors"] = dict(sorted(selected.items()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
