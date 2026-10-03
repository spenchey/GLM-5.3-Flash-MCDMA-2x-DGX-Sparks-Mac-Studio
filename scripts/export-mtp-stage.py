#!/usr/bin/env python3
"""Stream the GLM MTP head and shared token matrices into a small checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path


OUTPUT = "model-mtp-00001-of-00001.safetensors"
COPY_FILES = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "exl3-mcg-storage-abi.json",
    "generation_config.json",
    "processor_config.json",
    "quantization_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"short safetensors header: {path}")
        size = struct.unpack("<Q", raw)[0]
        header = json.loads(handle.read(size))
    return 8 + size, header


def selected_names(weight_map: dict[str, str], mtp_layer: int) -> list[str]:
    prefix = f"model.language_model.layers.{mtp_layer}."
    shared = ("model.language_model.embed_tokens", "lm_head")
    names = sorted(
        name for name in weight_map
        if name.startswith(prefix) or any(
            name == f"{base}.{part}"
            for base in shared
            for part in ("weight", "scales", "biases")
        )
    )
    if not all(f"{base}.weight" in names for base in shared):
        raise ValueError("checkpoint lacks the shared embedding or language-model head")
    if not any(name.startswith(prefix) for name in names):
        raise ValueError(f"checkpoint lacks MTP layer {mtp_layer}")
    return names


def export(source: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config") or config
    mtp_layer = int(text["num_hidden_layers"])
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    names = selected_names(weight_map, mtp_layer)

    headers: dict[str, tuple[int, dict]] = {}
    tensors: dict[str, dict] = {}
    offset = 0
    for name in names:
        shard = weight_map[name]
        if shard not in headers:
            headers[shard] = read_header(source / shard)
        _data_start, header = headers[shard]
        info = header.get(name)
        if not isinstance(info, dict):
            raise ValueError(f"index/header mismatch for {name}")
        begin, end = map(int, info["data_offsets"])
        if begin < 0 or end <= begin:
            raise ValueError(f"invalid tensor offsets for {name}")
        size = end - begin
        tensors[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, offset + size],
            "source_shard": shard,
            "source_offsets": [begin, end],
        }
        offset += size

    destination.mkdir(parents=True)
    temporary = destination / f".{OUTPUT}.partial-{os.getpid()}"
    output = destination / OUTPUT
    public_header = {"__metadata__": {"format": "pt"}}
    public_header.update({
        name: {key: value for key, value in info.items() if key in ("dtype", "shape", "data_offsets")}
        for name, info in tensors.items()
    })
    encoded = json.dumps(public_header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    try:
        with temporary.open("xb") as target:
            target.write(struct.pack("<Q", len(encoded)))
            target.write(encoded)
            for name in names:
                info = tensors[name]
                data_start, _header = headers[info["source_shard"]]
                begin, end = info["source_offsets"]
                remaining = end - begin
                with (source / info["source_shard"]).open("rb") as origin:
                    origin.seek(data_start + begin)
                    while remaining:
                        chunk = origin.read(min(8 << 20, remaining))
                        if not chunk:
                            raise ValueError(f"short tensor body for {name}")
                        target.write(chunk)
                        remaining -= len(chunk)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    for name in COPY_FILES:
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
    out_index = {
        "metadata": {"total_size": offset},
        "weight_map": {name: OUTPUT for name in names},
    }
    (destination / "model.safetensors.index.json").write_text(
        json.dumps(out_index, indent=2, sort_keys=True) + "\n"
    )
    files = {}
    for path in sorted(p for p in destination.iterdir() if p.is_file()):
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "schema": 1,
        "purpose": "GLM-5.3-Flash MTP drafting stage for the Mac Studio",
        "source": str(source.resolve()),
        "source_index_sha256": sha256(index_path),
        "mtp_layer": mtp_layer,
        "tensor_count": len(names),
        "tensor_bytes": offset,
        "files": files,
    }
    (destination / "mtp-stage-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = export(args.source.resolve(), args.destination.expanduser().absolute())
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
