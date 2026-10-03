"""Export only GLM's embedding and layer 0 for the Mac TensorFold stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(source: Path, destination: Path) -> dict:
    from tensorfold.families.glm5_next import layouts

    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config") or config
    layers = int(text["num_hidden_layers"])
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    selected = {name: shard for name, shard in index.items()
                if (layouts.canonical(name, layers) == "embed_tokens.weight"
                    or (layouts.canonical(name, layers) or "").startswith("layers.0."))}
    if not selected:
        raise RuntimeError("no embedding/layer-0 tensors matched the pinned checkpoint layout")
    by_shard: dict[str, list[str]] = {}
    for name, shard in selected.items():
        by_shard.setdefault(shard, []).append(name)
    tensors = {}
    for shard, names in by_shard.items():
        with safe_open(str(source / shard), framework="pt", device="cpu") as opened:
            for name in names:
                tensors[name] = opened.get_tensor(name)
    shard_name = "model-stage-00001-of-00001.safetensors"
    save_file(tensors, str(destination / shard_name), metadata={"format": "pt", "tensorfold_stage": "0"})
    weight_map = {name: shard_name for name in sorted(tensors)}
    (destination / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
         "weight_map": weight_map}, indent=2, sort_keys=True) + "\n")
    for item in source.iterdir():
        if (not item.is_file() or item.name.endswith(".safetensors")
                or item.name == "model.safetensors.index.json" or item.name.endswith(".lock")):
            continue
        if item.stat().st_size <= 64 << 20:
            shutil.copy2(item, destination / item.name, follow_symlinks=True)
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(destination.iterdir()) if p.is_file()}
    manifest = {"source": str(source), "layers": [0], "tensor_count": len(tensors), "files": files}
    (destination / "stage-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source.resolve(), args.destination.resolve()), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
