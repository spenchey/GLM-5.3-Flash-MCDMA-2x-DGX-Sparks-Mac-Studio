"""Fail-closed identity checks for the already-present P07A checkpoint."""
import hashlib
import json
import math
import struct
from pathlib import Path

REVISION = "76add2a341a1cd90ad0e86bb69839ea9c35827c6"
HASHES = {
    "config.json": "f80682bcc8ba5fc6ecdc5ce553700b02befe999bc4a6ba6fd47f644df4864adf",
    "model.safetensors.index.json": "6c030d5b58f4c264f00becf9dddcb8f8b1f69b3bbaaf35c93592b201b673208f",
}


def expected():
    result = {}
    def tensor(name, dtype, shape):
        result[name] = {"dtype": dtype, "shape": shape}
    def quant(name, out, width):
        tensor(name + ".weight", "U32", [out, width // 8])
        for part in ("scales", "biases"):
            tensor(name + "." + part, "BF16", [out, width // 64])
    quant("embed_tokens", 154880, 4096)
    for branch in ("attn", "ffn"):
        tensor(f"layers.0.hc_{branch}_base", "F32", [24])
        tensor(f"layers.0.hc_{branch}_fn", "BF16", [24, 16384])
        tensor(f"layers.0.hc_{branch}_scale", "F32", [3])
    for norm in ("input_layernorm", "post_attention_layernorm"):
        tensor(f"layers.0.{norm}.weight", "BF16", [4096])
    for name, out, width in (
        ("q_proj", 8192, 4096), ("k_proj", 8192, 4096), ("v_proj", 8192, 4096),
        ("f_a_proj", 128, 4096), ("g_a_proj", 128, 4096), ("b_proj", 64, 4096),
        ("f_b_proj", 8192, 128), ("g_b_proj", 8192, 128), ("o_proj", 4096, 8192),
    ):
        quant("layers.0.self_attn." + name, out, width)
    for name in "qkv":
        tensor(f"layers.0.self_attn.{name}_conv1d.weight", "BF16", [8192, 1, 4])
    tensor("layers.0.self_attn.A_log", "F32", [64])
    tensor("layers.0.self_attn.dt_bias", "F32", [8192])
    tensor("layers.0.self_attn.o_norm.weight", "BF16", [128])
    quant("layers.0.mlp.gate_proj", 12288, 4096)
    quant("layers.0.mlp.up_proj", 12288, 4096)
    quant("layers.0.mlp.down_proj", 4096, 12288)
    assert len(result) == 53
    return result


def hash_payload(handle, start, size):
    """Hash one stored tensor range without reading adjacent tensor bodies."""
    assert start >= 0 and size >= 0, "invalid tensor byte range"
    handle.seek(start)
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        data = handle.read(min(remaining, 1024 * 1024))
        if not data:
            raise ValueError("truncated selected tensor payload")
        digest.update(data)
        remaining -= len(data)
    return digest.hexdigest()


def payload_evidence(path, ranges):
    """Ranges are short name -> (shard, absolute payload offset, byte count)."""
    hashes = {}
    for shard in sorted({item[0] for item in ranges.values()}):
        with (path / shard).open("rb") as handle:
            for name in sorted(ranges):
                selected_shard, start, size = ranges[name]
                if selected_shard == shard:
                    hashes[name] = hash_payload(handle, start, size)
    assert len(hashes) == 53, "expected exactly 53 selected payload hashes"
    # Canonical JSON name->hex digest is unambiguous and independent of shard order.
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": dict(sorted(hashes.items())),
            "combined_sha256": hashlib.sha256(canonical).hexdigest(),
            "stored_bytes_hashed": sum(item[2] for item in ranges.values())}


def verify(path: Path):
    assert path.name == REVISION, "wrong checkpoint revision directory"
    for name, digest in HASHES.items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest, name
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model."
    selected = {n: shard for n, shard in weight_map.items()
                if n.startswith((prefix + "embed_tokens.", prefix + "layers.0."))}
    found = {}
    ranges = {}
    for shard in sorted(set(selected.values())):
        with (path / shard).open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        for name in selected:
            if selected[name] == shard:
                item = header[name]
                short = name.removeprefix(prefix)
                found[short] = {key: item[key] for key in ("dtype", "shape")}
                start, end = item["data_offsets"]
                width = {"BF16": 2, "F32": 4, "U32": 4}[item["dtype"]]
                assert 0 <= start <= end and end - start == math.prod(item["shape"]) * width, short
                assert 8 + size + end <= (path / shard).stat().st_size, short
                ranges[short] = (shard, 8 + size + start, end - start)
    assert found == expected(), "53-tensor name/dtype/shape mismatch"
    payloads = payload_evidence(path, ranges)
    assert set(payloads["sha256"]) == set(found)
    return {"hashes": HASHES, "tensors": dict(sorted(found.items())), "revision": REVISION,
            "payloads": payloads}
