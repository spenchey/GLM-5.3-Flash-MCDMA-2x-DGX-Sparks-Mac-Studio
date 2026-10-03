import hashlib
import json
from pathlib import Path


def test_dense_stage_patch_covers_the_required_metal_seams():
    root = Path(__file__).parents[1]
    patch_path = root / "patches/tensorfold/0001-metal-read-exl3-dense-stage.patch"
    patch = patch_path.read_text()
    manifest = json.loads((root / "patches/tensorfold/metal-stage-manifest.json").read_text())
    assert hashlib.sha256(patch_path.read_bytes()).hexdigest() == manifest["patch_sha256"]
    assert set(manifest["files"]) == {
        "tensorfold/families/glm5_next/linear.py",
        "tensorfold/families/glm5_next/weights.py",
    }
    assert all(len(item["base_sha256"]) == 64 and len(item["patched_sha256"]) == 64
               for item in manifest["files"].values())
    assert "all(isinstance(p, Dense)" in patch
    assert 'aw: dict[str, Any] = {n: w.linear' in patch
    assert 'mlp = DenseMLP(w.linear' in patch
    assert 'embed = w.linear("embed_tokens")' in patch
