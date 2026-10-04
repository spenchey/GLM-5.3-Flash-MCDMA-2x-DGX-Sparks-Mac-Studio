import hashlib
import json
from pathlib import Path


def test_reviewed_stage_patches_cover_dense_weights_and_dflash():
    root = Path(__file__).parents[1]
    manifest = json.loads((root / "patches/tensorfold/metal-stage-manifest.json").read_text())
    patches = {
        record["path"]: (root / record["path"]).read_text()
        for record in manifest["patches"]
    }
    for record in manifest["patches"]:
        patch_path = root / record["path"]
        assert hashlib.sha256(patch_path.read_bytes()).hexdigest() == record["sha256"]
    assert set(manifest["files"]) == {
        "tensorfold/families/glm5_next/linear.py",
        "tensorfold/families/glm5_next/weights.py",
        "tensorfold/drafters/dflash_drafter.py",
        "tensorfold/families/glm5_next/__init__.py",
        "tensorfold/families/glm5_next/model.py",
        "tensorfold/families/glm5_next/runtime.py",
    }
    assert all(len(item["base_sha256"]) == 64 and len(item["patched_sha256"]) == 64
               for item in manifest["files"].values())
    dense = patches["patches/tensorfold/0001-metal-read-exl3-dense-stage.patch"]
    dflash = patches["patches/tensorfold/0002-glm-dflash2-metal.patch"]
    assert "all(isinstance(p, Dense)" in dense
    assert 'aw: dict[str, Any] = {n: w.linear' in dense
    assert 'mlp = DenseMLP(w.linear' in dense
    assert 'embed = w.linear("embed_tokens")' in dense
    assert "class GLMDFlash(GLMFlash)" in dflash
    assert "drafter=drafter, drafter_bits=drafter_bits" in dflash
    assert "DFlash2 block" in dflash
