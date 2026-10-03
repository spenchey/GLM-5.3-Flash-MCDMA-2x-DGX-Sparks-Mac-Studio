#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
bash -n "$PROJECT_DIR/scripts/run-p07a.sh"
PYTHONDONTWRITEBYTECODE=1 python3 - "$PROJECT_DIR" <<'PY'
import ast
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "experiments/p07a"))
from identity import expected
manifest = expected()
assert len(manifest) == 53
assert all(n.startswith(("embed_tokens.", "layers.0.")) for n in manifest)
tree = ast.parse((pathlib.Path(sys.argv[1]) / "experiments/p07a/window.py").read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        assert name not in {"load_backbone", "hidden_rows", "head", "load_tokenizer", "mean", "sum"}, name
        if name == "load_layer":
            assert isinstance(node.args[1], ast.Constant) and node.args[1].value == 0
for file in (pathlib.Path(sys.argv[1]) / "experiments/p07a").glob("*.py"):
    ast.parse(file.read_text())
fixture = ast.parse((pathlib.Path(sys.argv[1]) / "experiments/p07a/fixture.py").read_text())
assert any(isinstance(n, ast.Assert) and isinstance(n.test, ast.Subscript)
           and isinstance(n.test.slice, ast.Constant) and n.test.slice.value == "offset_equal"
           for n in ast.walk(fixture)), "batch/chunk offsets must be asserted"
print("PASS: exact 53-tensor contract, layer-0-only loader, forbidden conclusion calls, offset assertion and syntax")
PY
PYTHONDONTWRITEBYTECODE=1 python3 "$PROJECT_DIR/experiments/p07a/test_identity.py"
