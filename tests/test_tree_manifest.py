from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "tree-manifest.py"
SPEC = importlib.util.spec_from_file_location("tree_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_manifest_changes_with_content_but_metadata_manifest_does_not(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    artifact = root / "artifact.bin"
    artifact.write_bytes(b"first")
    content_before = MODULE.manifest(root)
    metadata_before = MODULE.manifest(root, metadata_only=True)

    artifact.write_bytes(b"other")

    assert MODULE.manifest(root) != content_before
    assert MODULE.manifest(root, metadata_only=True) == metadata_before


def test_manifest_pins_symlink_target_and_resolved_bytes(tmp_path: Path) -> None:
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    first = blobs / "first"
    second = blobs / "second"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    root = tmp_path / "snapshot"
    root.mkdir()
    link = root / "weights"
    link.symlink_to(first)
    before = MODULE.manifest(root)

    link.unlink()
    link.symlink_to(second)

    assert MODULE.manifest(root) != before


def test_manifest_ignores_generated_cache_files(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "module.py").write_text("value = 1\n")
    before = MODULE.manifest(root)
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"generated")

    assert MODULE.manifest(root) == before


def test_manifest_can_exclude_one_review_marker(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "source.py").write_text("value = 1\n")
    before = MODULE.manifest(root)
    (root / ".review-marker.json").write_text("changed later\n")

    assert MODULE.manifest(root) != before
    assert (
        MODULE.manifest(root, excluded_relative=frozenset({".review-marker.json"}))
        == before
    )
