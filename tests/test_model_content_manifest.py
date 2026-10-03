from pathlib import Path

from scripts.model_content_manifest import manifest


def test_manifest_binds_paths_and_contents(tmp_path: Path):
    first = tmp_path / "a"
    second = tmp_path / "nested" / "b"
    second.parent.mkdir()
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    original = manifest(tmp_path)
    assert original["files"] == 2
    assert original["bytes"] == 6
    second.write_bytes(b"three")
    assert manifest(tmp_path)["sha256"] != original["sha256"]
