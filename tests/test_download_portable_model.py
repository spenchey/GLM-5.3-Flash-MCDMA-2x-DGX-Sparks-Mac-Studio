from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "scripts" / "download-portable-model.py"
SPEC = importlib.util.spec_from_file_location("download_portable_model", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selected_file_sizes_honors_patterns() -> None:
    info = SimpleNamespace(siblings=[
        SimpleNamespace(rfilename="config.json", size=10),
        SimpleNamespace(rfilename="model-00001-of-00043.safetensors", size=20),
        SimpleNamespace(rfilename="zai-logo.png", size=30),
        SimpleNamespace(rfilename="notes.txt", size=40),
    ])

    assert MODULE.selected_file_sizes(info, ["*.json", "*.safetensors", "*.png"]) == {
        "config.json": 10,
        "model-00001-of-00043.safetensors": 20,
        "zai-logo.png": 30,
    }


def test_repair_incomplete_files_redownloads_only_invalid_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "complete.bin").write_bytes(b"abc")
    (tmp_path / "truncated.bin").write_bytes(b"x")
    expected = {"complete.bin": 3, "truncated.bin": 4, "missing.bin": 5}
    calls: list[str] = []

    def fake_download(*, filename: str, local_dir: Path, **_: object) -> None:
        calls.append(filename)
        (Path(local_dir) / filename).write_bytes(b"z" * expected[filename])

    monkeypatch.setattr(MODULE, "hf_hub_download", fake_download)
    MODULE.repair_incomplete_files(
        repo="example/model",
        revision="a" * 40,
        output=tmp_path,
        expected=expected,
    )

    assert calls == ["truncated.bin", "missing.bin"]
    assert {path.name: path.stat().st_size for path in tmp_path.iterdir()} == expected
