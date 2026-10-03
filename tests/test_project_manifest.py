import importlib.util
from pathlib import Path


def load_module(root: Path):
    path = root / "scripts" / "project-manifest.py"
    spec = importlib.util.spec_from_file_location("project_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manifest_is_deterministic_and_ignores_private_runtime_files(tmp_path):
    root = Path(__file__).parents[1]
    module = load_module(root)
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts" / "run.sh").write_text("echo ready\n")
    (project / "README.md").write_text("operator notes one\n")
    (project / "config.local.env").write_text("PRIVATE=one\n")
    (project / "results" / "raw").mkdir(parents=True)
    (project / "results" / "raw" / "request.log").write_text("private\n")
    first = module.manifest(project)
    (project / "config.local.env").write_text("PRIVATE=two\n")
    (project / ".deployment-sha256").write_text("old\n")
    (project / "README.md").write_text("operator notes two\n")
    assert module.manifest(project) == first
    (project / "scripts" / "run.sh").write_text("echo changed\n")
    assert module.manifest(project) != first
