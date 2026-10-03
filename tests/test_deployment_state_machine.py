from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "deployment-state-machine.sh"


def seed(path: Path, manifest: str, deploy_id: str, payload: str) -> None:
    path.mkdir()
    (path / ".deployment-sha256").write_text(f"{manifest}\n")
    (path / ".deployment-id").write_text(f"{deploy_id}\n")
    (path / "payload").write_text(payload)


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        check=check,
        text=True,
        capture_output=True,
    )


def test_first_install_and_exact_rollback_to_absent(tmp_path: Path) -> None:
    destination = tmp_path / "runtime"
    incoming = tmp_path / "runtime.incoming.deploy-new"
    seed(incoming, "manifest-new", "deploy-new", "new")

    run("install", str(destination), str(incoming), "manifest-new", "deploy-new")
    assert (destination / "payload").read_text() == "new"
    assert not incoming.exists()

    run("rollback", str(destination), "manifest-new", "deploy-new")
    assert not destination.exists()


def test_upgrade_retains_previous_and_exact_rollback_restores_it(tmp_path: Path) -> None:
    destination = tmp_path / "runtime"
    previous = tmp_path / "runtime.previous"
    incoming = tmp_path / "runtime.incoming.deploy-new"
    seed(destination, "manifest-old", "deploy-old", "old")
    seed(incoming, "manifest-new", "deploy-new", "new")

    run("install", str(destination), str(incoming), "manifest-new", "deploy-new")
    assert (destination / "payload").read_text() == "new"
    assert (previous / "payload").read_text() == "old"

    run("rollback", str(destination), "manifest-new", "deploy-new")
    assert (destination / "payload").read_text() == "old"
    assert not previous.exists()


def test_rollback_nonce_cannot_replace_an_unrelated_deployment(tmp_path: Path) -> None:
    destination = tmp_path / "runtime"
    previous = tmp_path / "runtime.previous"
    seed(destination, "manifest-current", "deploy-current", "current")
    seed(previous, "manifest-old", "deploy-old", "old")

    run("rollback", str(destination), "manifest-current", "another-deploy")

    assert (destination / "payload").read_text() == "current"
    assert (previous / "payload").read_text() == "old"


def test_missing_destination_restores_a_marked_previous_tree(tmp_path: Path) -> None:
    destination = tmp_path / "runtime"
    previous = tmp_path / "runtime.previous"
    seed(previous, "manifest-old", "deploy-old", "old")

    run("rollback", str(destination), "manifest-new", "deploy-new")

    assert (destination / "payload").read_text() == "old"
    assert not previous.exists()


def test_install_recovers_an_interrupted_swap_before_replacing_it(tmp_path: Path) -> None:
    destination = tmp_path / "runtime"
    previous = tmp_path / "runtime.previous"
    incoming = tmp_path / "runtime.incoming.deploy-new"
    seed(previous, "manifest-old", "deploy-old", "old")
    seed(incoming, "manifest-new", "deploy-new", "new")

    run("install", str(destination), str(incoming), "manifest-new", "deploy-new")

    assert (destination / "payload").read_text() == "new"
    assert (previous / "payload").read_text() == "old"
    assert not incoming.exists()


def test_install_refuses_unmarked_or_git_destination(tmp_path: Path) -> None:
    for suffix in ("unmarked", "git"):
        destination = tmp_path / f"runtime-{suffix}"
        incoming = tmp_path / f"runtime-{suffix}.incoming.deploy-new"
        destination.mkdir()
        (destination / "payload").write_text("keep")
        if suffix == "git":
            (destination / ".git").mkdir()
            (destination / ".deployment-sha256").write_text("manifest-old\n")
        seed(incoming, "manifest-new", "deploy-new", "new")

        result = run(
            "install",
            str(destination),
            str(incoming),
            "manifest-new",
            "deploy-new",
            check=False,
        )

        assert result.returncode != 0
        assert (destination / "payload").read_text() == "keep"


def test_rollback_retry_finishes_after_target_was_moved_aside(tmp_path: Path) -> None:
    destination = tmp_path / "runtime"
    previous = tmp_path / "runtime.previous"
    failed = tmp_path / "runtime.failed.deploy-new"
    seed(previous, "manifest-old", "deploy-old", "old")
    seed(failed, "manifest-new", "deploy-new", "new")

    run("rollback", str(destination), "manifest-new", "deploy-new")

    assert (destination / "payload").read_text() == "old"
    assert not previous.exists()
    assert not failed.exists()


def test_rollback_retry_removes_scratch_after_previous_was_restored(tmp_path: Path) -> None:
    destination = tmp_path / "runtime"
    failed = tmp_path / "runtime.failed.deploy-new"
    seed(destination, "manifest-old", "deploy-old", "old")
    seed(failed, "manifest-new", "deploy-new", "new")

    run("rollback", str(destination), "manifest-new", "deploy-new")

    assert (destination / "payload").read_text() == "old"
    assert not failed.exists()
