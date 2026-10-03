from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "api-command-match.py"
SPEC = importlib.util.spec_from_file_location("api_command_match", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def command(*, port: str = "8899", model: str = "/models/stage") -> str:
    return (
        "/runtime/Python experiments/three_machine/api_server.py "
        f"--model {model} --mailbox box --host 127.0.0.1 --port {port} --timeout 180"
    )


def matches(value: str) -> bool:
    return MODULE.matches(value, "/runtime/Python", "/models/stage", "box", "127.0.0.1", "8899")


def test_exact_owned_command_matches() -> None:
    assert matches(command())


def test_prefix_values_and_extra_or_duplicate_options_do_not_match() -> None:
    assert not matches(command(port="88990"))
    assert not matches(command(model="/models/stage-other"))
    assert not matches(command() + " --port 8899")
    assert not matches(command() + " --extra value")


def test_malformed_shell_command_does_not_match() -> None:
    assert not matches("'/runtime/Python")
