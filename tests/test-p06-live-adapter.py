#!/usr/bin/env python3
"""Offline contract checks for the narrow P06 live adapter."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "scripts/tests/p06-live-adapter.py"
MAILBOX_SOURCE = ROOT / "scripts/tests/p06-live-mailbox.c"
SPEC = importlib.util.spec_from_file_location("p06_live_adapter", ADAPTER_PATH)
assert SPEC and SPEC.loader
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class Result:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeRemote:
    def __init__(self, status: str = "") -> None:
        self.status = status

    def shell(self, _script: str, **_kwargs: object) -> Result:
        return Result(stdout=self.status)


class ResidueRemote:
    """Model process/socket/listener absence checks without live hosts."""

    def __init__(self, failing_fragment: str | None = None) -> None:
        self.failing_fragment = failing_fragment
        self.calls: list[str] = []

    def shell(self, script: str, **_kwargs: object) -> Result:
        self.calls.append(script)
        results = [self._one(command.strip()) for command in script.split(";")]
        return Result(
            returncode=results[-1].returncode,
            stdout="".join(result.stdout for result in results),
            stderr="".join(result.stderr for result in results),
        )

    def _one(self, command: str) -> Result:
        fails = bool(
            self.failing_fragment and self.failing_fragment in command
        )
        if "pgrep -x mcdma-rpcd" in command:
            result = Result(returncode=0, stdout="4321\n") if fails else Result(1)
            if "|| true" in command and result.returncode:
                result.returncode = 0
            return result
        if command.startswith("test ! -e"):
            return Result(returncode=1 if fails else 0)
        if "lsof -nP" in command:
            returncode = 0 if fails else 1
            stdout = "LISTEN 127.0.0.1:18620\n" if fails else ""
            if command.startswith("!"):
                returncode = int(not returncode)
            return Result(returncode=returncode, stdout=stdout)
        if "ss -ltnH" in command:
            if command.startswith("test -z"):
                return Result(returncode=1 if fails else 0)
            return Result(
                returncode=0,
                stdout="LISTEN 127.0.0.1:18620\n" if fails else "",
            )
        return Result(returncode=1 if fails else 0)


class FakeSsh:
    """Offline executable that reproduces OpenSSH remote-command joining."""

    def __init__(self, test: unittest.TestCase) -> None:
        temporary = tempfile.TemporaryDirectory()
        test.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "fake-ssh"
        self.path.write_text(
            f"#!{sys.executable}\n"
            "import subprocess\n"
            "import sys\n"
            "command = ' '.join(sys.argv[1:])\n"
            "result = subprocess.run(['/bin/sh', '-c', command], "
            "input=sys.stdin.buffer.read(), stdout=subprocess.PIPE, "
            "stderr=subprocess.PIPE)\n"
            "sys.stdout.buffer.write(result.stdout)\n"
            "sys.stderr.buffer.write(result.stderr)\n"
            "raise SystemExit(result.returncode)\n"
        )
        self.path.chmod(0o700)

    def remote(self) -> object:
        return ADAPTER.Remote([str(self.path)])


class RemoteTransportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.remote = FakeSsh(self).remote()

    def test_entire_script_is_one_shell_payload(self) -> None:
        payload = "spaces ; $HOME * 'single quote'"
        result = self.remote.shell(
            f"umask 077; printf '%s' {ADAPTER.q(payload)}"
        )
        self.assertEqual(result.stdout, payload)
        self.assertEqual(result.stderr, "")

    def test_compile_preserves_stdin_and_owner_only_umask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "mailbox-helper"
            self.remote.compile("int main(void) { return 0; }\n", str(binary))
            self.assertTrue(binary.is_file())
            self.assertEqual(binary.stat().st_mode & 0o777, 0o700)
            result = subprocess.run([str(binary)], check=False)
        self.assertEqual(result.returncode, 0)

    def test_failure_preserves_exit_and_sanitizes_both_streams(self) -> None:
        with self.assertRaises(ADAPTER.Failure) as caught:
            self.remote.shell(
                "printf 'stdout detail'; "
                "printf 'token=do-not-expose' >&2; "
                "exit 7"
            )
        message = str(caught.exception)
        self.assertIn("exited 7", message)
        self.assertIn("stdout detail", message)
        self.assertIn("<redacted-sensitive-output>", message)
        self.assertNotIn("do-not-expose", message)

    def test_sensitive_diagnostic_forms_are_fail_closed(self) -> None:
        sensitive_values = (
            "password is abc123",
            "token failed: abc123",
            '{"token":"abc123"}',
            "Authorization: Basic abc123",
            "Bearer abc123",
            "api-key=abc123",
            "API_KEY: abc123",
            "access_token=abc123",
            "refresh_token=abc123",
            "auth_token=abc123",
            "client_secret=abc123",
            "session_cookie=abc123",
            "password_hash=abc123",
            "refresh-token=abc123",
            "auth.token=abc123",
            "clientSecret=abc123",
            "accessToken=abc123",
            "refreshToken=abc123",
            "authToken=abc123",
            "sessionCookie=abc123",
            "passwordHash=abc123",
            "ClientSecret=abc123",
            "AccessToken=abc123",
            "RefreshToken=abc123",
            "AuthToken=abc123",
            "SessionCookie=abc123",
            "PasswordHash=abc123",
            "apiToken=abc123",
            "ApiToken=abc123",
            "APIToken=abc123",
            "oauthToken=abc123",
            "OauthToken=abc123",
            "OAuthToken=abc123",
            "OAUTHToken=abc123",
            "idToken=abc123",
            "IdToken=abc123",
            "IDToken=abc123",
            "csrfToken=abc123",
            "CsrfToken=abc123",
            "CSRFToken=abc123",
            "sessionToken=abc123",
            "SessionToken=abc123",
            "clientToken=abc123",
            "ClientToken=abc123",
            "apiKey=abc123",
            "accessKey=abc123",
            "privateKey=abc123",
            "https://example.com/callback?access_token=abc123",
            "X-Auth-Token: abc123",
            '{"client_secret":"abc123"}',
        )
        for value in sensitive_values:
            with self.subTest(value=value):
                cleaned = ADAPTER.sanitized_diagnostic(value)
                self.assertEqual(cleaned, "<redacted-sensitive-output>")
                self.assertNotIn("abc123", cleaned)
        credential_url = "https://alice:abc123@example.com/path"
        cleaned_url = ADAPTER.sanitized_diagnostic(credential_url)
        self.assertEqual(cleaned_url, "https://<redacted>@example.com/path")
        self.assertNotIn("alice", cleaned_url)
        self.assertNotIn("abc123", cleaned_url)
        harmless_values = (
            "tokenization completed successfully",
            "passwordless login enabled",
            "secretary process exited",
            "cookiecutter template ready",
            "credentialed helper compiled",
            "private keyboard mapping loaded",
            "apiTokenizer ready",
            "APITokenizer ready",
            "OAuthTokenizer ready",
            "idTokenizer ready",
            "CSRFTokenizer ready",
            "sessionTokenization completed",
            "useful compiler detail on line 7",
        )
        for value in harmless_values:
            with self.subTest(value=value):
                self.assertEqual(ADAPTER.sanitized_diagnostic(value), value)

    def test_missing_socket_check_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.sock"
            result = self.remote.shell(
                f"test ! -e {ADAPTER.q(missing)}", check=False
            )
        self.assertEqual(result.returncode, 0)

    def test_active_socket_check_fails_without_removing_socket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            socket_path = Path(directory) / "active.sock"
            listener = socket.socket(socket.AF_UNIX)
            self.addCleanup(listener.close)
            listener.bind(str(socket_path))
            listener.listen(1)
            result = self.remote.shell(
                f"test ! -e {ADAPTER.q(socket_path)}", check=False
            )
            self.assertEqual(result.returncode, 1)
            self.assertTrue(socket_path.exists())


class LiveAdapterTest(unittest.TestCase):
    def make_session(self, scenario: str = "no-service"):
        temporary = tempfile.TemporaryDirectory()
        environ = {"P06_LIVE_STATE": temporary.name, "P06_LIVE_SCENARIO": scenario}
        patcher = mock.patch.dict(os.environ, environ, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(temporary.cleanup)
        return ADAPTER.Session()

    def test_exact_status_parser(self) -> None:
        session = self.make_session()
        remote = FakeRemote("p06cns up calls 7 failures 0 MiB 4 service=attached\n")
        with mock.patch.object(session, "unix", return_value=remote.status.strip()):
            parsed = session.status(remote, "/tmp/example.sock")
        self.assertEqual(parsed["link"], "up")
        self.assertEqual(parsed["calls"], 7)
        self.assertEqual(parsed["failures"], 0)
        self.assertEqual(parsed["service"], "attached")

    def test_snapshot_reports_configured_reply_transport(self) -> None:
        session = self.make_session()
        session.state["started"] = True
        status = {"link": "up", "calls": 1, "failures": 0, "service": "none"}
        with mock.patch.object(session, "alive", return_value=True), \
             mock.patch.object(session, "status", return_value=status), \
             mock.patch.object(session, "helper_json", return_value={
                 "generation": 1, "request_seq": 1, "reply_seq": 1,
             }):
            result = session.snapshot()
        self.assertEqual(
            result["reply_transport"],
            "rdma-read-pull" if ADAPTER.PULL_MODE else "rdma-write-push",
        )

    def test_versioned_multiline_status_parser(self) -> None:
        session = self.make_session()
        status = (
            "VERSION mcdma-rpcd 1 1.0.0\n"
            "PEER p06cns up calls 9 failures 0 MiB 8 "
            "device=rocep1s0f0 service=none\n"
            "END\n"
        )
        with mock.patch.object(session, "unix", return_value=status):
            parsed = session.status(FakeRemote(), "/tmp/example.sock")
        self.assertEqual(parsed["link"], "up")
        self.assertEqual(parsed["calls"], 9)
        self.assertEqual(parsed["failures"], 0)
        self.assertEqual(parsed["service"], "none")

    def test_status_parser_selects_owned_peer(self) -> None:
        session = self.make_session()
        status = (
            "PEER unrelated down calls 100 failures 10 MiB 0 service=none\n"
            "PEER p06cns up calls 2 failures 0 MiB 4 service=attached\n"
        )
        with mock.patch.object(session, "unix", return_value=status):
            parsed = session.status(FakeRemote(), "/tmp/example.sock")
        self.assertEqual(parsed["link"], "up")
        self.assertEqual(parsed["calls"], 2)
        self.assertEqual(parsed["service"], "attached")

    def test_malformed_status_fails_closed(self) -> None:
        session = self.make_session()
        with mock.patch.object(session, "unix", return_value="almost status"):
            with self.assertRaisesRegex(ADAPTER.Failure, "unexpected STATUS"):
                session.status(FakeRemote(), "/tmp/example.sock")

    def test_cleanup_order_is_connector_listener_tunnel(self) -> None:
        session = self.make_session()
        session.state.update({"started": True, "mac_pid": 111, "spark_pid": 222})
        calls: list[str] = []
        session.mac = ResidueRemote()
        session.spark = ResidueRemote()
        with mock.patch.object(session, "shutdown_daemon", side_effect=lambda remote, path, key: calls.append(key)), \
             mock.patch.object(session, "stop_tunnel", side_effect=lambda: calls.append("tunnel")), \
             mock.patch.object(session, "alive", return_value=False):
            result = session.cleanup()
        self.assertEqual(calls, ["mac_pid", "spark_pid", "tunnel"])
        self.assertTrue(result["clean"])
        self.assertFalse(result["sigkill_used"])
        self.assertEqual(result["children_alive"], [])

    def cleanup_with_predicate_failure(
        self, side: str, fragment: str
    ) -> tuple[dict[str, object], ResidueRemote]:
        session = self.make_session()
        mac = ResidueRemote(fragment if side == "Mac" else None)
        spark = ResidueRemote(fragment if side == "Spark" else None)
        session.mac = mac
        session.spark = spark
        with mock.patch.object(session, "shutdown_daemon"), \
             mock.patch.object(session, "stop_tunnel"), \
             mock.patch.object(session, "alive", return_value=False):
            result = session.cleanup()
        return result, mac if side == "Mac" else spark

    def test_cleanup_reports_spark_socket_before_later_success(self) -> None:
        session = self.make_session()
        session.mac = ResidueRemote()
        session.spark = FakeSsh(self).remote()
        with tempfile.TemporaryDirectory() as directory:
            command_dir = Path(directory) / "bin"
            command_dir.mkdir()
            for name, returncode in (("pgrep", 1), ("ss", 0)):
                command = command_dir / name
                command.write_text(f"#!/bin/sh\nexit {returncode}\n")
                command.chmod(0o700)
            socket_path = Path(directory) / "active.sock"
            session.spark_socket = str(socket_path)
            with socket.socket(socket.AF_UNIX) as listener:
                listener.bind(str(socket_path))
                listener.listen(1)
                path = f"{command_dir}:{os.environ['PATH']}"
                with mock.patch.dict(os.environ, {"PATH": path}), \
                     mock.patch.object(session, "shutdown_daemon"), \
                     mock.patch.object(session, "stop_tunnel"), \
                     mock.patch.object(session, "alive", return_value=False):
                    result = session.cleanup()
                self.assertTrue(socket_path.exists())
        self.assertFalse(result["clean"])
        self.assertEqual(
            result["problems"],
            [
                f"Spark RPC socket remains: {session.spark_socket} "
                "(exit=1; stdout='<empty>'; stderr='<empty>')"
            ],
        )

    def test_each_cleanup_predicate_fails_independently(self) -> None:
        session = self.make_session()
        cases = (
            ("Mac", "pgrep -x mcdma-rpcd",
             "Mac mcdma-rpcd process remains", 0, "4321"),
            ("Mac", "/tmp/mcdma-rpcd.sock",
             "Mac RPC socket remains: /tmp/mcdma-rpcd.sock", 1, "<empty>"),
            ("Mac", "lsof -nP",
             "Mac loopback listener remains: 127.0.0.1:18620", 0,
             "LISTEN 127.0.0.1:18620"),
            ("Spark", "pgrep -x mcdma-rpcd",
             "Spark mcdma-rpcd process remains", 0, "4321"),
            ("Spark", session.spark_socket,
             f"Spark RPC socket remains: {session.spark_socket}", 1, "<empty>"),
            ("Spark", "ss -ltnH",
             "Spark loopback listener remains: 127.0.0.1:18620", 0,
             "LISTEN 127.0.0.1:18620"),
        )
        for side, fragment, expected, exit_code, stdout in cases:
            with self.subTest(side=side, fragment=fragment):
                result, _remote = self.cleanup_with_predicate_failure(side, fragment)
                self.assertFalse(result["clean"])
                self.assertEqual(len(result["problems"]), 1)
                self.assertEqual(
                    result["problems"][0],
                    f"{expected} (exit={exit_code}; stdout={stdout!r}; "
                    "stderr='<empty>')",
                )

    def test_no_forced_kill_primitive(self) -> None:
        text = ADAPTER_PATH.read_text()
        for forbidden in ("signal.SIGKILL", "kill -9", ".kill(", "pkill"):
            self.assertNotIn(forbidden, text)

    def test_fixed_hashes_match_p05_contract(self) -> None:
        self.assertEqual(ADAPTER.REQUEST_SHA, "66c60468130441e8b45cdf0e7e79ae9e74c08bf3c84d05e4d40ded131e363b9c")
        self.assertEqual(ADAPTER.REPLY_SHA, "cf7ef6b9110e2389d0140a03d32bf910de7a02b6ea8f39283cc237eed1776096")

    def test_mailbox_helper_compiles_and_usage_is_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "mailbox"
            subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                            str(MAILBOX_SOURCE), "-o", str(binary)], check=True)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)
        self.assertIn("reply acknowledgement timeout", MAILBOX_SOURCE.read_text())

    def test_state_is_owner_only_and_mismatch_rejected(self) -> None:
        session = self.make_session("fresh")
        session.save()
        self.assertEqual(session.state_path.stat().st_mode & 0o777, 0o600)
        value = json.loads(session.state_path.read_text())
        value["name"] = "wrong"
        session.state_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ADAPTER.Failure, "mismatch"):
            session.load()


if __name__ == "__main__":
    unittest.main(verbosity=2)
