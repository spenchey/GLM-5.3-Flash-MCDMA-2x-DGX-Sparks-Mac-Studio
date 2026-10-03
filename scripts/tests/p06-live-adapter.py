#!/usr/bin/env python3
"""Narrow live adapter for the reviewed P06 fault controller.

The adapter owns only one named P06 session.  It changes no network settings;
it starts the pinned daemons and an authenticated loopback SSH forward, exposes
the JSON operations required by p06-fault-helper.py, and always shuts down in
connector, listener, tunnel order.  It never sends SIGKILL.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class Failure(RuntimeError):
    pass


def integer_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default))
    if not raw.isdigit() or not minimum <= int(raw) <= maximum:
        raise Failure(f"{name} must be an integer from {minimum} through {maximum}")
    return int(raw)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/tests/p06-live-mailbox.c"
REQUEST_SHA = "66c60468130441e8b45cdf0e7e79ae9e74c08bf3c84d05e4d40ded131e363b9c"
REPLY_SHA = "cf7ef6b9110e2389d0140a03d32bf910de7a02b6ea8f39283cc237eed1776096"
STUDIO_DAEMON = os.environ.get("P06_LIVE_STUDIO_DAEMON", "/opt/mcdma/bin/mcdma-rpcd")
SPARK_DAEMON = os.environ.get("P06_LIVE_SPARK_DAEMON", "/opt/mcdma/bin/mcdma-rpcd")
STUDIO_HOST = os.environ.get("P06_LIVE_STUDIO_HOST", "mac-studio")
SPARK_HOST = os.environ.get("P06_LIVE_SPARK_HOST", "spark-head")
TUNNEL_HOST = os.environ.get("P06_LIVE_TUNNEL_HOST", SPARK_HOST)
STUDIO_CONTROL_SOCKET = os.environ.get(
    "P06_LIVE_STUDIO_CONTROL_SOCKET", str(Path.home() / ".ssh/cm-studio")
)
CONTROL_PORT = integer_env("P06_LIVE_CONTROL_PORT", 18620, 1, 65535)
STUDIO_DEVICE = os.environ.get("P06_LIVE_STUDIO_DEVICE", "rdma_mcrdma1")
SPARK_DEVICE = os.environ.get("P06_LIVE_SPARK_DEVICE", "rocep1s0f0")
STUDIO_DEVICE_INDEX = integer_env("P06_LIVE_STUDIO_DEVICE_INDEX", 0, 0, 255)
SPARK_DEVICE_INDEX = integer_env("P06_LIVE_SPARK_DEVICE_INDEX", 1, 0, 255)
REQUEST_MIB = integer_env("P06_LIVE_REQUEST_MIB", 4, 4, 256)
REPLY_MIB = integer_env("P06_LIVE_REPLY_MIB", 4, 4, 256)
if REQUEST_MIB % 4 or REPLY_MIB % 4:
    raise Failure("P06_LIVE_REQUEST_MIB and P06_LIVE_REPLY_MIB must be multiples of 4")
PULL_MODE = integer_env("P06_LIVE_PULL", 0, 0, 1) == 1
STUDIO_SSH = ["ssh", "-S", STUDIO_CONTROL_SOCKET, "-o", "ControlMaster=no", STUDIO_HOST]
SPARK_SSH = ["ssh", "-o", "BatchMode=yes", SPARK_HOST]
STATUS_RE = re.compile(
    r"^(?:PEER\s+)?(\S+) (up|down) calls (\d+) failures (\d+) MiB (\d+)(.*)$"
)
URL_CREDENTIAL_RE = re.compile(
    r"(?i)(\b[a-z][a-z0-9+.-]*://)[^\s/@]+@"
)
SENSITIVE_MARKER_RE = re.compile(
    r"(?i:(?<![A-Za-z0-9])(?:password|passwd|token|secret|authorization|"
    r"bearer|credential|cookie)(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])(?:api|access|private)[\s_.-]*key"
    r"(?![A-Za-z0-9]))|"
    r"(?<![A-Za-z0-9])(?:[cC]lient(?:Secret|Token)|[aA]ccessToken|"
    r"[rR]efreshToken|[aA]uthToken|[sS]ession(?:Cookie|Token)|"
    r"[pP]asswordHash|[aA]piKey|[aA]ccessKey|[pP]rivateKey|"
    r"(?:api|Api|API|oauth|Oauth|OAuth|OAUTH|id|Id|ID|csrf|Csrf|CSRF)Token)"
    r"(?![A-Za-z0-9])"
)
DIAGNOSTIC_LIMIT = 2_000


def sanitized_diagnostic(value: str) -> str:
    redacted_urls = URL_CREDENTIAL_RE.sub(r"\1<redacted>@", value)
    if SENSITIVE_MARKER_RE.search(redacted_urls):
        return "<redacted-sensitive-output>"
    normalized = " ".join(redacted_urls.split())
    if len(normalized) > DIAGNOSTIC_LIMIT:
        normalized = normalized[:DIAGNOSTIC_LIMIT] + "...[truncated]"
    return normalized or "<empty>"


def run(command: list[str], *, input_text: str | None = None, timeout: float = 30,
        check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, input=input_text, text=True, capture_output=True,
                            timeout=timeout)
    if check and result.returncode:
        raise Failure(
            f"command exited {result.returncode}: "
            f"stdout={sanitized_diagnostic(result.stdout)!r}; "
            f"stderr={sanitized_diagnostic(result.stderr)!r}"
        )
    return result


class Remote:
    def __init__(self, prefix: list[str]) -> None:
        self.prefix = prefix

    def shell(self, script: str, *, input_text: str | None = None,
              timeout: float = 30, check: bool = True) -> subprocess.CompletedProcess[str]:
        # OpenSSH joins every argument after the host with spaces before the
        # login shell parses it.  Quote the complete script so `sh -c` receives
        # exactly one payload rather than only its first word.
        return run([*self.prefix, "sh", "-c", shlex.quote(script)],
                   input_text=input_text, timeout=timeout, check=check)

    def compile(self, source: str, output: str) -> None:
        command = f"umask 077; cc -std=c11 -O2 -Wall -Wextra -Werror -x c -o {shlex.quote(output)} -"
        result = self.shell(command, input_text=source, timeout=45)
        if result.stdout.strip() or result.stderr.strip():
            raise Failure(
                "mailbox helper compile emitted unexpected output: "
                f"stdout={sanitized_diagnostic(result.stdout)!r}; "
                f"stderr={sanitized_diagnostic(result.stderr)!r}"
            )


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


class Session:
    def __init__(self) -> None:
        state_value = os.environ.get("P06_LIVE_STATE")
        scenario = os.environ.get("P06_LIVE_SCENARIO")
        if not state_value or scenario not in {"no-service", "idle-loss", "reconnect", "fresh"}:
            raise Failure("P06_LIVE_STATE and a valid P06_LIVE_SCENARIO are required")
        self.state_dir = Path(state_value).resolve()
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.state_dir, 0o700)
        self.state_path = self.state_dir / "state.json"
        self.scenario = scenario
        self.name = "p06c" + {"no-service": "ns", "idle-loss": "il", "reconnect": "rc", "fresh": "fr"}[scenario]
        self.remote_dir = f"/tmp/{self.name}-{os.getuid()}"
        self.mac = Remote(STUDIO_SSH)
        self.spark = Remote(SPARK_SSH)
        self.helper = f"{self.remote_dir}/p06-live-mailbox"
        self.mac_socket = "/tmp/mcdma-rpcd.sock"
        self.spark_socket = f"/tmp/mcdma-rpcd.{self.name}.sock"
        self.tunnel_socket = f"{self.remote_dir}/control.sock"
        self.state = self.load()

    def load(self) -> dict[str, Any]:
        if self.state_path.exists():
            value = json.loads(self.state_path.read_text())
            if value.get("scenario") != self.scenario or value.get("name") != self.name:
                raise Failure("state scenario/name mismatch")
            return value
        return {"scenario": self.scenario, "name": self.name, "started": False,
                "mac_calls": 0, "spark_calls": 0, "mac_failures": 0,
                "spark_failures": 0, "generation": 0, "disconnected": False,
                "sigkill_used": False}

    def save(self) -> None:
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, sort_keys=True) + "\n")
        os.chmod(temp, 0o600)
        os.replace(temp, self.state_path)

    def alive(self, remote: Remote, pid: int | None) -> bool:
        if not pid:
            return False
        return remote.shell(f"kill -0 {int(pid)} 2>/dev/null", check=False).returncode == 0

    def pid(self, remote: Remote, file: str) -> int:
        text = remote.shell(f"cat {q(file)}").stdout.strip()
        if not text.isdigit() or int(text) <= 1:
            raise Failure(f"invalid pid file {file}: {text!r}")
        return int(text)

    def unix(self, remote: Remote, path: str, line: str, *, check: bool = True) -> str:
        # A VERSION/PEER/END status response may arrive in more than one Unix
        # socket packet. Reading once made health checks intermittently see
        # only VERSION and abort a healthy startup.
        program = (
            "import socket,sys\n"
            "s=socket.socket(socket.AF_UNIX)\n"
            "s.settimeout(3)\n"
            f"s.connect({path!r})\n"
            f"s.sendall({(line + chr(10)).encode()!r})\n"
            "chunks=[]\n"
            "first=s.recv(4096)\n"
            "chunks.append(first)\n"
            "s.settimeout(0.05)\n"
            "while first:\n"
            "    try:\n"
            "        first=s.recv(4096)\n"
            "    except TimeoutError:\n"
            "        break\n"
            "    chunks.append(first)\n"
            "sys.stdout.buffer.write(b''.join(chunks))\n"
        )
        result = remote.shell(f"python3 -c {q(program)}", timeout=8, check=check)
        return result.stdout.strip()

    def wait_until(self, predicate: Any, seconds: float, message: str) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.1)
        raise Failure(message)

    def status(self, remote: Remote, path: str) -> dict[str, Any]:
        raw = self.unix(remote, path, "STATUS")
        match = next(
            (
                candidate
                for line in raw.splitlines()
                if (candidate := STATUS_RE.match(line))
                and candidate.group(1) == self.name
            ),
            None,
        )
        if not match:
            raise Failure(f"unexpected STATUS response: {raw!r}")
        tail = match.group(6)
        service = "attached" if "service=attached" in tail else "none" if "service=none" in tail else None
        return {"link": match.group(2), "calls": int(match.group(3)),
                "failures": int(match.group(4)), "service": service, "raw": raw}

    def helper_json(self, remote: Remote, arguments: list[str], *, timeout: float = 30) -> dict[str, Any]:
        command = " ".join(q(value) for value in [self.helper, *arguments])
        result = remote.shell(command, timeout=timeout)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise Failure(f"helper emitted {len(lines)} JSON lines")
        return json.loads(lines[0])

    def start_tunnel(self) -> None:
        exposure = (
            f"test \"$(ss -ltnH 'sport = :{CONTROL_PORT}' 2>/dev/null | "
            f"awk '$4 ~ /(^|])127\\.0\\.0\\.1:{CONTROL_PORT}$/ {{n++}} END {{print n+0}}')\" = 1"
        )
        self.mac.shell(f"ssh -o BatchMode=yes {q(TUNNEL_HOST)} {q(exposure)}")
        command = (
            f"ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o GatewayPorts=no "
            f"-o ServerAliveInterval=15 -o ServerAliveCountMax=2 -M -S {q(self.tunnel_socket)} "
            f"-fN -L 127.0.0.1:{CONTROL_PORT}:127.0.0.1:{CONTROL_PORT} {q(TUNNEL_HOST)}"
        )
        self.mac.shell(command)
        self.mac.shell(f"ssh -o BatchMode=yes -S {q(self.tunnel_socket)} -O check {q(TUNNEL_HOST)}")

    def stop_tunnel(self) -> None:
        check = f"ssh -o BatchMode=yes -S {q(self.tunnel_socket)} -O check {q(TUNNEL_HOST)}"
        if self.mac.shell(check, check=False).returncode == 0:
            self.mac.shell(f"ssh -o BatchMode=yes -S {q(self.tunnel_socket)} -O exit {q(TUNNEL_HOST)}")
        self.wait_until(lambda: self.mac.shell(f"test ! -e {q(self.tunnel_socket)}", check=False).returncode == 0,
                        8, "tunnel control socket survived stop")

    def ensure_started(self) -> None:
        if self.state["started"]:
            return
        # Fail closed on any pre-existing owned or global endpoint.
        mac_absent = f"! pgrep -x mcdma-rpcd >/dev/null 2>&1 && test ! -e /tmp/mcdma-rpcd.sock && ! lsof -nP -iTCP@127.0.0.1:{CONTROL_PORT} -sTCP:LISTEN >/dev/null 2>&1"
        spark_absent = f"! pgrep -x mcdma-rpcd >/dev/null 2>&1 && test ! -e /tmp/mcdma-rpcd.*.sock && test -z \"$(ss -ltnH 'sport = :{CONTROL_PORT}' 2>/dev/null)\""
        self.mac.shell(mac_absent)
        self.spark.shell(spark_absent)
        source = SOURCE.read_text()
        for remote in (self.mac, self.spark):
            # A clean shutdown deliberately preserves its small private log
            # directory. Reusing that exact owned directory is safe after the
            # process/socket absence gates above have passed.
            remote.shell(f"umask 077; mkdir -p {q(self.remote_dir)}; chmod 700 {q(self.remote_dir)}")
            remote.compile(source, self.helper)
        spark_log = f"{self.remote_dir}/listener.log"
        spark_pid = f"{self.remote_dir}/listener.pid"
        self.spark.shell(
            f"umask 077; nohup {q(SPARK_DAEMON)} listen {q(self.name)} {q(SPARK_DEVICE)} "
            f"{SPARK_DEVICE_INDEX} 4096 127.0.0.1:{CONTROL_PORT} {REQUEST_MIB} {REPLY_MIB} "
            f">{q(spark_log)} 2>&1 & echo $! >{q(spark_pid)}"
        )
        self.state["spark_pid"] = self.pid(self.spark, spark_pid)
        self.wait_until(lambda: self.spark.shell(f"test -S {q(self.spark_socket)}", check=False).returncode == 0,
                        10, "Spark listener socket did not appear")
        self.start_tunnel()
        mac_log = f"{self.remote_dir}/connector.log"
        mac_pid = f"{self.remote_dir}/connector.pid"
        pull_environment = "env MCDMA_RPC_PULL=1" if PULL_MODE else "env -u MCDMA_RPC_PULL"
        self.mac.shell(
            f"umask 077; {pull_environment} nohup {q(STUDIO_DAEMON)} connect "
            f"{q(self.name + f',127.0.0.1,{CONTROL_PORT},{STUDIO_DEVICE},{STUDIO_DEVICE_INDEX},4096,{REQUEST_MIB},{REPLY_MIB}')} "
            f">{q(mac_log)} 2>&1 & echo $! >{q(mac_pid)}"
        )
        self.state["mac_pid"] = self.pid(self.mac, mac_pid)
        self.wait_until(lambda: self.mac.shell(f"test -S {q(self.mac_socket)}", check=False).returncode == 0,
                        10, "Mac connector socket did not appear")
        self.wait_until(lambda: self.status(self.mac, self.mac_socket)["link"] == "up",
                        20, "MCDMA link did not become ready")
        inspect = self.helper_json(self.mac, ["inspect", self.name])
        if not inspect.get("ready") or int(inspect.get("generation", 0)) < 1:
            raise Failure("mailbox did not publish ready generation")
        self.state["generation"] = int(inspect["generation"])
        self.state["started"] = True
        self.save()
        if self.scenario in {"idle-loss", "reconnect"}:
            # The registered stages require one exact, byte-checked baseline call
            # before the idle software fault is introduced.
            self.verify_call()
            hold_log = f"{self.remote_dir}/hold.log"
            hold_pid = f"{self.remote_dir}/hold.pid"
            self.spark.shell(
                f"umask 077; nohup {q(self.helper)} hold-service {q(self.spark_socket)} 120 "
                f">{q(hold_log)} 2>&1 & echo $! >{q(hold_pid)}"
            )
            self.state["hold_pid"] = self.pid(self.spark, hold_pid)
            self.wait_until(lambda: self.status(self.spark, self.spark_socket)["service"] == "attached",
                            5, "Spark service did not attach")
            self.save()

    def snapshot(self) -> dict[str, Any]:
        self.ensure_started()
        mac_alive = self.alive(self.mac, self.state.get("mac_pid"))
        spark_alive = self.alive(self.spark, self.state.get("spark_pid"))
        mac_status = self.status(self.mac, self.mac_socket) if mac_alive else None
        spark_status = self.status(self.spark, self.spark_socket) if spark_alive else None
        if mac_status:
            self.state["mac_calls"] = mac_status["calls"]
            self.state["mac_failures"] = mac_status["failures"]
        if spark_status:
            self.state["spark_calls"] = spark_status["calls"]
            self.state["spark_failures"] = spark_status["failures"]
        inspect = None
        if mac_alive:
            inspect = self.helper_json(self.mac, ["inspect", self.name])
            self.state["generation"] = int(inspect["generation"])
        self.save()
        service = "none"
        if self.state.get("disconnected"):
            service = "disconnected"
        elif spark_status and spark_status.get("service") == "attached":
            service = "poll"
        return {
            "reply_transport": "rdma-read-pull" if PULL_MODE else "rdma-write-push",
            "link": mac_status["link"] if mac_status else "down",
            "generation": self.state["generation"],
            "inflight": bool(inspect and inspect["request_seq"] != inspect["reply_seq"]),
            # A completed historical pair (request_seq == reply_seq) is idle,
            # not a stale reply.  Only an impossible reply ahead of its request
            # is reported as present without an active call.
            "reply_present": bool(inspect and inspect["reply_seq"] > inspect["request_seq"]),
            "mac": {"pid": int(self.state.get("mac_pid", 0)), "calls": self.state["mac_calls"],
                    "failures": self.state["mac_failures"], "alive": mac_alive},
            "spark": {"pid": int(self.state.get("spark_pid", 0)), "calls": self.state["spark_calls"],
                      "failures": self.state["spark_failures"], "alive": spark_alive,
                      "service": service},
        }

    def peek(self) -> dict[str, Any]:
        """Return state without starting daemons or changing the link."""

        if not self.state.get("started"):
            return {
                "reply_transport": "rdma-read-pull" if PULL_MODE else "rdma-write-push",
                "link": "down", "generation": int(self.state.get("generation", 0)),
                "inflight": False, "reply_present": False,
                "mac": {"pid": int(self.state.get("mac_pid", 0)), "calls": int(self.state.get("mac_calls", 0)),
                        "failures": int(self.state.get("mac_failures", 0)), "alive": False},
                "spark": {"pid": int(self.state.get("spark_pid", 0)),
                          "calls": int(self.state.get("spark_calls", 0)),
                          "failures": int(self.state.get("spark_failures", 0)),
                          "alive": False, "service": "none"},
            }
        return self.snapshot()

    def stage_request(self) -> dict[str, Any]:
        path = f"{self.remote_dir}/no-service-request.bin"
        self.helper_json(self.mac, ["stage", self.name, path])
        digest = self.mac.shell(f"shasum -a 256 {q(path)} | awk '{{print $1}}'").stdout.strip()
        return {"request_sha256": digest}

    def wait_reply(self, seconds: str) -> dict[str, Any]:
        return self.helper_json(self.mac, ["wait", self.name, seconds], timeout=int(seconds) + 3)

    def wait_pid_exit(self, remote: Remote, key: str, seconds: float) -> None:
        pid = self.state.get(key)
        if pid:
            self.wait_until(lambda: not self.alive(remote, int(pid)), seconds,
                            f"{key} {pid} survived orderly shutdown")

    def idle_loss(self) -> dict[str, Any]:
        if self.unix(self.mac, self.mac_socket, "SHUTDOWN") != "BYE":
            raise Failure("Mac SHUTDOWN did not return BYE")
        self.wait_pid_exit(self.mac, "mac_pid", 30)
        self.wait_pid_exit(self.spark, "hold_pid", 10)
        connector_log = self.mac.shell(f"cat {q(self.remote_dir + '/connector.log')}").stdout
        hold_log = self.spark.shell(f"cat {q(self.remote_dir + '/hold.log')}").stdout
        verbs = "every verbs object destroyed" in connector_log
        disconnected = '"disconnected":true' in hold_log
        if not verbs or not disconnected:
            raise Failure("orderly connector/service cleanup proof missing")
        self.state["disconnected"] = True
        self.save()
        return {"orderly": True, "sigkill_used": False, "verbs_destroyed": True}

    def drop_control(self) -> dict[str, Any]:
        self.stop_tunnel()
        self.wait_until(lambda: self.status(self.mac, self.mac_socket)["link"] == "down", 10,
                        "connector did not report link down")
        self.wait_pid_exit(self.spark, "hold_pid", 10)
        hold_log = self.spark.shell(f"cat {q(self.remote_dir + '/hold.log')}").stdout
        if '"disconnected":true' not in hold_log:
            raise Failure("service disconnect proof missing")
        self.state["disconnected"] = True
        self.save()
        return {"ok": True}

    def restore_control(self) -> dict[str, Any]:
        self.start_tunnel()
        return {"ok": True}

    def verify_call(self) -> dict[str, Any]:
        tag = f"call-{int(time.time_ns())}"
        service_req = f"{self.remote_dir}/{tag}-spark-request.bin"
        service_rep = f"{self.remote_dir}/{tag}-spark-reply.bin"
        client_req = f"{self.remote_dir}/{tag}-mac-request.bin"
        client_rep = f"{self.remote_dir}/{tag}-mac-reply.bin"
        service_log = f"{self.remote_dir}/{tag}-service.log"
        service_pid = f"{self.remote_dir}/{tag}-service.pid"
        self.spark.shell(
            f"umask 077; nohup {q(self.helper)} service-one {q(self.name)} {q(self.spark_socket)} "
            f"{q(service_req)} {q(service_rep)} 30 >{q(service_log)} 2>&1 & echo $! >{q(service_pid)}"
        )
        pid = self.pid(self.spark, service_pid)
        self.wait_until(lambda: self.status(self.spark, self.spark_socket)["service"] == "attached",
                        5, "verification service did not attach")
        self.helper_json(self.mac, ["client-one", self.name, client_req, client_rep, "30"], timeout=35)
        self.wait_until(lambda: not self.alive(self.spark, pid), 10, "verification service survived call")
        hashes = {}
        for side, remote, request, reply in (
            ("mac", self.mac, client_req, client_rep),
            ("spark", self.spark, service_req, service_rep),
        ):
            raw = remote.shell(f"shasum -a 256 {q(request)} {q(reply)}").stdout.splitlines()
            hashes[side] = [line.split()[0] for line in raw]
        if hashes["mac"] != hashes["spark"] or hashes["mac"] != [REQUEST_SHA, REPLY_SHA]:
            raise Failure(f"verification hashes differ: {hashes}")
        return {"request_sha256": REQUEST_SHA, "reply_sha256": REPLY_SHA,
                "partial_reply": False}

    def shutdown_daemon(self, remote: Remote, path: str, key: str) -> None:
        if self.alive(remote, self.state.get(key)):
            reply = self.unix(remote, path, "SHUTDOWN", check=False)
            if reply != "BYE":
                raise Failure(f"{key} SHUTDOWN did not return BYE: {reply!r}")
            self.wait_pid_exit(remote, key, 30)

    def residue_problem(self, remote: Remote, label: str, command: str,
                        expected_returncode: int = 0) -> str | None:
        result = remote.shell(command, check=False)
        if (result.returncode == expected_returncode and
                not result.stdout.strip() and not result.stderr.strip()):
            return None
        return (
            f"{label} (exit={result.returncode}; "
            f"stdout={sanitized_diagnostic(result.stdout)!r}; "
            f"stderr={sanitized_diagnostic(result.stderr)!r})"
        )

    def cleanup(self) -> dict[str, Any]:
        problems: list[str] = []
        try:
            self.shutdown_daemon(self.mac, self.mac_socket, "mac_pid")
        except Exception as exc:
            problems.append(f"connector: {exc}")
        try:
            self.shutdown_daemon(self.spark, self.spark_socket, "spark_pid")
        except Exception as exc:
            problems.append(f"listener: {exc}")
        try:
            self.stop_tunnel()
        except Exception as exc:
            problems.append(f"tunnel: {exc}")
        children = []
        for remote, key in ((self.mac, "mac_pid"), (self.spark, "spark_pid"), (self.spark, "hold_pid")):
            if self.alive(remote, self.state.get(key)):
                children.append(int(self.state[key]))
        residue_checks = (
            (self.mac, "Mac mcdma-rpcd process remains",
             "pgrep -x mcdma-rpcd", 1),
            (self.mac, "Mac RPC socket remains: /tmp/mcdma-rpcd.sock",
             "test ! -e /tmp/mcdma-rpcd.sock", 0),
            (self.mac, f"Mac loopback listener remains: 127.0.0.1:{CONTROL_PORT}",
             f"lsof -nP -iTCP@127.0.0.1:{CONTROL_PORT} -sTCP:LISTEN", 1),
            (self.spark, "Spark mcdma-rpcd process remains",
             "pgrep -x mcdma-rpcd", 1),
            (self.spark, f"Spark RPC socket remains: {self.spark_socket}",
             f"test ! -e {q(self.spark_socket)}", 0),
            (self.spark, f"Spark loopback listener remains: 127.0.0.1:{CONTROL_PORT}",
             f"ss -ltnH 'sport = :{CONTROL_PORT}'", 0),
        )
        for remote, label, command, expected_returncode in residue_checks:
            problem = self.residue_problem(
                remote, label, command, expected_returncode
            )
            if problem:
                problems.append(problem)
        clean = not problems and not children
        self.state["cleanup_problems"] = problems
        self.state["started"] = False
        self.save()
        return {"clean": clean, "sigkill_used": False, "children_alive": children,
                "problems": problems}


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def main() -> int:
    if len(sys.argv) < 2:
        print("operation required", file=sys.stderr)
        return 2
    try:
        session = Session()
        operation = sys.argv[1]
        if operation == "snapshot": emit(session.snapshot())
        elif operation == "peek": emit(session.peek())
        elif operation == "stage-request": emit(session.stage_request())
        elif operation == "wait-reply" and len(sys.argv) == 3: emit(session.wait_reply(sys.argv[2]))
        elif operation == "idle-daemon-loss": emit(session.idle_loss())
        elif operation == "drop-control": emit(session.drop_control())
        elif operation == "restore-control": emit(session.restore_control())
        elif operation == "verify-call": emit(session.verify_call())
        elif operation == "fresh-call":
            session.ensure_started(); emit(session.verify_call())
        elif operation == "cleanup": emit(session.cleanup())
        else:
            raise Failure(f"unsupported operation {operation!r}")
        return 0
    except (Failure, subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        print(f"P06_LIVE_ADAPTER_FAILURE: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
