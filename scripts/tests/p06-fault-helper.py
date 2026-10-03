#!/usr/bin/env python3
"""Bounded P06 fault-test controller.

This program never starts an MCDMA daemon or changes a network device.  It
drives a separately reviewed adapter and enforces the timing, state, counter,
generation, hash, operator-prompt, and cleanup contracts for P06.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


EXIT_CONTRACT = 1
EXIT_USAGE = 2
EXIT_DEADLINE = 3
EXIT_OPERATOR_ABORT = 4
EXIT_CLEANUP = 5

NO_SERVICE_SECONDS = 5
RECONNECT_SECONDS = 30
OPERATOR_SECONDS = 30
CHILD_TERM_SECONDS = 3

REQUEST_SHA256 = "66c60468130441e8b45cdf0e7e79ae9e74c08bf3c84d05e4d40ded131e363b9c"
REPLY_SHA256 = "cf7ef6b9110e2389d0140a03d32bf910de7a02b6ea8f39283cc237eed1776096"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    pass


class DeadlineError(RuntimeError):
    pass


class OperatorAbort(RuntimeError):
    pass


class CleanupError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def integer(value: Any, name: str) -> int:
    require(type(value) is int and value >= 0, f"{name} must be a non-negative integer")
    return value


def daemon(state: dict[str, Any], side: str) -> dict[str, Any]:
    value = state.get(side)
    require(isinstance(value, dict), f"snapshot missing {side} object")
    integer(value.get("pid"), f"{side}.pid")
    integer(value.get("calls"), f"{side}.calls")
    integer(value.get("failures"), f"{side}.failures")
    require(type(value.get("alive")) is bool, f"{side}.alive must be boolean")
    return value


def snapshot_shape(state: dict[str, Any]) -> dict[str, Any]:
    require(state.get("link") in {"up", "down"}, "snapshot link must be up or down")
    integer(state.get("generation"), "generation")
    require(type(state.get("inflight")) is bool, "inflight must be boolean")
    require(type(state.get("reply_present")) is bool, "reply_present must be boolean")
    daemon(state, "mac")
    daemon(state, "spark")
    return state


def require_hash(value: Any, expected: str, name: str) -> None:
    require(isinstance(value, str) and SHA_RE.fullmatch(value) is not None,
            f"{name} must be a lowercase SHA-256")
    require(value == expected, f"{name} mismatch: got {value}, expected {expected}")


class Adapter:
    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.active: subprocess.Popen[str] | None = None
        self.cleanup_attempted = False
        self.cleanup_ok = False

    def call(self, operation: str, timeout: float, *arguments: str) -> dict[str, Any]:
        command = [*self.command, operation, *arguments]
        started = time.monotonic()
        self.active = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = self.active.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            self._term_active()
            raise DeadlineError(
                f"adapter operation {operation} exceeded {timeout:.1f}s"
            ) from exc
        finally:
            process = self.active
            self.active = None
        assert process is not None
        elapsed = time.monotonic() - started
        if process.returncode != 0:
            raise ContractError(
                f"adapter operation {operation} exited {process.returncode}: {stderr.strip()}"
            )
        lines = [line for line in stdout.splitlines() if line.strip()]
        require(len(lines) == 1, f"adapter operation {operation} must emit one JSON line")
        try:
            result = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise ContractError(f"adapter operation {operation} emitted invalid JSON") from exc
        require(isinstance(result, dict), f"adapter operation {operation} JSON must be an object")
        result["_elapsed"] = elapsed
        return result

    def _term_active(self) -> None:
        process = self.active
        if process is None or process.poll() is not None:
            return
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=CHILD_TERM_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise CleanupError(
                f"adapter child {process.pid} ignored SIGTERM; refusing SIGKILL"
            ) from exc

    def cleanup(self) -> None:
        if self.cleanup_attempted:
            return
        self.cleanup_attempted = True
        self._term_active()
        result = self.call("cleanup", 35)
        require(result.get("clean") is True, "adapter cleanup did not report clean=true")
        require(result.get("sigkill_used") is False, "adapter cleanup used SIGKILL")
        require(result.get("children_alive") == [], "adapter cleanup left child processes alive")
        self.cleanup_ok = True


def require_idle_up(state: dict[str, Any]) -> None:
    snapshot_shape(state)
    require(state["link"] == "up", "baseline link is not up")
    require(state["generation"] >= 1, "baseline generation must be at least 1")
    require(state["inflight"] is False, "baseline is not idle")
    require(state["reply_present"] is False, "baseline has a stale reply")
    require(state["mac"]["alive"] is True and state["spark"]["alive"] is True,
            "both daemons must be alive")


def wait_for_reconnect(adapter: Adapter, before: dict[str, Any]) -> dict[str, Any]:
    deadline = time.monotonic() + RECONNECT_SECONDS
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DeadlineError("same-daemon reconnect exceeded 30 seconds")
        state = snapshot_shape(adapter.call("snapshot", min(5.0, remaining)))
        require(state["mac"]["pid"] == before["mac"]["pid"], "Mac daemon PID changed")
        require(state["spark"]["pid"] == before["spark"]["pid"], "Spark daemon PID changed")
        require(state["mac"]["failures"] == before["mac"]["failures"],
                "Mac failure counter changed")
        require(state["spark"]["failures"] == before["spark"]["failures"],
                "Spark failure counter changed")
        if state["link"] == "up" and state["generation"] == before["generation"] + 1:
            return state
        require(state["generation"] in {before["generation"], before["generation"] + 1},
                "generation skipped or regressed during reconnect")
        time.sleep(min(0.25, max(0.01, remaining)))


def verify_call(adapter: Adapter, request_hash: str, reply_hash: str) -> dict[str, Any]:
    result = adapter.call("verify-call", 35)
    require_hash(result.get("request_sha256"), request_hash, "request_sha256")
    require_hash(result.get("reply_sha256"), reply_hash, "reply_sha256")
    require(result.get("partial_reply") is False, "verified call reported a partial reply")
    return result


def operator_token(prompt: str, expected: str) -> None:
    print(prompt, flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], OPERATOR_SECONDS)
    if not ready:
        raise OperatorAbort(f"operator did not enter {expected} within 30 seconds")
    value = sys.stdin.readline().strip()
    if value != expected:
        raise OperatorAbort(f"operator response must be exactly {expected}")


def no_service_timeout(adapter: Adapter, request_hash: str) -> None:
    before = snapshot_shape(adapter.call("snapshot", 5))
    require_idle_up(before)
    require(before["mac"]["calls"] == 0 and before["spark"]["calls"] == 0,
            "no-service baseline calls must both be zero")
    require(before["mac"]["failures"] == 0 and before["spark"]["failures"] == 0,
            "no-service baseline failures must both be zero")
    require(before["spark"].get("service") == "none", "Spark service must be none")
    staged = adapter.call("stage-request", 5)
    require_hash(staged.get("request_sha256"), request_hash, "request_sha256")
    waited = adapter.call("wait-reply", NO_SERVICE_SECONDS + 1.0, str(NO_SERVICE_SECONDS))
    elapsed = waited["_elapsed"]
    require(waited.get("outcome") == "timeout", "no-service wait did not report timeout")
    require(waited.get("reply_present") is False, "no-service wait produced a reply")
    require(NO_SERVICE_SECONDS - 0.15 <= elapsed <= NO_SERVICE_SECONDS + 0.75,
            f"no-service wait was not five seconds: {elapsed:.3f}s")
    after = snapshot_shape(adapter.call("snapshot", 5))
    require(after["generation"] == before["generation"], "generation changed during timeout")
    require(after["mac"]["calls"] == before["mac"]["calls"] + 1,
            "Mac call counter did not advance exactly once")
    require(after["spark"]["calls"] == before["spark"]["calls"],
            "Spark call counter changed without a service")
    require(after["mac"]["failures"] == before["mac"]["failures"],
            "Mac failure counter changed")
    require(after["spark"]["failures"] == before["spark"]["failures"],
            "Spark failure counter changed")
    require(after["reply_present"] is False, "reply appeared after timeout")
    print("P06_NO_SERVICE_TIMEOUT_PASS seconds=5 generation_unchanged=true")


def idle_daemon_loss(adapter: Adapter) -> None:
    before = snapshot_shape(adapter.call("snapshot", 5))
    require_idle_up(before)
    result = adapter.call("idle-daemon-loss", 35)
    require(result.get("orderly") is True, "daemon loss was not orderly")
    require(result.get("sigkill_used") is False, "daemon loss used SIGKILL")
    require(result.get("verbs_destroyed") is True, "verbs cleanup was not proven")
    after = snapshot_shape(adapter.call("snapshot", 5))
    require(after["link"] == "down", "link did not go down after connector loss")
    require(after["generation"] == before["generation"], "generation changed during idle loss")
    require(after["mac"]["alive"] is False, "Mac connector remained alive")
    require(after["spark"]["alive"] is True, "Spark listener did not remain alive")
    require(after["spark"]["pid"] == before["spark"]["pid"], "Spark daemon PID changed")
    require(after["mac"]["calls"] == before["mac"]["calls"], "Mac calls changed while idle")
    require(after["spark"]["calls"] == before["spark"]["calls"], "Spark calls changed while idle")
    require(after["mac"]["failures"] == before["mac"]["failures"], "Mac failures changed")
    require(after["spark"]["failures"] == before["spark"]["failures"], "Spark failures changed")
    require(after["spark"].get("service") == "disconnected", "service disconnect was not observed")
    print("P06_IDLE_DAEMON_LOSS_PASS orderly=true sigkill=false")


def same_daemon_reconnect(adapter: Adapter, request_hash: str, reply_hash: str) -> None:
    before = snapshot_shape(adapter.call("snapshot", 5))
    require_idle_up(before)
    adapter.call("drop-control", 10)
    down = snapshot_shape(adapter.call("snapshot", 5))
    require(down["link"] == "down", "control loss did not produce link down")
    require(down["mac"]["pid"] == before["mac"]["pid"], "Mac daemon PID changed on control loss")
    require(down["spark"]["pid"] == before["spark"]["pid"], "Spark daemon PID changed on control loss")
    require(down["mac"]["alive"] is True and down["spark"]["alive"] is True,
            "a daemon exited during control loss")
    require(down["inflight"] is False, "control was dropped with a call in flight")
    adapter.call("restore-control", 10)
    connected = wait_for_reconnect(adapter, before)
    verify_call(adapter, request_hash, reply_hash)
    after = snapshot_shape(adapter.call("snapshot", 5))
    require(after["generation"] == connected["generation"], "generation changed during verified call")
    require(after["mac"]["calls"] == before["mac"]["calls"] + 1,
            "Mac calls did not advance exactly once")
    require(after["spark"]["calls"] == before["spark"]["calls"] + 1,
            "Spark calls did not advance exactly once")
    print(f"P06_SAME_DAEMON_RECONNECT_PASS generation={after['generation']}")


def cable_window(adapter: Adapter, request_hash: str, reply_hash: str) -> None:
    before = snapshot_shape(adapter.call("snapshot", 5))
    require_idle_up(before)
    staged = adapter.call("stage-cable-call", 5)
    require_hash(staged.get("request_sha256"), request_hash, "request_sha256")
    operator_token("P06 SAFE TO DISCONNECT MAC-FACING QSFP28 NOW; enter DISCONNECT", "DISCONNECT")
    adapter.call("operator-disconnected", 5)
    result = adapter.call("wait-cable-result", OPERATOR_SECONDS + 1, str(OPERATOR_SECONDS))
    require(result.get("outcome") in {"timeout", "failure", "complete"},
            "cable call did not end explicitly")
    require(result.get("partial_reply") is False, "cable call produced a partial reply")
    if result["outcome"] == "complete":
        require_hash(result.get("reply_sha256"), reply_hash, "reply_sha256")
    operator_token("P06 RECONNECT THE SAME QSFP28 TO THE SAME PORT; enter RECONNECTED", "RECONNECTED")
    adapter.call("operator-reconnected", 5)
    connected = wait_for_reconnect(adapter, before)
    verify_call(adapter, request_hash, reply_hash)
    after = snapshot_shape(adapter.call("snapshot", 5))
    require(after["generation"] == connected["generation"], "generation changed after recovery call")
    require(after["mac"]["calls"] >= before["mac"]["calls"] + 1,
            "Mac calls did not record recovery call")
    require(after["spark"]["calls"] >= before["spark"]["calls"] + 1,
            "Spark calls did not record recovery call")
    require(after["mac"]["failures"] >= before["mac"]["failures"], "Mac failures regressed")
    require(after["spark"]["failures"] >= before["spark"]["failures"], "Spark failures regressed")
    print(f"P06_CABLE_WINDOW_PASS outcome={result['outcome']} generation={after['generation']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enforce bounded P06 fault scenarios through a reviewed adapter",
        epilog=(
            "Adapter operations are: snapshot, stage-request, wait-reply, idle-daemon-loss, "
            "drop-control, restore-control, verify-call, stage-cable-call, "
            "operator-disconnected, wait-cable-result, operator-reconnected, cleanup. "
            "Each must emit exactly one JSON object. cleanup must report clean=true, "
            "sigkill_used=false, children_alive=[]."
        ),
    )
    parser.add_argument(
        "mode",
        choices=("no-service-timeout", "idle-daemon-loss", "same-daemon-reconnect", "cable-window"),
    )
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--expected-request-sha256", default=REQUEST_SHA256)
    parser.add_argument("--expected-reply-sha256", default=REPLY_SHA256)
    args = parser.parse_args()
    if not args.adapter.is_file() or not os.access(args.adapter, os.X_OK):
        parser.error("--adapter must be an executable regular file")
    for name in ("expected_request_sha256", "expected_reply_sha256"):
        if SHA_RE.fullmatch(getattr(args, name)) is None:
            parser.error(f"--{name.replace('_', '-')} must be a lowercase SHA-256")
    return args


def main() -> int:
    args = parse_args()
    adapter = Adapter([str(args.adapter)])
    cleanup_problem: list[BaseException] = []

    def cleanup() -> None:
        try:
            adapter.cleanup()
        except BaseException as exc:  # cleanup evidence must never be hidden
            cleanup_problem.append(exc)

    atexit.register(cleanup)
    previous: dict[int, Any] = {}

    def on_signal(signum: int, _frame: Any) -> None:
        cleanup()
        raise KeyboardInterrupt(f"signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.signal(signum, on_signal)

    try:
        if args.mode == "no-service-timeout":
            no_service_timeout(adapter, args.expected_request_sha256)
        elif args.mode == "idle-daemon-loss":
            idle_daemon_loss(adapter)
        elif args.mode == "same-daemon-reconnect":
            same_daemon_reconnect(adapter, args.expected_request_sha256, args.expected_reply_sha256)
        else:
            cable_window(adapter, args.expected_request_sha256, args.expected_reply_sha256)
        cleanup()
        if cleanup_problem:
            raise CleanupError(str(cleanup_problem[0]))
        print("P06_CLEANUP_PASS sigkill=false children_alive=0")
        return 0
    except OperatorAbort as exc:
        print(f"OPERATOR_ABORT: {exc}", file=sys.stderr)
        return EXIT_OPERATOR_ABORT
    except DeadlineError as exc:
        print(f"DEADLINE_FAILURE: {exc}", file=sys.stderr)
        return EXIT_DEADLINE
    except CleanupError as exc:
        print(f"CLEANUP_FAILURE: {exc}", file=sys.stderr)
        return EXIT_CLEANUP
    except (ContractError, KeyError, TypeError) as exc:
        print(f"CONTRACT_FAILURE: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    except KeyboardInterrupt as exc:
        print(f"INTERRUPTED: {exc}", file=sys.stderr)
        return 130
    finally:
        cleanup()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
