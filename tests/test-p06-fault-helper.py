#!/usr/bin/env python3
"""Deterministic, offline tests for the bounded P06 fault helper."""

from __future__ import annotations

import errno
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/tests/p06-fault-helper.py"
REQUEST_HASH = "66c60468130441e8b45cdf0e7e79ae9e74c08bf3c84d05e4d40ded131e363b9c"
REPLY_HASH = "cf7ef6b9110e2389d0140a03d32bf910de7a02b6ea8f39283cc237eed1776096"

MOCK_ADAPTER = r'''#!/usr/bin/env python3
import json, os, signal, subprocess, sys, time

path = os.environ["MOCK_STATE"]
scenario = os.environ.get("MOCK_SCENARIO", "normal")
request_hash = "66c60468130441e8b45cdf0e7e79ae9e74c08bf3c84d05e4d40ded131e363b9c"
reply_hash = "cf7ef6b9110e2389d0140a03d32bf910de7a02b6ea8f39283cc237eed1776096"

def save(s):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(s, f)
    os.replace(tmp, path)

def load():
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    service = "none" if scenario.startswith("no_service") else "poll"
    child = subprocess.Popen(["sleep", "300"], start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, close_fds=True)
    s = {
        "link": "up", "generation": 1, "inflight": False,
        "reply_present": False,
        "mac": {"pid": 111, "calls": 0, "failures": 0, "alive": True},
        "spark": {"pid": 222, "calls": 0, "failures": 0, "alive": True,
                  "service": service},
        "child_pid": child.pid,
    }
    save(s)
    return s

op = sys.argv[1]
s = load()
out = {}
if op == "snapshot":
    out = {k: v for k, v in s.items() if k != "child_pid"}
elif op == "stage-request":
    s["inflight"] = True; s["mac"]["calls"] += 1; save(s)
    out = {"request_sha256": "0" * 64 if scenario.endswith("bad_hash") else request_hash}
elif op == "wait-reply":
    if scenario.endswith("adapter_deadline"):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        time.sleep(20)
    else:
        time.sleep(float(sys.argv[2]))
        if scenario.endswith("bad_counter"):
            s["spark"]["calls"] += 1; save(s)
        out = {"outcome": "timeout", "reply_present": False}
elif op == "idle-daemon-loss":
    s["link"] = "down"; s["mac"]["alive"] = False
    s["spark"]["service"] = "disconnected"; save(s)
    out = {"orderly": True, "sigkill_used": False, "verbs_destroyed": True}
elif op == "drop-control":
    s["link"] = "down"; save(s); out = {"ok": True}
elif op == "restore-control":
    s["link"] = "up"
    s["generation"] = 3 if scenario == "bad_generation" else 2
    save(s); out = {"ok": True}
elif op == "verify-call":
    s["mac"]["calls"] += 1; s["spark"]["calls"] += 1
    s["inflight"] = False; save(s)
    out = {"request_sha256": request_hash,
           "reply_sha256": "f" * 64 if scenario == "bad_reply_hash" else reply_hash,
           "partial_reply": False}
elif op == "stage-cable-call":
    s["inflight"] = True; save(s); out = {"request_sha256": request_hash}
elif op == "operator-disconnected":
    s["link"] = "down"; save(s); out = {"ok": True}
elif op == "wait-cable-result":
    s["inflight"] = False; save(s)
    out = {"outcome": "timeout", "partial_reply": False}
elif op == "operator-reconnected":
    s["link"] = "up"; s["generation"] = 2; save(s); out = {"ok": True}
elif op == "cleanup":
    pid = s.get("child_pid")
    if pid:
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
        for _ in range(100):
            try: os.kill(pid, 0)
            except ProcessLookupError: break
            time.sleep(.01)
    bad = scenario == "cleanup_bad"
    out = {"clean": not bad, "sigkill_used": False,
           "children_alive": [pid] if bad else []}
else:
    print("unknown operation", file=sys.stderr); sys.exit(9)
print(json.dumps(out), flush=True)
'''


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        raise


class FaultHelperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.adapter = self.directory / "adapter.py"
        self.adapter.write_text(MOCK_ADAPTER)
        self.adapter.chmod(self.adapter.stat().st_mode | stat.S_IXUSR)
        self.state = self.directory / "state.json"

    def tearDown(self) -> None:
        if self.state.exists():
            state = json.loads(self.state.read_text())
            pid = state.get("child_pid")
            if pid and alive(pid):
                os.kill(pid, signal.SIGTERM)
        self.tmp.cleanup()

    def run_helper(self, mode: str, scenario: str = "normal", stdin: str = "", extra=None):
        env = {**os.environ, "MOCK_STATE": str(self.state), "MOCK_SCENARIO": scenario}
        command = [sys.executable, str(HELPER), mode, "--adapter", str(self.adapter)]
        if extra:
            command.extend(extra)
        started = time.monotonic()
        result = subprocess.run(command, input=stdin, text=True, capture_output=True, env=env,
                                timeout=45)
        return result, time.monotonic() - started

    def assert_cleanup(self) -> None:
        state = json.loads(self.state.read_text())
        pid = state["child_pid"]
        for _ in range(100):
            if not alive(pid):
                return
            time.sleep(.01)
        self.fail(f"mock child {pid} survived cleanup")

    def test_usage_is_exit_2(self) -> None:
        result = subprocess.run([sys.executable, str(HELPER), "no-service-timeout"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)

    def test_no_service_exact_timeout_hash_and_cleanup(self) -> None:
        result, elapsed = self.run_helper("no-service-timeout", "no_service")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(elapsed, 4.85)
        self.assertLess(elapsed, 6.5)
        self.assertIn("P06_NO_SERVICE_TIMEOUT_PASS", result.stdout)
        self.assertIn("P06_CLEANUP_PASS", result.stdout)
        self.assert_cleanup()

    def test_hash_rejection_is_exit_1(self) -> None:
        result, _ = self.run_helper("no-service-timeout", "no_service_bad_hash")
        self.assertEqual(result.returncode, 1)
        self.assertIn("request_sha256 mismatch", result.stderr)
        self.assert_cleanup()

    def test_first_counter_mismatch_is_exit_1(self) -> None:
        result, _ = self.run_helper("no-service-timeout", "no_service_bad_counter")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Spark call counter changed", result.stderr)
        self.assert_cleanup()

    def test_adapter_deadline_is_exit_3_and_cleanup(self) -> None:
        result, elapsed = self.run_helper("no-service-timeout", "no_service_adapter_deadline")
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertGreaterEqual(elapsed, 5.9)
        self.assertLess(elapsed, 10)
        self.assertIn("DEADLINE_FAILURE", result.stderr)
        self.assert_cleanup()

    def test_idle_loss_requires_orderly_state(self) -> None:
        result, _ = self.run_helper("idle-daemon-loss")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("P06_IDLE_DAEMON_LOSS_PASS", result.stdout)
        self.assert_cleanup()

    def test_same_daemon_reconnect_and_hashes(self) -> None:
        result, _ = self.run_helper("same-daemon-reconnect")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("generation=2", result.stdout)
        self.assert_cleanup()

    def test_generation_skip_is_exit_1(self) -> None:
        result, _ = self.run_helper("same-daemon-reconnect", "bad_generation")
        self.assertEqual(result.returncode, 1)
        self.assertIn("generation skipped", result.stderr)
        self.assert_cleanup()

    def test_reply_hash_rejection_is_exit_1(self) -> None:
        result, _ = self.run_helper("same-daemon-reconnect", "bad_reply_hash")
        self.assertEqual(result.returncode, 1)
        self.assertIn("reply_sha256 mismatch", result.stderr)
        self.assert_cleanup()

    def test_operator_abort_is_exit_4(self) -> None:
        result, _ = self.run_helper("cable-window", stdin="NO\n")
        self.assertEqual(result.returncode, 4)
        self.assertIn("OPERATOR_ABORT", result.stderr)
        self.assert_cleanup()

    def test_cable_prompt_gate_reconnect_and_cleanup(self) -> None:
        result, _ = self.run_helper("cable-window", stdin="DISCONNECT\nRECONNECTED\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("P06 SAFE TO DISCONNECT", result.stdout)
        self.assertIn("P06 RECONNECT THE SAME", result.stdout)
        self.assertIn("P06_CABLE_WINDOW_PASS", result.stdout)
        self.assert_cleanup()

    def test_cleanup_failure_is_exit_5(self) -> None:
        result, _ = self.run_helper("idle-daemon-loss", "cleanup_bad")
        self.assertEqual(result.returncode, 5, (result.stdout, result.stderr))
        self.assertIn("CLEANUP_FAILURE", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
