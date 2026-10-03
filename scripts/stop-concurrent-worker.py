#!/usr/bin/env python3
"""Stop only the paired-capacity worker identified by its private PID record."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    try:
        record = json.loads(args.pid_file.read_text())
    except FileNotFoundError:
        return 0
    expected = {
        "schema": 1,
        "pid": record.get("pid"),
        "worker_id": args.worker_id,
        "module": "experiments.three_machine.concurrent_capacity",
    }
    if record != expected or type(record.get("pid")) is not int or record["pid"] <= 1:
        raise SystemExit("refusing to stop a paired worker with a mismatched identity record")
    pid = int(record["pid"])
    if alive(pid):
        command = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="], text=True
        ).strip()
        required = (
            "experiments.three_machine.concurrent_capacity",
            "--worker-id",
            args.worker_id,
        )
        if any(value not in command for value in required):
            raise SystemExit("refusing to stop a process whose command does not match the worker record")
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + args.timeout
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        if alive(pid):
            os.kill(pid, signal.SIGKILL)
            deadline = time.monotonic() + 5.0
            while alive(pid) and time.monotonic() < deadline:
                time.sleep(0.1)
        if alive(pid):
            raise SystemExit("paired worker did not stop")
    try:
        current = json.loads(args.pid_file.read_text())
    except FileNotFoundError:
        return 0
    if current == record:
        args.pid_file.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
