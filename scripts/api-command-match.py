#!/usr/bin/env python3
"""Match one Mac API process command against the exact owned launch shape."""

from __future__ import annotations

import argparse
import shlex


def matches(command: str, python: str, model: str, mailbox: str, host: str, port: str) -> bool:
    try:
        words = shlex.split(command)
    except ValueError:
        return False
    return words == [
        python,
        "experiments/three_machine/api_server.py",
        "--model",
        model,
        "--mailbox",
        mailbox,
        "--host",
        host,
        "--port",
        port,
        "--timeout",
        "180",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("python")
    parser.add_argument("model")
    parser.add_argument("mailbox")
    parser.add_argument("host")
    parser.add_argument("port")
    args = parser.parse_args()
    return 0 if matches(
        args.command, args.python, args.model, args.mailbox, args.host, args.port
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
