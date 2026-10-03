#!/usr/bin/env python3
"""Fail when the tracked public snapshot contains private deployment data."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"LICENSE", "NOTICE", "CREDITS.md", "SECURITY.md", "UPSTREAM.lock"}
BLOCKED_SUFFIXES = {".safetensors", ".gguf", ".ckpt", ".onnx", ".pt", ".pth"}

PATTERNS = {
    "private IPv4 address": re.compile(
        r"(?<![0-9])(?:10\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}|"
        r"192\.168\.[0-9]{1,3}\.[0-9]{1,3}|"
        r"100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\."
        r"[0-9]{1,3}\.[0-9]{1,3})(?![0-9])"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "OpenAI-style token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}

HOME_PATH = re.compile(r"/(Users|home)/([^/\s`:'\"<>]+)")
ALLOWED_HOME_USERS = {"user", "root"}


def tracked_files() -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / part.decode() for part in raw.split(b"\0") if part]


def optional_markers() -> list[str]:
    marker_file = os.environ.get("PRIVATE_MARKERS_FILE")
    if not marker_file:
        marker_file = str(ROOT / ".private-markers")
    path = Path(marker_file)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines()
            if len(line.strip()) >= 4 and not line.lstrip().startswith("#")]


def main() -> int:
    errors: list[str] = []
    names = {path.relative_to(ROOT).as_posix() for path in tracked_files()}
    for required in sorted(REQUIRED - names):
        errors.append(f"missing required public file: {required}")

    markers = optional_markers()
    for path in tracked_files():
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            errors.append(f"model artifact is tracked: {rel}")
            continue
        try:
            text = path.read_text(errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} found in {rel}")
        for match in HOME_PATH.finditer(text):
            if match.group(2) not in ALLOWED_HOME_USERS:
                errors.append(f"private home path found in {rel}")
                break
        for marker in markers:
            if marker in text:
                errors.append(f"local private marker found in {rel}")
                break

    if errors:
        print("PUBLIC RELEASE CHECK FAILED", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"PASS: {len(names)} tracked files contain required notices and no detected private data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
