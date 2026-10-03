#!/usr/bin/env python3
"""Hash the exact reusable project payload deployed to all three machines."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


RUNTIME_ROOT_FILES = {"UPSTREAM.lock", "config.example.env"}
RUNTIME_ROOT_DIRS = {"scripts", "experiments/three_machine", "patches/tensorfold"}
EXCLUDED_DIRS = {".git", ".state", ".context", ".pytest_cache", "__pycache__", "raw"}
EXCLUDED_FILES = {
    "config.local.env",
    ".DS_Store",
    ".deployment-id",
    ".deployment-sha256",
}
EXCLUDED_SUFFIXES = {".pyc", ".log", ".exit"}


def included(relative: Path) -> bool:
    in_runtime_payload = (
        relative.as_posix() in RUNTIME_ROOT_FILES
        or any(
            relative == Path(root) or Path(root) in relative.parents
            for root in RUNTIME_ROOT_DIRS
        )
    )
    return in_runtime_payload and not (
        any(part in EXCLUDED_DIRS for part in relative.parts)
        or relative.name in EXCLUDED_FILES
        or relative.suffix in EXCLUDED_SUFFIXES
    )


def manifest(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        (path for path in root.rglob("*") if (path.is_file() or path.is_symlink())
         and included(path.relative_to(root))),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        if path.is_symlink():
            kind = b"L"
            payload = os.readlink(path).encode()
            executable = 0
        else:
            kind = b"F"
            payload = path.read_bytes()
            executable = 1 if path.stat().st_mode & 0o111 else 0
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(kind)
        digest.update(bytes([executable]))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(manifest(args.root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
