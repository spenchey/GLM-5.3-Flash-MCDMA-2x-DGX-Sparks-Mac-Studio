#!/usr/bin/env python3
"""Create a deterministic content or metadata manifest for one artifact tree."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


EXCLUDED_DIRS = {".git", ".mypy_cache", ".pytest_cache", "__pycache__"}
EXCLUDED_FILES = {".DS_Store"}
EXCLUDED_SUFFIXES = {".pyc"}


def included(relative: Path, excluded_relative: frozenset[str] = frozenset()) -> bool:
    return not (
        relative.as_posix() in excluded_relative
        or
        any(part in EXCLUDED_DIRS for part in relative.parts)
        or relative.name in EXCLUDED_FILES
        or relative.suffix in EXCLUDED_SUFFIXES
    )


def manifest(
    root: Path,
    *,
    metadata_only: bool = False,
    excluded_relative: frozenset[str] = frozenset(),
) -> str:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"artifact root is not a directory: {root}")
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if (path.is_file() or path.is_symlink())
            and included(path.relative_to(root), excluded_relative)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths:
        raise ValueError(f"artifact tree has no included files: {root}")

    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        executable = 1 if path.stat().st_mode & 0o111 else 0
        link_target = os.readlink(path).encode() if path.is_symlink() else b""
        size = path.stat().st_size
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(b"L" if path.is_symlink() else b"F")
        digest.update(bytes([executable]))
        digest.update(len(link_target).to_bytes(4, "big"))
        digest.update(link_target)
        digest.update(size.to_bytes(8, "big"))
        if not metadata_only:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--exclude-relative", action="append", default=[])
    args = parser.parse_args()
    print(
        manifest(
            args.root,
            metadata_only=args.metadata_only,
            excluded_relative=frozenset(args.exclude_relative),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
