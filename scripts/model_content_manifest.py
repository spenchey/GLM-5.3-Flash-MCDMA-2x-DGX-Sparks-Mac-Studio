#!/usr/bin/env python3
"""Create one path-and-content digest for an immutable model directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def manifest(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    combined = hashlib.sha256()
    sizes = hashlib.sha256()
    total = 0
    for path in paths:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        sizes.update(f"{relative}\t{size}\n".encode("utf-8"))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
                total += len(chunk)
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.digest())
        combined.update(b"\0")
    return {
        "sha256": combined.hexdigest(),
        "size_sha256": sizes.hexdigest(),
        "files": len(paths),
        "bytes": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(manifest(args.root), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
