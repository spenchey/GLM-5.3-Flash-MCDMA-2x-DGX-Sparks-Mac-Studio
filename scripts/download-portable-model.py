#!/usr/bin/env python3
"""Download one pinned portable GLM checkpoint into a staging directory."""

from __future__ import annotations

import argparse
from fnmatch import fnmatch
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def visible_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    )


def selected_file_sizes(info: object, patterns: list[str] | None) -> dict[str, int]:
    """Return the pinned Hub inventory selected by ``allow_patterns``."""

    selected: dict[str, int] = {}
    for sibling in info.siblings:
        name = sibling.rfilename
        if patterns is not None and not any(fnmatch(name, pattern) for pattern in patterns):
            continue
        if sibling.size is None:
            raise RuntimeError(f"the Hub returned no byte size for {name}")
        selected[name] = int(sibling.size)
    return selected


def repair_incomplete_files(
    *,
    repo: str,
    revision: str,
    output: Path,
    expected: dict[str, int],
) -> None:
    """Force-redownload files that a resumed local-dir snapshot left truncated."""

    for name, size in expected.items():
        path = output / name
        if path.is_file() and path.stat().st_size == size:
            continue
        hf_hub_download(
            repo_id=repo,
            filename=name,
            revision=revision,
            local_dir=output,
            force_download=True,
        )

    invalid = {
        name: None if not (output / name).is_file() else (output / name).stat().st_size
        for name, size in expected.items()
        if not (output / name).is_file() or (output / name).stat().st_size != size
    }
    if invalid:
        raise RuntimeError(f"download has incomplete files after repair: {invalid}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-start", type=int)
    parser.add_argument("--shard-end", type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    if (args.shard_start is None) != (args.shard_end is None):
        parser.error("--shard-start and --shard-end must be supplied together")
    if args.shard_start is not None and not 1 <= args.shard_start <= args.shard_end <= 43:
        parser.error("the requested shard range must be within 1..43")

    allow_patterns = None
    if args.shard_start is not None:
        allow_patterns = [
            ".gitattributes",
            "LICENSE",
            "README.md",
            "*.json",
            "*.jinja",
            "*.png",
            "tokenizer*",
            "vocab*",
            *[
                f"model-{index:05d}-of-00043.safetensors"
                for index in range(args.shard_start, args.shard_end + 1)
            ],
        ]

    info = HfApi().model_info(
        args.repo,
        revision=args.revision,
        files_metadata=True,
    )
    if info.sha != args.revision:
        raise RuntimeError(f"resolved revision {info.sha!r} does not match the requested pin")

    result = Path(
        snapshot_download(
            repo_id=args.repo,
            revision=args.revision,
            local_dir=args.output,
            max_workers=args.workers,
            allow_patterns=allow_patterns,
        )
    ).resolve()
    repair_incomplete_files(
        repo=args.repo,
        revision=args.revision,
        output=result,
        expected=selected_file_sizes(info, allow_patterns),
    )
    files = visible_files(result)
    required = {"config.json", "model.safetensors.index.json"}
    missing = sorted(required - {path.name for path in files})
    if missing:
        raise RuntimeError(f"download is missing required files: {', '.join(missing)}")
    print(
        json.dumps(
            {
                "requested_repo": args.repo,
                "resolved_repo": info.id,
                "revision": info.sha,
                "output": str(result),
                "files": len(files),
                "bytes": sum(path.stat().st_size for path in files),
                "shard_start": args.shard_start,
                "shard_end": args.shard_end,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
