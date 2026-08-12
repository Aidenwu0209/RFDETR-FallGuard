#!/usr/bin/env python3
"""Download a pinned public Qwen checkpoint and record exact local file hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_file_manifest(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative.startswith(".cache/") or relative == "download-manifest.json":
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if len(args.revision) != 40 or any(
        character not in "0123456789abcdef" for character in args.revision
    ):
        parser.error("--revision must be a full lowercase 40-character Git commit hash")
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint.rstrip("/")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import model_info, snapshot_download

    info = model_info(args.repo_id, revision=args.revision)
    if info.sha != args.revision:
        raise RuntimeError(
            f"resolved revision {info.sha!r} differs from requested {args.revision!r}"
        )
    args.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
        max_workers=args.max_workers,
    )
    files = local_file_manifest(args.local_dir)
    manifest = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
    }
    output = args.local_dir / "download-manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
