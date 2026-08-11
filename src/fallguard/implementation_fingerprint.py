"""Hash complete validation implementations and the runtime behavior core separately."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from fallguard.exceptions import ConfigurationError

GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
RUNTIME_CONTROL_PLANE_EXCLUSIONS = frozenset(
    {
        "src/fallguard/implementation_fingerprint.py",
        "src/fallguard/status.py",
        "src/fallguard/threshold_selection.py",
    }
)


def _hash_named_blobs(blobs: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for relative, content in sorted(blobs):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def pipeline_implementation_sha256(project_root: Path) -> str:
    sources = sorted((project_root / "src/fallguard").rglob("*.py"))
    sources.append(project_root / "scripts/validate_grouped_pipeline.py")
    return _hash_named_blobs(
        [(str(source.relative_to(project_root)), source.read_bytes()) for source in sources]
    )


def runtime_core_sha256(project_root: Path) -> str:
    sources = [
        source
        for source in sorted((project_root / "src/fallguard").rglob("*.py"))
        if str(source.relative_to(project_root)) not in RUNTIME_CONTROL_PLANE_EXCLUSIONS
    ]
    return _hash_named_blobs(
        [(str(source.relative_to(project_root)), source.read_bytes()) for source in sources]
    )


def git_runtime_core_sha256(project_root: Path, revision: str) -> str:
    if GIT_COMMIT_PATTERN.fullmatch(revision) is None:
        raise ConfigurationError("invalid Git revision for runtime-core comparison")
    listed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", revision, "src/fallguard"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        raise ConfigurationError("cannot list runtime-core sources from the confirmation commit")
    paths = [
        path
        for path in listed.stdout.splitlines()
        if path.endswith(".py") and path not in RUNTIME_CONTROL_PLANE_EXCLUSIONS
    ]
    blobs: list[tuple[str, bytes]] = []
    for path in paths:
        shown = subprocess.run(
            ["git", "show", f"{revision}:{path}"],
            cwd=project_root,
            capture_output=True,
            check=False,
        )
        if shown.returncode != 0:
            raise ConfigurationError(f"cannot read runtime-core source from Git: {path}")
        blobs.append((path, shown.stdout))
    if not blobs:
        raise ConfigurationError("confirmation commit contains no runtime-core sources")
    return _hash_named_blobs(blobs)
