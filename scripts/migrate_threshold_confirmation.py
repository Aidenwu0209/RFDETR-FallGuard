#!/usr/bin/env python3
"""Attach a control-plane-only migration proof to a grouped threshold confirmation."""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path

from fallguard.exceptions import ConfigurationError
from fallguard.implementation_fingerprint import (
    git_runtime_core_sha256,
    pipeline_implementation_sha256,
    runtime_core_sha256,
)
from fallguard.threshold_selection import (
    canonical_sha256,
    is_threshold_confirmation,
    read_json_object,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-confirmation", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        parser.error("control-plane migration requires a clean committed worktree")
    confirmation = read_json_object(args.threshold_confirmation)
    if not is_threshold_confirmation(confirmation):
        parser.error("input is not a grouped threshold confirmation")
    if "control_plane_migration" in confirmation:
        parser.error("confirmation already contains a control-plane migration")
    selected = confirmation.get("selected")
    if not isinstance(selected, dict):
        parser.error("confirmation has no selected candidate")
    from_revision = selected.get("implementation_git_commit")
    from_fingerprint = selected.get("pipeline_implementation_sha256")
    if not isinstance(from_revision, str) or not isinstance(from_fingerprint, str):
        parser.error("confirmation has incomplete implementation identity")
    try:
        before_core = git_runtime_core_sha256(project_root, from_revision)
        after_core = runtime_core_sha256(project_root)
    except ConfigurationError as exc:
        parser.error(str(exc))
    if before_core != after_core:
        parser.error("runtime behavior core changed; control-plane migration is forbidden")
    current_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    migrated = deepcopy(confirmation)
    migrated["control_plane_migration"] = {
        "migration_kind": "JSON_KEY_CANONICALIZATION_ONLY",
        "reason": (
            "JSON object keys are strings while validated class-name mappings use integer keys"
        ),
        "original_confirmation_sha256": canonical_sha256(confirmation),
        "from_implementation_git_commit": from_revision,
        "from_pipeline_implementation_sha256": from_fingerprint,
        "to_implementation_git_commit": current_revision,
        "to_pipeline_implementation_sha256": pipeline_implementation_sha256(project_root),
        "runtime_core_sha256_before": before_core,
        "runtime_core_sha256_after": after_core,
        "model_or_threshold_parameter_changed": False,
        "validation_partition_reused": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(migrated, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(migrated["control_plane_migration"], indent=2))


if __name__ == "__main__":
    main()
