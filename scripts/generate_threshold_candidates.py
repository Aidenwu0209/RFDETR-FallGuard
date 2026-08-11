#!/usr/bin/env python3
"""Generate a small predeclared threshold set before S1-S2 evaluation begins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from fallguard.config import load_config
from fallguard.exceptions import ConfigurationError
from fallguard.threshold_selection import (
    CANDIDATE_PRESETS,
    file_sha256,
    generate_candidate_configs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        candidates = generate_candidate_configs(load_config(args.base_config))
    except ConfigurationError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, config in candidates.items():
        path = args.output_dir / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        records.append(
            {
                "candidate": name,
                "config_path": str(path),
                "config_sha256": file_sha256(path),
                "predeclared_parameters": CANDIDATE_PRESETS[name],
            }
        )
    manifest = {
        "manifest_kind": "PREDECLARED_THRESHOLD_CANDIDATES",
        "selection_partition": "threshold_development",
        "candidate_count": len(records),
        "candidates": records,
        "warning": (
            "candidate values are provisional until selected on S1-S2 and confirmed once on S3"
        ),
    }
    manifest_path = args.output_dir / "candidate-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
