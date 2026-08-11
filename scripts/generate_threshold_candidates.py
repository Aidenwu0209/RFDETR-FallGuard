#!/usr/bin/env python3
"""Generate a small threshold set before grouped development evaluation begins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from fallguard.config import load_config
from fallguard.exceptions import ConfigurationError
from fallguard.threshold_selection import (
    CANDIDATE_PRESETS,
    EXPANDED_PRECISION_PRESETS,
    file_sha256,
    generate_candidate_configs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--include-expanded-precision-grid",
        action="store_true",
        help="add the documented development-only confidence grid",
    )
    parser.add_argument(
        "--precision-grid-only",
        action="store_true",
        help="emit only the predeclared 0.40-0.80 confidence grid",
    )
    args = parser.parse_args()
    try:
        candidates = generate_candidate_configs(
            load_config(args.base_config),
            include_expanded_precision_grid=args.include_expanded_precision_grid,
            precision_grid_only=args.precision_grid_only,
        )
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
                "predeclared_parameters": {
                    **CANDIDATE_PRESETS,
                    **EXPANDED_PRECISION_PRESETS,
                }[name],
            }
        )
    manifest = {
        "manifest_kind": "PREDECLARED_THRESHOLD_CANDIDATES",
        "selection_partition": "threshold_development",
        "candidate_count": len(records),
        "development_stage": (
            "cross_subject_precision_grid"
            if args.precision_grid_only
            else (
                "stage2_precision_expansion_after_stage1_false_positive_diagnostic"
                if args.include_expanded_precision_grid
                else "stage1_initial_presets"
            )
        ),
        "candidates": records,
        "warning": (
            "candidate values are provisional until selected on development subjects and "
            "confirmed once on an untouched validation group"
        ),
    }
    manifest_path = args.output_dir / "candidate-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
