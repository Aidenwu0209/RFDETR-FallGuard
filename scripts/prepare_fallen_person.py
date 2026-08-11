#!/usr/bin/env python3
"""Audit a Roboflow Fallen Person COCO export and emit its RF-DETR class profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from fallguard.data_audit import audit_fallen_person, posture_profile
from fallguard.exceptions import ConfigurationError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    parser.add_argument("--strict-leakage", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.near_duplicate_distance <= 64:
        parser.error("--near-duplicate-distance must be in [0, 64]")
    try:
        report = audit_fallen_person(
            args.dataset_dir, near_duplicate_distance=args.near_duplicate_distance
        )
    except ConfigurationError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "audit.json"
    profile_path = args.output_dir / "posture_profile.yaml"
    audit_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    profile_path.write_text(
        yaml.safe_dump(posture_profile(report), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key != "records"}
    summary.update({"audit_path": str(audit_path), "profile_path": str(profile_path)})
    print(json.dumps(summary, indent=2))
    if args.strict_leakage and not report["formal_original_split_evaluation_eligible"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
