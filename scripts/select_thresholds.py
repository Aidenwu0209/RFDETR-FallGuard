#!/usr/bin/env python3
"""Select one S1-S2 candidate, freeze its full pipeline parameters, and emit an S3 config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from fallguard.exceptions import ConfigurationError
from fallguard.threshold_selection import (
    frozen_config_from_lock,
    read_json_object,
    select_thresholds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-report", action="append", required=True, type=Path)
    parser.add_argument("--minimum-recall", type=float, default=1.0)
    parser.add_argument("--maximum-false-positive-clips", type=int, default=0)
    parser.add_argument("--preferred-variant", choices=("nano", "small"), default="nano")
    parser.add_argument("--output-lock", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    args = parser.parse_args()
    try:
        candidates = [(str(path), read_json_object(path)) for path in args.development_report]
        lock = select_thresholds(
            candidates,
            minimum_recall=args.minimum_recall,
            maximum_false_positive_clips=args.maximum_false_positive_clips,
            preferred_variant=args.preferred_variant,
        )
        frozen_config = frozen_config_from_lock(lock)
    except ConfigurationError as exc:
        parser.error(str(exc))
    args.output_lock.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_lock.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    args.output_config.write_text(
        yaml.safe_dump(frozen_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
