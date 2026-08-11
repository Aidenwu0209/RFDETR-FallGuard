#!/usr/bin/env python3
"""Confirm frozen thresholds once on S3 without allowing parameter retuning or S4 access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fallguard.exceptions import ConfigurationError
from fallguard.threshold_selection import confirm_thresholds, read_json_object


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-lock", required=True, type=Path)
    parser.add_argument("--validation-report", required=True, type=Path)
    parser.add_argument("--minimum-recall", type=float, default=1.0)
    parser.add_argument("--maximum-false-positive-clips", type=int, default=0)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    try:
        confirmation = confirm_thresholds(
            read_json_object(args.threshold_lock),
            read_json_object(args.validation_report),
            minimum_recall=args.minimum_recall,
            maximum_false_positive_clips=args.maximum_false_positive_clips,
        )
    except ConfigurationError as exc:
        parser.error(str(exc))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(confirmation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(confirmation, indent=2))


if __name__ == "__main__":
    main()
