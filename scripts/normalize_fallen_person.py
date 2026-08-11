#!/usr/bin/env python3
"""Create an RF-DETR-safe four-class copy of the Fallen Person COCO export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fallguard.dataset_normalization import normalize_fallen_person
from fallguard.exceptions import ConfigurationError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    try:
        report = normalize_fallen_person(
            args.source_dir,
            args.output_dir,
            source_archive=args.source_archive,
        )
    except ConfigurationError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
