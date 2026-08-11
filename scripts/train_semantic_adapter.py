#!/usr/bin/env python3
"""Validate QLoRA data/config by default; training requires an explicit --execute flag."""

from __future__ import annotations

import argparse
import json

from fallguard.exceptions import ConfigurationError
from fallguard.training.semantic import execute_qlora, load_qlora_config, validate_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-external-blockers",
        action="store_true",
        help="report missing manifests/model as BLOCKED_EXTERNAL with exit 0",
    )
    args = parser.parse_args()
    config = load_qlora_config(args.config)
    try:
        counts = validate_manifests(config)
    except ConfigurationError as exc:
        if not args.allow_external_blockers:
            raise
        print(
            json.dumps(
                {
                    "state": "BLOCKED_EXTERNAL",
                    "execute": False,
                    "reason": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            {
                "state": "VERIFIED_UNIT" if not args.execute else "EXECUTION_REQUESTED",
                "execute": args.execute,
                "manifest_counts": counts,
            },
            indent=2,
        )
    )
    if args.execute:
        execute_qlora(config)


if __name__ == "__main__":
    main()
