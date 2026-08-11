#!/usr/bin/env python3
"""Validate local API-key/package presence without network or paid calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fallguard.status import environment_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--require",
        action="append",
        choices=("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ROBOFLOW_API_KEY"),
        default=[],
    )
    args = parser.parse_args()
    api_keys = environment_status()["api_keys"]
    keys: dict[str, dict[str, bool]] = {
        name: {"present": value["present"]}
        for name, value in api_keys.items()
        if isinstance(value, dict) and "present" in value and isinstance(value["present"], bool)
    }
    report = {
        "validation_kind": "LOCAL_CONFIGURATION_ONLY",
        "keys": keys,
        "required": args.require,
        "network_or_paid_call_performed": False,
        "secret_values_rendered": False,
    }
    rendered = json.dumps(report, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    missing = [name for name in args.require if not keys[name]["present"]]
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
