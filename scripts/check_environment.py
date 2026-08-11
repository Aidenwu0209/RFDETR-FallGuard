#!/usr/bin/env python3
"""Local-only environment audit; never downloads or calls an API."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from fallguard.config import load_config
from fallguard.device import inspect_device


def package_state(module: str, distribution: str | None = None) -> dict[str, str | bool | None]:
    available = importlib.util.find_spec(module) is not None
    try:
        installed = version(distribution or module) if available else None
    except PackageNotFoundError:
        installed = None
    return {"available": available, "version": installed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/profiles/development.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "config_profile": config.runtime.profile,
        "device": inspect_device(config.runtime.device).as_dict(),
        "core": {
            "pydantic": package_state("pydantic"),
            "yaml": package_state("yaml", "PyYAML"),
            "numpy": package_state("numpy"),
            "cv2": package_state("cv2", "opencv-python-headless"),
        },
        "optional": {
            name: package_state(name)
            for name in ("rfdetr", "supervision", "gradio", "openai", "transformers", "peft")
        },
        "network_or_paid_call_performed": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(item["available"] for item in report["core"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
