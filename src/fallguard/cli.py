"""Small package-level command router."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="RFDETR-FallGuard command index")
    parser.add_argument(
        "command",
        nargs="?",
        help="Use scripts/infer_image.py, infer_video.py, run_pipeline.py, or benchmark.py",
    )
    args = parser.parse_args()
    if args.command:
        parser.error("run the named script directly so its evidence boundary is explicit")
    parser.print_help()


if __name__ == "__main__":
    main()
