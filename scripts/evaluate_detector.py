#!/usr/bin/env python3
"""Delegate detection metrics to the official RF-DETR/COCO evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fallguard.config import load_config
from fallguard.detection.rfdetr_adapter import RFDETRDetector
from fallguard.evaluation.detection import evaluate_with_official_rfdetr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/profiles/experiment.yaml")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    config = load_config(args.config)
    detector = RFDETRDetector(
        config.detector.model_copy(update={"weights_path": args.weights}),
        device=config.runtime.device,
    )
    detector.load()
    metrics = evaluate_with_official_rfdetr(detector, args.dataset_dir, split=args.split)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
