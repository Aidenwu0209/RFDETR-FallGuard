#!/usr/bin/env python3
"""Validate or explicitly execute RF-DETR training with official parameter names."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fallguard.config import load_config
from fallguard.detection.rfdetr_adapter import RFDETRDetector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/profiles/experiment.yaml")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rfdetr-training"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--model-variant", choices=("nano", "small"))
    parser.add_argument("--allow-weight-download", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    detector_config = config.detector.model_copy(
        update={
            "model_variant": args.model_variant or config.detector.model_variant,
            "weights_path": args.weights or config.detector.weights_path,
            "allow_weight_download": args.allow_weight_download,
        }
    )
    detector = RFDETRDetector(detector_config, device=config.runtime.device)
    training = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "accelerator": args.accelerator,
        "devices": args.devices,
    }
    if args.gradient_checkpointing:
        training["gradient_checkpointing"] = True
    resolved = detector.resolve_train_configuration(training)
    print(json.dumps({"execute": args.execute, "official_parameters": resolved}, indent=2))
    if args.execute:
        detector.load()
        detector.train(**resolved)


if __name__ == "__main__":
    main()
