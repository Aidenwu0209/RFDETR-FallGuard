#!/usr/bin/env python3
"""Validate or explicitly execute RF-DETR training with official parameter names."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fallguard.config import load_config
from fallguard.data_audit import validate_training_audit
from fallguard.detection.rfdetr_adapter import RFDETRDetector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/profiles/experiment.yaml")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rfdetr-training"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--model-variant", choices=("nano", "small"))
    parser.add_argument("--allow-weight-download", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--amp-dtype", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scale-jitter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-test", action="store_true")
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
    audited = None
    if args.dataset_audit is not None:
        audited = validate_training_audit(
            args.dataset_audit, args.dataset_dir, detector_config.class_names
        )
    if args.execute and audited is None:
        parser.error("--dataset-audit is required with --execute")
    class_names = [
        detector_config.class_names[index] for index in range(len(detector_config.class_names))
    ]
    training = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "dataset_file": "roboflow",
        "class_names": class_names,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "checkpoint_interval": args.checkpoint_interval,
        "amp_dtype": args.amp_dtype,
        "multi_scale": args.multi_scale,
        "scale_jitter": args.scale_jitter,
        "tensorboard": args.tensorboard,
        "run_test": args.run_test,
        "square_resize_div_64": True,
        "notes": {
            "purpose": "authorized short posture fine-tune",
            "model_variant": detector_config.model_variant,
            "dataset_audit_sha256": (
                hashlib.sha256(args.dataset_audit.read_bytes()).hexdigest()
                if args.dataset_audit is not None
                else None
            ),
        },
    }
    if args.gradient_checkpointing:
        training["gradient_checkpointing"] = True
    resolved = detector.resolve_train_configuration(training)
    print(
        json.dumps(
            {
                "execute": args.execute,
                "audit_verified": audited is not None,
                "official_parameters": resolved,
            },
            indent=2,
        )
    )
    if args.execute:
        detector.load_for_training()
        detector.train(**resolved)


if __name__ == "__main__":
    main()
