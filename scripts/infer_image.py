#!/usr/bin/env python3
"""Run official RF-DETR inference on one image and export internal Detection JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from fallguard.config import load_config
from fallguard.detection.rfdetr_adapter import RFDETRDetector
from fallguard.schemas import FrameMetadata
from fallguard.session import make_session_id
from fallguard.video import write_image
from fallguard.visualization import draw_detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", default="configs/profiles/development.yaml")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--allow-weight-download", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-image", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    detector_config = config.detector.model_copy(
        update={
            "weights_path": args.weights or config.detector.weights_path,
            "allow_weight_download": args.allow_weight_download,
        }
    )
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"cannot read image: {args.image}")
    height, width = image.shape[:2]
    detector = RFDETRDetector(detector_config, device=config.runtime.device)
    detector.load()
    detections = detector.predict_image(
        image,
        FrameMetadata(
            frame_id=0,
            timestamp_seconds=0,
            frame_width=width,
            frame_height=height,
            source_id=config.runtime.source_id,
            session_id=config.runtime.session_id or make_session_id(args.image.stem),
        ),
    )
    payload = [item.model_dump(mode="json") for item in detections]
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if args.output_image:
        write_image(args.output_image, draw_detections(image, detections))
    print(rendered)


if __name__ == "__main__":
    main()
