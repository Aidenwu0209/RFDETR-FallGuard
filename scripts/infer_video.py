#!/usr/bin/env python3
"""Run official RF-DETR video inference and export one Detection JSON record per line."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from fallguard.config import load_config
from fallguard.detection.rfdetr_adapter import RFDETRDetector
from fallguard.pipeline import append_jsonl
from fallguard.session import make_session_id
from fallguard.video import VideoReader
from fallguard.visualization import draw_detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", default="configs/profiles/development.yaml")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--allow-weight-download", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-video", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    detector_config = config.detector.model_copy(
        update={
            "weights_path": args.weights or config.detector.weights_path,
            "allow_weight_download": args.allow_weight_download,
        }
    )
    detector = RFDETRDetector(detector_config, device=config.runtime.device)
    detector.load()
    session_id = config.runtime.session_id or make_session_id(args.video.stem)
    writer: cv2.VideoWriter | None = None
    try:
        with VideoReader(
            args.video,
            source_id=config.runtime.source_id,
            session_id=session_id,
        ) as reader:
            for packet in reader.frames():
                detections = detector.predict_frame(packet.image_bgr, packet.metadata)
                append_jsonl(args.output_jsonl, detections)
                if args.output_video:
                    if writer is None:
                        args.output_video.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(
                            str(args.output_video),
                            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
                            reader.fps or 30.0,
                            (packet.metadata.frame_width, packet.metadata.frame_height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"cannot create output video: {args.output_video}")
                    writer.write(draw_detections(packet.image_bgr, detections))
    finally:
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
