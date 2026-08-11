#!/usr/bin/env python3
"""Run RF-DETR plus pinned class-agnostic ByteTrack and export tracked detections."""

from __future__ import annotations

import argparse
from pathlib import Path

from fallguard.config import load_config
from fallguard.detection.rfdetr_adapter import RFDETRDetector
from fallguard.pipeline import append_jsonl
from fallguard.session import make_session_id
from fallguard.tracking.bytetrack_adapter import ByteTrackAdapter
from fallguard.video import VideoReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", default="configs/profiles/development.yaml")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--allow-weight-download", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    detector_config = config.detector.model_copy(
        update={
            "weights_path": args.weights or config.detector.weights_path,
            "allow_weight_download": args.allow_weight_download,
        }
    )
    detector = RFDETRDetector(detector_config, device=config.runtime.device)
    tracker = ByteTrackAdapter(config.tracking)
    detector.load()
    tracker.load()
    session_id = config.runtime.session_id or make_session_id(args.video.stem)
    with VideoReader(
        args.video, source_id=config.runtime.source_id, session_id=session_id
    ) as reader:
        for packet in reader.frames():
            detections = detector.predict_frame(packet.image_bgr, packet.metadata)
            append_jsonl(args.output_jsonl, tracker.update(detections, packet.metadata))


if __name__ == "__main__":
    main()
