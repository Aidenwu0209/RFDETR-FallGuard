#!/usr/bin/env python3
"""Run a clearly labeled synthetic MOCK slice or a real video pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fallguard.config import load_config
from fallguard.factory import build_pipeline
from fallguard.logging import setup_logging
from fallguard.mock_run import run_mock_vertical_slice
from fallguard.pipeline import append_jsonl
from fallguard.session import make_session_id
from fallguard.video import VideoReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "real"), required=True)
    parser.add_argument("--config", default="configs/profiles/development.yaml")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--allow-weight-download", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pipeline"))
    parser.add_argument(
        "--ablation",
        choices=("detector_only", "detector_tracking", "temporal", "full"),
        default="full",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config.detector = config.detector.model_copy(
        update={
            "weights_path": args.weights or config.detector.weights_path,
            "allow_weight_download": args.allow_weight_download,
        }
    )
    setup_logging(config.runtime.log_level, json_output=config.runtime.log_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "mock":
        result = run_mock_vertical_slice(config, args.output_dir, ablation_mode=args.ablation)
        append_jsonl(args.output_dir / "events.jsonl", result.events)
        append_jsonl(args.output_dir / "alerts.jsonl", result.alerts)
        print(
            json.dumps(
                {
                    "mode": "MOCK",
                    "ablation": args.ablation,
                    "formal_benchmark_eligible": False,
                    "events": len(result.events),
                    "alerts": len(result.alerts),
                    "transitions": len(result.transitions),
                },
                indent=2,
            )
        )
        return
    if args.video is None:
        parser.error("--video is required in real mode")
    pipeline = build_pipeline(
        config,
        with_real_frontend=True,
        keyframe_output_dir=args.output_dir,
        ablation_mode=args.ablation,
    )
    assert pipeline.detector is not None
    pipeline.detector.load()
    session_id = config.runtime.session_id or make_session_id(args.video.stem)
    with VideoReader(
        args.video, source_id=config.runtime.source_id, session_id=session_id
    ) as reader:
        for packet in reader.frames():
            result = pipeline.process_frame(packet.image_bgr, packet.metadata)
            append_jsonl(args.output_dir / "events.jsonl", result.events)
            append_jsonl(args.output_dir / "alerts.jsonl", result.alerts)


if __name__ == "__main__":
    main()
