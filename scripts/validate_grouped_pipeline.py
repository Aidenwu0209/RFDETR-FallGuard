#!/usr/bin/env python3
"""Run posture -> ByteTrack -> temporal -> event on a grouped GMDCSA-24 manifest."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from fallguard.config import AppConfig, load_config
from fallguard.detection.rfdetr_adapter import RFDETRDetector
from fallguard.exceptions import ConfigurationError
from fallguard.factory import build_pipeline
from fallguard.session import make_session_id
from fallguard.threshold_selection import (
    clip_metrics,
    file_sha256,
    read_json_object,
    validate_locked_test_confirmation,
)
from fallguard.video import VideoReader

GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def implementation_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    revision = result.stdout.strip().lower()
    if result.returncode != 0 or GIT_COMMIT_PATTERN.fullmatch(revision) is None:
        raise ConfigurationError("cannot resolve a valid implementation Git commit")
    return revision


def pipeline_parameters(config: AppConfig) -> dict[str, Any]:
    return {
        "detector": {
            "confidence_threshold": config.detector.confidence_threshold,
            "class_names": config.detector.class_names,
            "posture_groups": config.detector.posture_groups,
        },
        "tracking": config.tracking.model_dump(mode="json"),
        "temporal": config.temporal.model_dump(mode="json"),
        "event": config.event.model_dump(mode="json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--model-variant", choices=("nano", "small"))
    parser.add_argument(
        "--partition",
        required=True,
        choices=("threshold_development", "threshold_validation", "locked_test"),
    )
    parser.add_argument("--all-videos", action="store_true")
    parser.add_argument(
        "--unlock-locked-test",
        action="store_true",
        help="explicitly unlock Subject 4 only after thresholds are frozen and confirmed",
    )
    parser.add_argument(
        "--threshold-confirmation",
        type=Path,
        help="required proof that frozen parameters passed the one-time S3 gate before S4",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if args.partition == "locked_test":
        if not args.unlock_locked_test or args.threshold_confirmation is None:
            parser.error("locked_test requires --unlock-locked-test and --threshold-confirmation")
        if not args.all_videos:
            parser.error("locked_test final evaluation requires --all-videos")
    config = load_config(args.config)
    if config.detector.mode.value != "posture_multiclass":
        parser.error("config detector.mode must be posture_multiclass")
    if not config.detector.class_names:
        parser.error("config detector.class_names must match checkpoint metadata")
    weights = args.weights or config.detector.weights_path
    if weights is None:
        parser.error("weights are required via --weights or config detector.weights_path")
    weights = Path(weights).resolve()
    model_variant = args.model_variant or config.detector.model_variant
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    selected = [
        record
        for record in records
        if (args.all_videos or record.get("small_validation_subset"))
        and record.get("partition") == args.partition
    ]
    if not selected:
        parser.error("no manifest videos matched the requested subset/partition")
    config.detector = config.detector.model_copy(
        update={
            "model_variant": model_variant,
            "weights_path": weights,
            "allow_weight_download": False,
        }
    )
    parameters = pipeline_parameters(config)
    weights_sha256 = file_sha256(weights)
    manifest_sha256 = file_sha256(args.manifest)
    if args.partition == "locked_test":
        assert args.threshold_confirmation is not None
        try:
            validate_locked_test_confirmation(
                read_json_object(args.threshold_confirmation),
                manifest_sha256=manifest_sha256,
                protocol=manifest["protocol"],
                model_variant=model_variant,
                weights_sha256=weights_sha256,
                pipeline_parameters=parameters,
                implementation_git_commit=implementation_git_commit(),
            )
        except ConfigurationError as exc:
            parser.error(str(exc))
    rows: list[dict[str, Any]] = []
    detector = RFDETRDetector(config.detector, device=config.runtime.device)
    detector.load()
    for record in selected:
        video = args.dataset_root / record["relative_path"]
        pipeline = build_pipeline(
            config,
            with_real_frontend=True,
            keyframe_output_dir=args.output_json.parent / "keyframes",
            ablation_mode="temporal",
        )
        assert pipeline.detector is not None and pipeline.tracker is not None
        pipeline.detector = detector
        pipeline.tracker.load()
        session_id = make_session_id(record["video_id"])
        with VideoReader(video, source_id=record["video_id"], session_id=session_id) as reader:
            for packet in reader.frames():
                pipeline.process_frame(packet.image_bgr, packet.metadata)
        events = pipeline.event_manager.all_events()
        rows.append(
            {
                "video_id": record["video_id"],
                "subject_id": record["subject_id"],
                "partition": record["partition"],
                "expected_fall": record["label"] == "fall",
                "predicted_fall": bool(events),
                "predicted_event_count": len(events),
                "first_event_start_seconds": min(
                    (event.start_time for event in events), default=None
                ),
            }
        )
    detector.close()
    report = {
        "validation_kind": "GROUPED_CLIP_LEVEL_INTERNAL_VALIDATION",
        "formal_generalization_claim": False,
        "implementation_git_commit": implementation_git_commit(),
        "partition": args.partition,
        "subset": "all_videos" if args.all_videos else "deterministic_small_subset",
        "weights_sha256": weights_sha256,
        "weights_path": str(weights),
        "config_sha256": file_sha256(args.config),
        "manifest_sha256": manifest_sha256,
        "model_variant": model_variant,
        "pipeline_parameters": parameters,
        "config_snapshot": config.model_dump(mode="json"),
        "protocol": manifest["protocol"],
        "selected_video_ids": sorted(row["video_id"] for row in rows),
        "metrics_by_partition": {args.partition: clip_metrics(rows)},
        "detection_delay_available": False,
        "detection_delay_unavailable_reason": (
            "GMDCSA-24 supplies clip labels but no human-confirmed fall onset timestamps"
        ),
        "rows": rows,
        "paid_api_call_performed": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
