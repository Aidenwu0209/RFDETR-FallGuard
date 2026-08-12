#!/usr/bin/env python3
"""Replay known false negatives for stage attribution without threshold selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fallguard.config import AppConfig, load_config
from fallguard.detection.rfdetr_adapter import RFDETRDetector
from fallguard.factory import build_pipeline
from fallguard.schemas import MotionState
from fallguard.session import make_session_id
from fallguard.video import VideoReader


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_attribution(
    *,
    production_detections: int,
    tracked_detections: int,
    transition_states: list[str],
    event_count: int,
) -> str:
    """Return the earliest stage that failed to pass a production candidate."""

    if production_detections == 0:
        return "detector_no_production_detection"
    if tracked_detections == 0:
        return "tracking_no_output"
    if MotionState.SUSPECTED.value not in transition_states:
        return "temporal_never_suspected"
    if not ({MotionState.FALLING.value, MotionState.LYING.value} & set(transition_states)):
        return "temporal_candidate_not_confirmed"
    if event_count == 0:
        return "event_not_created"
    return "event_reproduced_during_diagnostic"


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
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--locked-report", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--diagnostic-confidence-floor", type=float, default=0.05)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--acknowledge-test-diagnostic-only", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_test_diagnostic_only:
        parser.error("test-set replay requires --acknowledge-test-diagnostic-only")
    if not 0 <= args.diagnostic_confidence_floor < 1:
        parser.error("--diagnostic-confidence-floor must be in [0, 1)")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    config = load_config(args.config)
    locked = json.loads(args.locked_report.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if locked.get("partition") != "locked_test":
        parser.error("--locked-report must be the previously opened locked_test report")
    normalized_pipeline_parameters = json.loads(json.dumps(pipeline_parameters(config)))
    if normalized_pipeline_parameters != locked.get("pipeline_parameters"):
        parser.error("config pipeline parameters differ from the locked report")

    false_negative_ids = {
        row["video_id"]
        for row in locked.get("rows", [])
        if row.get("expected_fall") is True and row.get("predicted_fall") is False
    }
    records = [
        record
        for record in manifest.get("records", [])
        if record.get("video_id") in false_negative_ids
    ]
    records.sort(key=lambda record: str(record["video_id"]))
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        parser.error("the locked report contains no matching false-negative videos")

    production_threshold = config.detector.confidence_threshold
    diagnostic_detector_config = config.detector.model_copy(
        update={
            "weights_path": args.weights.resolve(),
            "allow_weight_download": False,
            "confidence_threshold": args.diagnostic_confidence_floor,
        }
    )
    detector = RFDETRDetector(diagnostic_detector_config, device=config.runtime.device)
    detector.load()
    rows: list[dict[str, Any]] = []
    fall_classes = {
        *config.detector.posture_groups.get("fall", []),
        *config.detector.posture_groups.get("lying", []),
    }
    try:
        for index, record in enumerate(records, start=1):
            video_id = str(record["video_id"])
            video = args.dataset_root / str(record["relative_path"])
            pipeline = build_pipeline(
                config,
                with_real_frontend=True,
                keyframe_output_dir=args.output_json.parent / "diagnostic-keyframes",
                ablation_mode="temporal",
            )
            assert pipeline.tracker is not None
            pipeline.tracker.load()
            session_id = make_session_id(video_id)
            frame_count = 0
            low_detections = 0
            production_detections = 0
            tracked_detections = 0
            low_fall_posture_detections = 0
            production_fall_posture_detections = 0
            frames_with_production_detections = 0
            frames_with_tracks = 0
            production_class_counts: Counter[str] = Counter()
            low_class_max_confidence: defaultdict[str, float] = defaultdict(float)
            production_class_max_confidence: defaultdict[str, float] = defaultdict(float)
            track_ids: set[int] = set()
            transition_states: list[str] = []
            transition_reasons: list[str] = []
            with VideoReader(video, source_id=video_id, session_id=session_id) as reader:
                for packet in reader.frames():
                    frame_count += 1
                    detections = detector.predict_frame(packet.image_bgr, packet.metadata)
                    low_detections += len(detections)
                    for detection in detections:
                        low_class_max_confidence[detection.class_name] = max(
                            low_class_max_confidence[detection.class_name], detection.confidence
                        )
                        if detection.class_name in fall_classes:
                            low_fall_posture_detections += 1
                    production = [
                        detection
                        for detection in detections
                        if detection.confidence >= production_threshold
                    ]
                    production_detections += len(production)
                    frames_with_production_detections += bool(production)
                    for detection in production:
                        production_class_counts[detection.class_name] += 1
                        production_class_max_confidence[detection.class_name] = max(
                            production_class_max_confidence[detection.class_name],
                            detection.confidence,
                        )
                        if detection.class_name in fall_classes:
                            production_fall_posture_detections += 1
                    tracked = pipeline.tracker.update(production, packet.metadata)
                    tracked_detections += len(tracked)
                    frames_with_tracks += bool(tracked)
                    track_ids.update(item.track_id for item in tracked)
                    result = pipeline.process_tracked(packet.image_bgr, packet.metadata, tracked)
                    transition_states.extend(item.next_state.value for item in result.transitions)
                    transition_reasons.extend(item.reason for item in result.transitions)
            events = pipeline.event_manager.all_events()
            attribution = stage_attribution(
                production_detections=production_detections,
                tracked_detections=tracked_detections,
                transition_states=transition_states,
                event_count=len(events),
            )
            rows.append(
                {
                    "video_id": video_id,
                    "subject_id": record["subject_id"],
                    "frames": frame_count,
                    "stage_attribution": attribution,
                    "diagnostic_floor": args.diagnostic_confidence_floor,
                    "production_threshold": production_threshold,
                    "low_floor_detections": low_detections,
                    "production_detections": production_detections,
                    "frames_with_production_detections": frames_with_production_detections,
                    "low_floor_fall_or_lying_detections": low_fall_posture_detections,
                    "production_fall_or_lying_detections": production_fall_posture_detections,
                    "low_floor_class_max_confidence": dict(
                        sorted(low_class_max_confidence.items())
                    ),
                    "production_class_counts": dict(sorted(production_class_counts.items())),
                    "production_class_max_confidence": dict(
                        sorted(production_class_max_confidence.items())
                    ),
                    "tracked_detections": tracked_detections,
                    "frames_with_tracks": frames_with_tracks,
                    "unique_track_ids": len(track_ids),
                    "transition_states": transition_states,
                    "transition_reasons": transition_reasons,
                    "event_count": len(events),
                }
            )
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(records)}",
                        "video_id": video_id,
                        "stage_attribution": attribution,
                    }
                ),
                flush=True,
            )
    finally:
        detector.close()

    summary = Counter(row["stage_attribution"] for row in rows)
    report = {
        "diagnostic_kind": "OPENED_TEST_SET_FALSE_NEGATIVE_STAGE_ATTRIBUTION_ONLY",
        "must_not_select_or_tune_parameters": True,
        "formal_evaluation_claim": False,
        "source_locked_report": str(args.locked_report),
        "source_locked_report_sha256": file_sha256(args.locked_report),
        "manifest_sha256": file_sha256(args.manifest),
        "weights_sha256": file_sha256(args.weights),
        "diagnostic_confidence_floor": args.diagnostic_confidence_floor,
        "production_threshold": production_threshold,
        "false_negative_count_in_source_report": len(false_negative_ids),
        "diagnosed_count": len(rows),
        "stage_attribution_counts": dict(sorted(summary.items())),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage_attribution_counts": report["stage_attribution_counts"]}, indent=2))


if __name__ == "__main__":
    main()
