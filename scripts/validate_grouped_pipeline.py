#!/usr/bin/env python3
"""Run posture -> ByteTrack -> temporal -> event on a grouped GMDCSA-24 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from fallguard.config import load_config
from fallguard.detection.rfdetr_adapter import RFDETRDetector
from fallguard.factory import build_pipeline
from fallguard.session import make_session_id
from fallguard.video import VideoReader


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["expected_fall"] and row["predicted_fall"] for row in rows)
    fp = sum(not row["expected_fall"] and row["predicted_fall"] for row in rows)
    fn = sum(row["expected_fall"] and not row["predicted_fall"] for row in rows)
    tn = sum(not row["expected_fall"] and not row["predicted_fall"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "clips": len(rows),
        "true_positive_clips": tp,
        "false_positive_clips": fp,
        "false_negative_clips": fn,
        "true_negative_clips": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--model-variant", required=True, choices=("nano", "small"))
    parser.add_argument(
        "--partition", choices=("threshold_development", "threshold_validation", "locked_test")
    )
    parser.add_argument("--all-videos", action="store_true")
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.detector.mode.value != "posture_multiclass":
        parser.error("config detector.mode must be posture_multiclass")
    if not config.detector.class_names:
        parser.error("config detector.class_names must match checkpoint metadata")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    selected = [
        record
        for record in records
        if (args.all_videos or record.get("small_validation_subset"))
        and (args.partition is None or record.get("partition") == args.partition)
    ]
    if not selected:
        parser.error("no manifest videos matched the requested subset/partition")
    config.detector = config.detector.model_copy(
        update={
            "model_variant": args.model_variant,
            "weights_path": args.weights,
            "allow_weight_download": False,
        }
    )
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
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["partition"]].append(row)
    report = {
        "validation_kind": "GROUPED_CLIP_LEVEL_INTERNAL_VALIDATION",
        "formal_generalization_claim": False,
        "weights_sha256": file_sha256(args.weights),
        "model_variant": args.model_variant,
        "thresholds": {
            "confidence_threshold": config.detector.confidence_threshold,
            **config.temporal.model_dump(mode="json"),
            **config.tracking.model_dump(mode="json"),
        },
        "protocol": manifest["protocol"],
        "metrics_by_partition": {key: metrics(value) for key, value in grouped.items()},
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
