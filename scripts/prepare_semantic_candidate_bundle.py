#!/usr/bin/env python3
"""Extract three-frame semantic-review bundles from grouped temporal candidate reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from fallguard.config import load_config
from fallguard.detection.rfdetr_adapter import RFDETRDetector
from fallguard.schemas import FrameMetadata, ImageRef
from fallguard.session import make_session_id


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_frame(capture: cv2.VideoCapture, frame_id: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise ValueError(f"cannot read frame {frame_id}")
    return frame


def save_image(
    path: Path,
    image_bgr: np.ndarray,
    kind: Literal["full_frame", "person_crop"],
) -> ImageRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"failed to write image: {path}")
    height, width = image_bgr.shape[:2]
    return ImageRef(
        path=path.resolve(),
        sha256=file_sha256(path),
        width=width,
        height=height,
        kind=kind,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, action="append", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--offset-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.offset_seconds <= 0:
        parser.error("--offset-seconds must be positive")

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.report]
    fingerprints = {report.get("pipeline_implementation_sha256") for report in reports}
    config_hashes = {report.get("config_sha256") for report in reports}
    if len(fingerprints) != 1 or len(config_hashes) != 1:
        parser.error("all reports must use one implementation fingerprint and config")
    report_rows = {
        str(row["video_id"]): row
        for report in reports
        for row in report.get("rows", [])
        if row.get("predicted_fall") is True
    }
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = {str(record["video_id"]): record for record in manifest.get("records", [])}
    missing = sorted(set(report_rows) - set(records))
    if missing:
        parser.error(f"report videos missing from manifest: {missing}")

    config = load_config(args.config)
    detector_config = config.detector.model_copy(
        update={"weights_path": args.weights.resolve(), "allow_weight_download": False}
    )
    detector = RFDETRDetector(detector_config, device=config.runtime.device)
    detector.load()
    candidates: list[dict[str, Any]] = []
    try:
        for video_id, row in sorted(report_rows.items()):
            record = records[video_id]
            video = args.dataset_root / str(record["relative_path"])
            fps = float(record["fps"])
            total_frames = int(record["frames"])
            event_starts = row.get("event_start_seconds") or [row["first_event_start_seconds"]]
            for event_index, raw_event_seconds in enumerate(event_starts, start=1):
                event_seconds = float(raw_event_seconds)
                target_seconds = {
                    "before": max(0.0, event_seconds - args.offset_seconds),
                    "during": event_seconds,
                    "after": min((total_frames - 1) / fps, event_seconds + args.offset_seconds),
                }
                capture = cv2.VideoCapture(str(video))
                if not capture.isOpened():
                    raise ValueError(f"cannot open video: {video}")
                try:
                    frames = {
                        role: (
                            min(total_frames - 1, max(0, round(timestamp * fps))),
                            read_frame(
                                capture,
                                min(total_frames - 1, max(0, round(timestamp * fps))),
                            ),
                        )
                        for role, timestamp in target_seconds.items()
                    }
                finally:
                    capture.release()
                candidate_dir = (
                    args.output_dir / video_id.replace("/", "__") / f"event-{event_index}"
                )
                image_sets: dict[str, dict[str, Any]] = {}
                selected_by_role: dict[str, Any] = {}
                for role, (frame_id, frame) in frames.items():
                    height, width = frame.shape[:2]
                    metadata = FrameMetadata(
                        frame_id=frame_id,
                        timestamp_seconds=frame_id / fps,
                        frame_width=width,
                        frame_height=height,
                        source_id=video_id,
                        session_id=make_session_id(f"{video_id}-{event_index}-{role}"),
                    )
                    detections = detector.predict_frame(frame, metadata)
                    selected = max(detections, key=lambda item: item.confidence, default=None)
                    selected_by_role[role] = (
                        selected.model_dump(mode="json") if selected is not None else None
                    )
                    full = save_image(candidate_dir / f"{role}-full.jpg", frame, "full_frame")
                    crop = None
                    if selected is not None:
                        x1, y1, x2, y2 = (round(value) for value in selected.bbox_xyxy)
                        person = frame[y1:y2, x1:x2]
                        if person.size:
                            crop = save_image(
                                candidate_dir / f"{role}-person.jpg", person, "person_crop"
                            )
                    image_sets[role] = {
                        "frame_id": frame_id,
                        "timestamp_seconds": frame_id / fps,
                        "full_frame": full.model_dump(mode="json"),
                        "person_crop": (crop.model_dump(mode="json") if crop is not None else None),
                    }
                candidates.append(
                    {
                        "candidate_id": f"{video_id}#event-{event_index}",
                        "video_id": video_id,
                        "subject_id": record["subject_id"],
                        "clip_label": record["label"],
                        "expected_semantic_decision_weak": (
                            "fall" if record["label"] == "fall" else "not_fall"
                        ),
                        "label_provenance": "clip_level_weak_not_event_confirmed",
                        "human_review_status": "pending",
                        "event_start_seconds": event_seconds,
                        "temporal_predicted_event_count": row["predicted_event_count"],
                        "crop_detections": selected_by_role,
                        "images": image_sets,
                    }
                )
    finally:
        detector.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "semantic-candidate-manifest.json"
    bundle = {
        "manifest_kind": "SEMANTIC_CANDIDATE_BUNDLE",
        "formal_ground_truth": False,
        "human_confirmation_required_before_qlora_or_formal_claim": True,
        "source_report_sha256": [file_sha256(path) for path in args.report],
        "source_manifest_sha256": file_sha256(args.manifest),
        "weights_sha256": file_sha256(args.weights),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    output.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "candidate_count": len(candidates)}, indent=2))


if __name__ == "__main__":
    main()
