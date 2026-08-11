"""Synthetic, explicitly MOCK vertical-slice input; no fake detector enters production."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fallguard.config import AppConfig
from fallguard.factory import build_pipeline
from fallguard.pipeline import AblationMode, PipelineResult
from fallguard.schemas import FrameMetadata, TrackedDetection


def run_mock_vertical_slice(
    config: AppConfig,
    output_dir: str | Path,
    *,
    ablation_mode: AblationMode = "full",
) -> PipelineResult:
    if not config.semantic.allow_mock or config.semantic.provider != "mock":
        raise ValueError("mock vertical slice requires an explicitly enabled mock provider")
    pipeline = build_pipeline(
        config,
        with_real_frontend=False,
        keyframe_output_dir=Path(output_dir) / "keyframes",
        ablation_mode=ablation_mode,
    )
    combined = PipelineResult()
    for frame_id in range(13):
        timestamp = frame_id * 0.2
        if frame_id < 2:
            box = (40.0, 10.0 + frame_id * 2.0, 60.0, 90.0 + frame_id * 2.0)
            posture = "standing"
        elif frame_id < 5:
            top = 20.0 + (frame_id - 2) * 10.0
            box = (25.0, top, 85.0, min(100.0, top + 55.0))
            posture = "falling"
        else:
            box = (10.0, 65.0, 95.0, 95.0)
            posture = "lying"
        metadata = FrameMetadata(
            frame_id=frame_id,
            timestamp_seconds=timestamp,
            frame_width=120,
            frame_height=120,
            source_id="MOCK-source",
            session_id="MOCK-session",
        )
        observation = TrackedDetection(
            frame_id=frame_id,
            timestamp_seconds=timestamp,
            bbox_xyxy=box,
            frame_width=120,
            frame_height=120,
            class_id={"standing": 0, "falling": 1, "lying": 2}[posture],
            class_name=posture,
            confidence=0.99,
            source_id=metadata.source_id,
            session_id=metadata.session_id,
            track_id=1,
        )
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        image[:, :, frame_id % 3] = min(255, 20 * frame_id)
        result = pipeline.process_tracked(image, metadata, [observation])
        combined.tracked_detections.extend(result.tracked_detections)
        combined.transitions.extend(result.transitions)
        combined.events.extend(result.events)
        combined.semantic_assessments.extend(result.semantic_assessments)
        combined.alerts.extend(result.alerts)
    combined.events = pipeline.event_manager.all_events()
    return combined
