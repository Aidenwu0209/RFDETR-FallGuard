from __future__ import annotations

import json

import numpy as np
import pytest

from fallguard.factory import build_pipeline
from fallguard.mock_run import run_mock_vertical_slice
from fallguard.schemas import FrameMetadata, MotionState, TrackedDetection

pytestmark = pytest.mark.integration


def test_mock_vertical_slice_creates_event_keyframes_semantics_and_alert(
    development_config,
    tmp_path,
) -> None:
    result = run_mock_vertical_slice(development_config, tmp_path)
    assert len(result.events) == 1
    event = result.events[0]
    assert {item.role.value for item in event.keyframes} == {"before", "during", "after"}
    assert all(item.person_crop and item.person_crop.path.is_file() for item in event.keyframes)
    assert len(result.semantic_assessments) == 1
    assert result.semantic_assessments[0].provider == "mock"
    assert result.semantic_assessments[0].ground_truth_verified is False
    assert len(result.alerts) == 1 and result.alerts[0].should_alert
    serialized = json.loads(event.model_dump_json())
    assert "image_bgr" not in serialized


def test_temporal_ablation_stops_before_semantic_and_alert(development_config, tmp_path) -> None:
    result = run_mock_vertical_slice(development_config, tmp_path, ablation_mode="temporal")
    assert result.events
    assert result.semantic_assessments == []
    assert result.alerts == []


def test_keyframe_privacy_config_can_retain_full_and_person_references(
    development_config,
    tmp_path,
) -> None:
    config = development_config.model_copy(deep=True)
    config.privacy.retain_full_frames = True
    result = run_mock_vertical_slice(config, tmp_path)
    assert result.events[0].keyframes
    assert all(
        item.full_frame and item.full_frame.path.is_file() for item in result.events[0].keyframes
    )
    assert all(
        item.person_crop and item.person_crop.path.is_file() for item in result.events[0].keyframes
    )


def test_empty_frame_expires_candidate_track_with_auditable_reason(
    development_config,
    tmp_path,
) -> None:
    pipeline = build_pipeline(
        development_config,
        with_real_frontend=False,
        keyframe_output_dir=tmp_path,
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    first_meta = FrameMetadata(
        frame_id=0,
        timestamp_seconds=0,
        frame_width=100,
        frame_height=100,
        source_id="source",
        session_id="session",
    )
    falling = TrackedDetection(
        frame_id=0,
        timestamp_seconds=0,
        bbox_xyxy=(5, 50, 95, 90),
        frame_width=100,
        frame_height=100,
        class_id=1,
        class_name="falling",
        confidence=0.9,
        source_id="source",
        session_id="session",
        track_id=1,
    )
    pipeline.process_tracked(image, first_meta, [falling])
    empty_meta = first_meta.model_copy(update={"frame_id": 1, "timestamp_seconds": 1.2})
    result = pipeline.process_tracked(image, empty_meta, [])
    assert any(
        item.next_state == MotionState.RESOLVED and "timed out" in item.reason
        for item in result.transitions
    )
