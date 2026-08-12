from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fallguard.events.manager import EventManager
from fallguard.schemas import (
    Detection,
    EventStatus,
    FrameMetadata,
    MotionState,
    TemporalFeatures,
    TrackedDetection,
    TransitionRecord,
)
from fallguard.temporal.features import TemporalFeatureExtractor
from fallguard.temporal.state_machine import TemporalStateMachine
from fallguard.tracking.bytetrack_adapter import ByteTrackAdapter
from fallguard.tracking.manager import TrackManager

pytestmark = pytest.mark.unit


def detection(class_name: str, class_id: int, *, session: str = "session") -> Detection:
    return Detection(
        frame_id=0,
        timestamp_seconds=0,
        bbox_xyxy=(10, 10, 30, 90),
        frame_width=100,
        frame_height=100,
        class_id=class_id,
        class_name=class_name,
        confidence=0.9,
        source_id="source",
        session_id=session,
    )


def tracked(frame: int, timestamp: float, box: tuple[float, float, float, float], name: str):
    return TrackedDetection(
        frame_id=frame,
        timestamp_seconds=timestamp,
        bbox_xyxy=box,
        frame_width=100,
        frame_height=100,
        class_id={"standing": 0, "falling": 1, "lying": 2}[name],
        class_name=name,
        confidence=0.9,
        source_id="source",
        session_id="session",
        track_id=7,
    )


class FakeTracker:
    def __init__(self) -> None:
        self.last_input: Any = None

    def update_with_detections(self, detections: Any) -> Any:
        self.last_input = detections
        return SimpleNamespace(tracker_id=np.asarray([9, 10]), data=detections.data)

    def reset(self) -> None:
        self.last_input = None


class FakeDetections:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def test_bytetrack_input_is_class_agnostic_but_observations_keep_posture(
    development_config,
) -> None:
    tracker = FakeTracker()
    adapter = ByteTrackAdapter(
        development_config.tracking,
        tracker=tracker,
        detections_factory=FakeDetections,
    )
    result = adapter.update([detection("standing", 3), detection("falling", 8)])
    assert tracker.last_input.class_id.tolist() == [0, 0]
    assert [item.class_name for item in result] == ["standing", "falling"]
    assert [item.track_id for item in result] == [9, 10]


def test_track_manager_preserves_identity_across_posture_change() -> None:
    manager = TrackManager(max_history=4)
    first = tracked(0, 0.0, (40, 10, 60, 90), "standing")
    second = tracked(1, 0.2, (20, 40, 90, 90), "falling")
    manager.update([first])
    track = manager.update([second])[0]
    assert track.track_id == 7
    assert [item.class_name for item in track.observations] == ["standing", "falling"]


@pytest.mark.integration
def test_pinned_real_bytetrack_keeps_id_when_posture_class_changes(
    development_config,
) -> None:
    pytest.importorskip("supervision", reason="optional pinned tracking extra is not installed")
    adapter = ByteTrackAdapter(development_config.tracking)
    adapter.load()
    first = detection("standing", 3)
    second = detection("falling", 8).model_copy(
        update={
            "frame_id": 1,
            "timestamp_seconds": 1 / development_config.tracking.frame_rate,
            "bbox_xyxy": (11, 10, 31, 90),
        }
    )
    first_result = adapter.update([first])
    second_result = adapter.update([second])
    assert first_result and second_result
    assert first_result[0].track_id == second_result[0].track_id
    assert second_result[0].class_name == "falling"


@pytest.mark.integration
def test_pinned_real_bytetrack_recovers_id_after_short_detection_gap(
    development_config,
) -> None:
    pytest.importorskip("supervision", reason="optional pinned tracking extra is not installed")
    adapter = ByteTrackAdapter(development_config.tracking)
    adapter.load()
    first = detection("standing", 3)
    first_result = adapter.update([first])
    missing_frame = FrameMetadata(
        frame_id=1,
        timestamp_seconds=1 / development_config.tracking.frame_rate,
        frame_width=100,
        frame_height=100,
        source_id="source",
        session_id="session",
    )
    assert adapter.update([], missing_frame) == []
    reappeared = detection("falling", 8).model_copy(
        update={
            "frame_id": 2,
            "timestamp_seconds": 2 / development_config.tracking.frame_rate,
            "bbox_xyxy": (11, 10, 31, 90),
        }
    )
    reappeared_result = adapter.update([reappeared])
    assert first_result and reappeared_result
    assert first_result[0].track_id == reappeared_result[0].track_id


def test_temporal_features_use_elapsed_seconds_and_downward_positive() -> None:
    extractor = TemporalFeatureExtractor(smoothing_window=2)
    first = tracked(0, 0.0, (40, 10, 60, 90), "standing")
    second = tracked(1, 0.5, (40, 30, 60, 90), "falling")
    features = extractor.compute(first, second)
    assert features.center_dy_pixels == 10
    assert features.vertical_speed_pixels_per_second == 20
    assert features.vertical_speed_frame_height_per_second == pytest.approx(0.2)


def feature(frame: int, timestamp: float, ratio: float, speed: float, posture: str):
    return TemporalFeatures(
        frame_id=frame,
        timestamp_seconds=timestamp,
        aspect_ratio_width_over_height=ratio,
        center_dx_pixels=0,
        center_dy_pixels=0,
        center_dx_frame_width=0,
        center_dy_frame_height=0,
        center_dy_person_height=0,
        vertical_speed_pixels_per_second=speed * 100,
        vertical_speed_frame_height_per_second=speed,
        posture_class_name=posture,
        posture_confidence=0.9,
        elapsed_seconds=0.2,
    )


def transition(
    frame: int,
    timestamp: float,
    previous: MotionState,
    next_state: MotionState,
) -> TransitionRecord:
    return TransitionRecord(
        track_id=7,
        source_id="source",
        session_id="session",
        frame_id=frame,
        timestamp_seconds=timestamp,
        previous_state=previous,
        next_state=next_state,
        reason=f"test {previous.value} to {next_state.value}",
    )


def test_state_machine_and_event_manager_do_not_duplicate_continuous_event(
    development_config,
) -> None:
    machine = TemporalStateMachine(
        development_config.temporal,
        development_config.detector.posture_groups,
    )
    events = EventManager(development_config.event)
    observations = [
        (tracked(0, 0.0, (40, 10, 60, 90), "standing"), feature(0, 0, 0.25, 0, "standing")),
        (tracked(1, 0.2, (20, 30, 90, 90), "falling"), feature(1, 0.2, 1.2, 0.4, "falling")),
        (tracked(2, 0.4, (15, 50, 95, 90), "falling"), feature(2, 0.4, 2.0, 0.2, "falling")),
        (tracked(3, 0.6, (10, 60, 95, 90), "falling"), feature(3, 0.6, 2.8, 0.1, "falling")),
        (tracked(4, 0.8, (10, 60, 95, 90), "lying"), feature(4, 0.8, 2.8, 0, "lying")),
    ]
    transitions = []
    for observation, features in observations:
        transition = machine.update(observation, features)
        if transition:
            transitions.append(transition)
            events.on_transition(transition)
    assert [item.next_state for item in transitions] == [
        MotionState.SUSPECTED,
        MotionState.FALLING,
        MotionState.LYING,
    ]
    assert len(events.all_events()) == 1
    assert "lying_started_at_seconds" in events.all_events()[0].metadata


def test_static_lying_or_wide_box_cannot_start_fall_candidate(development_config) -> None:
    machine = TemporalStateMachine(
        development_config.temporal,
        development_config.detector.posture_groups,
    )
    lying = tracked(0, 0.0, (10, 60, 95, 90), "lying")
    assert machine.update(lying, feature(0, 0.0, 2.8, 0.0, "lying")) is None
    assert machine.state_for(lying) == MotionState.UPRIGHT

    wide_upright = tracked(1, 0.2, (10, 60, 95, 90), "standing")
    assert machine.update(wide_upright, feature(1, 0.2, 2.8, 0.0, "standing")) is None
    assert machine.state_for(wide_upright) == MotionState.UPRIGHT


def test_high_recall_profile_can_confirm_explicit_lying_posture(development_config) -> None:
    temporal = development_config.temporal.model_copy(
        update={
            "candidate_on_lying_posture": True,
            "confirm_on_falling_posture": True,
            "confirm_on_lying_posture": True,
        }
    )
    machine = TemporalStateMachine(temporal, development_config.detector.posture_groups)
    events = EventManager(development_config.event)
    observation = tracked(0, 0.0, (10, 60, 95, 90), "lying")

    state_transition = machine.update(observation, feature(0, 0.0, 2.8, 0.0, "lying"))

    assert state_transition is not None
    assert state_transition.next_state == MotionState.LYING
    event = events.on_transition(state_transition)
    assert event is not None
    assert event.start_frame == 0
    assert event.metadata["lying_started_at_seconds"] == 0.0


def test_high_recall_profile_can_confirm_explicit_falling_posture(development_config) -> None:
    temporal = development_config.temporal.model_copy(update={"confirm_on_falling_posture": True})
    machine = TemporalStateMachine(temporal, development_config.detector.posture_groups)
    observation = tracked(0, 0.0, (20, 30, 90, 90), "falling")

    state_transition = machine.update(observation, feature(0, 0.0, 1.2, 0.0, "falling"))

    assert state_transition is not None
    assert state_transition.next_state == MotionState.FALLING
    assert "immediately confirmed" in state_transition.reason


def test_downward_speed_can_start_candidate_without_falling_class(development_config) -> None:
    machine = TemporalStateMachine(
        development_config.temporal,
        development_config.detector.posture_groups,
    )
    observation = tracked(0, 0.0, (40, 10, 60, 90), "standing")
    transition = machine.update(observation, feature(0, 0.0, 0.25, 0.4, "standing"))
    assert transition is not None
    assert transition.next_state == MotionState.SUSPECTED
    assert "downward speed" in transition.reason


def test_event_manager_merges_nearby_episode_and_suppresses_cooldown(
    development_config,
) -> None:
    manager = EventManager(development_config.event)
    assert (
        manager.on_transition(transition(0, 0.0, MotionState.UPRIGHT, MotionState.SUSPECTED))
        is None
    )
    first = manager.on_transition(transition(1, 0.5, MotionState.SUSPECTED, MotionState.FALLING))
    assert first is not None
    manager.on_transition(transition(2, 1.0, MotionState.RECOVERING, MotionState.RESOLVED))
    assert (
        manager.on_transition(transition(3, 2.0, MotionState.RESOLVED, MotionState.SUSPECTED))
        is None
    )
    merged = manager.on_transition(transition(4, 2.5, MotionState.SUSPECTED, MotionState.FALLING))
    assert merged is not None and merged.event_id == first.event_id
    assert merged.status == EventStatus.ACTIVE
    manager.on_transition(transition(5, 3.0, MotionState.RECOVERING, MotionState.RESOLVED))
    assert (
        manager.on_transition(transition(6, 6.0, MotionState.RESOLVED, MotionState.SUSPECTED))
        is None
    )
    suppressed = manager.on_transition(
        transition(7, 6.5, MotionState.SUSPECTED, MotionState.FALLING)
    )
    assert suppressed is not None and suppressed.event_id == first.event_id
    assert suppressed.metadata["cooldown_suppressed_candidate_at"] == 6.0
    assert manager.active_events() == []


def test_event_manager_timeout_closes_coupled_end_fields(development_config) -> None:
    manager = EventManager(development_config.event)
    assert (
        manager.on_transition(transition(0, 0.0, MotionState.UPRIGHT, MotionState.SUSPECTED))
        is None
    )
    event = manager.on_transition(transition(1, 0.5, MotionState.SUSPECTED, MotionState.FALLING))
    assert event is not None
    timed_out = manager.tick(development_config.event.timeout_seconds + 0.1, frame_id=99)
    assert len(timed_out) == 1
    assert timed_out[0].status == EventStatus.TIMED_OUT
    assert timed_out[0].end_frame == 99
    assert timed_out[0].end_time == pytest.approx(development_config.event.timeout_seconds + 0.1)


def test_event_manager_discards_candidate_that_never_reaches_falling(
    development_config,
) -> None:
    manager = EventManager(development_config.event)
    assert (
        manager.on_transition(transition(0, 0.0, MotionState.UPRIGHT, MotionState.SUSPECTED))
        is None
    )
    assert (
        manager.on_transition(transition(1, 0.2, MotionState.SUSPECTED, MotionState.UPRIGHT))
        is None
    )
    assert manager.all_events() == []
