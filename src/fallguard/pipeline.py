"""End-to-end orchestration with an explicit mock-free production surface."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from fallguard.alert import AlertManager
from fallguard.detection.base import Detector
from fallguard.events.keyframes import BoundedFrameBuffer, BufferedFrame
from fallguard.events.manager import EventManager
from fallguard.exceptions import FallGuardError
from fallguard.schemas import (
    AlertDecision,
    Detection,
    FallEvent,
    FrameMetadata,
    MotionState,
    SemanticAssessment,
    SemanticReviewRequest,
    TrackedDetection,
    TransitionRecord,
)
from fallguard.semantic.router import SemanticReviewRouter
from fallguard.temporal.features import TemporalFeatureExtractor, initial_feature
from fallguard.temporal.state_machine import TemporalStateMachine
from fallguard.timing import TimingCollector
from fallguard.tracking.bytetrack_adapter import ByteTrackAdapter
from fallguard.tracking.manager import TrackManager

LOGGER = logging.getLogger(__name__)
AblationMode = Literal["detector_only", "detector_tracking", "temporal", "full"]


@dataclass
class PipelineResult:
    detections: list[Detection] = field(default_factory=list)
    tracked_detections: list[TrackedDetection] = field(default_factory=list)
    transitions: list[TransitionRecord] = field(default_factory=list)
    events: list[FallEvent] = field(default_factory=list)
    semantic_assessments: list[SemanticAssessment] = field(default_factory=list)
    alerts: list[AlertDecision] = field(default_factory=list)


class FallGuardPipeline:
    component_kind = "real"

    def __init__(
        self,
        *,
        detector: Detector | None,
        tracker: ByteTrackAdapter | None,
        track_manager: TrackManager,
        feature_extractor: TemporalFeatureExtractor,
        state_machine: TemporalStateMachine,
        event_manager: EventManager,
        semantic_router: SemanticReviewRouter,
        alert_manager: AlertManager,
        frame_buffer: BoundedFrameBuffer,
        keyframe_output_dir: str | Path,
        cloud_image_consent: bool = False,
        ablation_mode: AblationMode = "full",
        timing: TimingCollector | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.track_manager = track_manager
        self.feature_extractor = feature_extractor
        self.state_machine = state_machine
        self.event_manager = event_manager
        self.semantic_router = semantic_router
        self.alert_manager = alert_manager
        self.frame_buffer = frame_buffer
        self.keyframe_output_dir = Path(keyframe_output_dir)
        self.cloud_image_consent = cloud_image_consent
        self.ablation_mode = ablation_mode
        self.timing = timing or TimingCollector()
        self._reviewed_event_ids: set[str] = set()

    def process_frame(self, image_bgr: np.ndarray, metadata: FrameMetadata) -> PipelineResult:
        if self.detector is None or self.tracker is None:
            raise RuntimeError("real frame processing requires detector and tracker")
        with self.timing.time("detection"):
            detections = self.detector.predict_frame(image_bgr, metadata)
        if self.ablation_mode == "detector_only":
            return PipelineResult(detections=detections)
        with self.timing.time("tracking"):
            tracked = self.tracker.update(detections, metadata)
        if self.ablation_mode == "detector_tracking":
            return PipelineResult(detections=detections, tracked_detections=tracked)
        result = self.process_tracked(image_bgr, metadata, tracked)
        result.detections = detections
        return result

    def process_tracked(
        self,
        image_bgr: np.ndarray,
        metadata: FrameMetadata,
        tracked: list[TrackedDetection],
    ) -> PipelineResult:
        result = PipelineResult(tracked_detections=tracked)
        if self.ablation_mode in {"detector_only", "detector_tracking"}:
            return result
        maximum_motion = 0.0
        first_bbox = tracked[0].bbox_xyxy if tracked else None
        with self.timing.time("temporal"):
            tracks = self.track_manager.update(tracked)
            for track in tracks:
                current = track.observations[-1]
                if len(track.observations) == 1:
                    features = initial_feature(current)
                else:
                    features = self.feature_extractor.compute(track.observations[-2], current)
                    key = (current.source_id, current.session_id, current.track_id)
                    features = self.feature_extractor.smoothed(features, key)
                maximum_motion = max(
                    maximum_motion,
                    abs(features.vertical_speed_frame_height_per_second),
                )
                transition = self.state_machine.update(current, features)
                if transition is not None:
                    result.transitions.append(transition)
                    event = self.event_manager.on_transition(transition)
                    if event is not None:
                        result.events.append(event)
            for expired in self.track_manager.expire(
                metadata.timestamp_seconds,
                self.state_machine.track_timeout_seconds,
            ):
                transition = self.state_machine.expire(
                    source_id=expired.source_id,
                    session_id=expired.session_id,
                    track_id=expired.track_id,
                    frame_id=metadata.frame_id,
                    now_seconds=metadata.timestamp_seconds,
                )
                if transition is not None:
                    result.transitions.append(transition)
                    event = self.event_manager.on_transition(transition)
                    if event is not None:
                        result.events.append(event)
        self.frame_buffer.add(
            BufferedFrame(
                metadata=metadata,
                image_bgr=image_bgr,
                person_bbox_xyxy=first_bbox,
                motion_score=maximum_motion,
            )
        )
        for event in self.event_manager.tick(metadata.timestamp_seconds, metadata.frame_id):
            result.events.append(event)
        if self.ablation_mode == "full":
            self._review_ready_events(metadata.timestamp_seconds, result)
        return result

    def _review_ready_events(self, now_seconds: float, result: PipelineResult) -> None:
        candidates = self.event_manager.all_events()
        for event in candidates:
            if event.event_id in self._reviewed_event_ids:
                continue
            lying_at = event.metadata.get("lying_started_at_seconds")
            ready = (
                lying_at is not None
                and now_seconds
                >= float(lying_at) + self.frame_buffer.event_config.after_offset_seconds
            )
            ready = ready or event.end_time is not None
            if not ready:
                continue
            with self.timing.time("keyframe"):
                keyframes = self.frame_buffer.select_and_persist(event, self.keyframe_output_dir)
                event = self.event_manager.add_keyframes(event.event_id, keyframes)
            request = SemanticReviewRequest(
                event=event,
                text_context=self._event_context(event),
                image_refs=[
                    reference
                    for item in event.keyframes
                    for reference in (item.person_crop or item.full_frame,)
                    if reference is not None
                ],
                cloud_image_consent=self.cloud_image_consent,
            )
            semantic: SemanticAssessment | None = None
            try:
                with self.timing.time("semantic"):
                    semantic = self.semantic_router.review(request)
                result.semantic_assessments.append(semantic)
            except FallGuardError as exc:
                LOGGER.warning(
                    "semantic review unavailable: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"event_id": event.event_id, "stage": "semantic"},
                )
            temporal_state = self._event_state(event)
            with self.timing.time("alert"):
                alert = self.alert_manager.decide(event, temporal_state, semantic)
                result.alerts.append(alert)
            self.timing.record(
                "event_trigger_video_time",
                max(0.0, now_seconds - event.start_time) * 1000.0,
            )
            self._reviewed_event_ids.add(event.event_id)

    @staticmethod
    def _event_context(event: FallEvent) -> str:
        return json.dumps(
            {
                "event_id": event.event_id,
                "start_time_seconds": event.start_time,
                "end_time_seconds": event.end_time,
                "status": event.status,
                "transition_reasons": event.transition_reasons,
                "keyframe_roles": [item.role for item in event.keyframes],
            },
            ensure_ascii=False,
            default=str,
        )

    def _event_state(self, event: FallEvent) -> MotionState:
        track = self.track_manager.get(event.source_id, event.session_id, event.track_id)
        if track is None or not track.observations:
            return MotionState.RESOLVED
        return self.state_machine.state_for(track.observations[-1])


def append_jsonl(path: str | Path, records: list[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        for record in records:
            if hasattr(record, "model_dump_json"):
                stream.write(record.model_dump_json() + "\n")
            else:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
