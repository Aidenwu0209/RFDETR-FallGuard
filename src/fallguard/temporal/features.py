"""Geometry features with explicit image-coordinate units."""

from __future__ import annotations

from collections import defaultdict, deque

from fallguard.exceptions import NonMonotonicTimestampError
from fallguard.schemas import TemporalFeatures, TrackedDetection

TrackKey = tuple[str, str, int]


class TemporalFeatureExtractor:
    def __init__(self, *, smoothing_window: int = 5) -> None:
        if smoothing_window <= 0:
            raise ValueError("smoothing_window must be positive")
        self.smoothing_window = smoothing_window
        self._history: dict[TrackKey, deque[TemporalFeatures]] = defaultdict(
            lambda: deque(maxlen=smoothing_window)
        )

    def compute(
        self,
        previous: TrackedDetection,
        current: TrackedDetection,
    ) -> TemporalFeatures:
        if (previous.source_id, previous.session_id, previous.track_id) != (
            current.source_id,
            current.session_id,
            current.track_id,
        ):
            raise ValueError("temporal features require observations from the same scoped track")
        elapsed = current.timestamp_seconds - previous.timestamp_seconds
        if elapsed <= 0:
            raise NonMonotonicTimestampError("temporal speed requires increasing timestamps")
        previous_center = previous.center
        current_center = current.center
        dx = current_center[0] - previous_center[0]
        dy = current_center[1] - previous_center[1]
        feature = TemporalFeatures(
            frame_id=current.frame_id,
            timestamp_seconds=current.timestamp_seconds,
            aspect_ratio_width_over_height=current.width / current.height,
            center_dx_pixels=dx,
            center_dy_pixels=dy,
            center_dx_frame_width=dx / current.frame_width,
            center_dy_frame_height=dy / current.frame_height,
            center_dy_person_height=dy / max(previous.height, 1.0),
            vertical_speed_pixels_per_second=dy / elapsed,
            vertical_speed_frame_height_per_second=(dy / current.frame_height) / elapsed,
            posture_class_name=current.class_name,
            posture_confidence=current.confidence,
            elapsed_seconds=elapsed,
        )
        key = (current.source_id, current.session_id, current.track_id)
        self._history[key].append(feature)
        return feature

    def smoothed(self, feature: TemporalFeatures, key: TrackKey) -> TemporalFeatures:
        history = list(self._history[key])
        if not history:
            return feature
        numeric = (
            "aspect_ratio_width_over_height",
            "center_dx_pixels",
            "center_dy_pixels",
            "center_dx_frame_width",
            "center_dy_frame_height",
            "center_dy_person_height",
            "vertical_speed_pixels_per_second",
            "vertical_speed_frame_height_per_second",
        )
        updates = {
            name: sum(float(getattr(item, name)) for item in history) / len(history)
            for name in numeric
        }
        return feature.model_copy(update=updates)


def initial_feature(observation: TrackedDetection) -> TemporalFeatures:
    """Create a zero-motion first observation without pretending a frame-based speed."""
    return TemporalFeatures(
        frame_id=observation.frame_id,
        timestamp_seconds=observation.timestamp_seconds,
        aspect_ratio_width_over_height=observation.width / observation.height,
        center_dx_pixels=0.0,
        center_dy_pixels=0.0,
        center_dx_frame_width=0.0,
        center_dy_frame_height=0.0,
        center_dy_person_height=0.0,
        vertical_speed_pixels_per_second=0.0,
        vertical_speed_frame_height_per_second=0.0,
        posture_class_name=observation.class_name,
        posture_confidence=observation.confidence,
        elapsed_seconds=1e-9,
    )
