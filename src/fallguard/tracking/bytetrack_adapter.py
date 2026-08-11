"""Pinned, class-agnostic ByteTrack adapter using Supervision."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from fallguard.config import TrackingConfig
from fallguard.exceptions import ConfigurationError, DependencyUnavailableError
from fallguard.schemas import Detection, FrameMetadata, TrackedDetection

PINNED_SUPERVISION_VERSION = "0.30.0"


class ByteTrackAdapter:
    component_kind = "real"

    def __init__(
        self,
        config: TrackingConfig,
        *,
        tracker: Any | None = None,
        detections_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._tracker = tracker
        self._detections_factory = detections_factory
        self._scope: tuple[str, str] | None = None

    def load(self) -> None:
        if self._tracker is not None:
            return
        try:
            installed = version("supervision")
        except PackageNotFoundError as exc:
            raise DependencyUnavailableError(
                "ByteTrack is optional; install the pinned integration with .[tracking]"
            ) from exc
        if installed != PINNED_SUPERVISION_VERSION:
            raise DependencyUnavailableError(
                f"unsupported supervision version {installed}; "
                f"expected {PINNED_SUPERVISION_VERSION}"
            )
        import supervision as sv

        self._tracker = sv.ByteTrack(
            track_activation_threshold=self.config.track_activation_threshold,
            lost_track_buffer=self.config.lost_track_buffer,
            minimum_matching_threshold=self.config.minimum_matching_threshold,
            frame_rate=self.config.frame_rate,
            minimum_consecutive_frames=self.config.minimum_consecutive_frames,
        )

    def update(
        self,
        detections: list[Detection],
        metadata: FrameMetadata | None = None,
    ) -> list[TrackedDetection]:
        if self._tracker is None:
            self.load()
        if detections:
            self._validate_single_frame(detections)
            scope = (detections[0].source_id, detections[0].session_id)
        elif metadata is not None:
            scope = (metadata.source_id, metadata.session_id)
        else:
            raise ValueError("metadata is required when advancing ByteTrack with no detections")
        if self._scope is None:
            self._scope = scope
        elif self._scope != scope:
            raise ConfigurationError(
                "ByteTrack scope changed; call reset() between sources/sessions "
                "to avoid ID collisions"
            )

        detections_factory: Callable[..., Any]
        if self._detections_factory is None:
            try:
                import supervision as sv
            except ImportError as exc:
                raise DependencyUnavailableError("supervision could not be imported") from exc
            detections_factory = sv.Detections
        else:
            detections_factory = self._detections_factory
        xyxy = np.asarray([item.bbox_xyxy for item in detections], dtype=float).reshape(-1, 4)
        tracker_input = detections_factory(
            xyxy=xyxy,
            confidence=np.asarray([item.confidence for item in detections], dtype=float),
            class_id=np.zeros(len(detections), dtype=int),
            data={"detection_id": np.asarray([item.detection_id for item in detections])},
        )
        tracker = self._tracker
        if tracker is None:
            raise DependencyUnavailableError("ByteTrack load returned no tracker")
        tracked = tracker.update_with_detections(tracker_input)
        if tracked.tracker_id is None:
            return []
        original = {item.detection_id: item for item in detections}
        ids = tracked.data.get("detection_id", [])
        result: list[TrackedDetection] = []
        for detection_id, tracker_id in zip(ids, tracked.tracker_id, strict=True):
            item = original[str(detection_id)]
            result.append(TrackedDetection(**item.model_dump(), track_id=int(tracker_id)))
        return result

    def reset(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()
        self._scope = None

    @staticmethod
    def _validate_single_frame(detections: list[Detection]) -> None:
        expected = (
            detections[0].frame_id,
            detections[0].timestamp_seconds,
            detections[0].source_id,
            detections[0].session_id,
        )
        for item in detections[1:]:
            actual = (item.frame_id, item.timestamp_seconds, item.source_id, item.session_id)
            if actual != expected:
                raise ValueError("one ByteTrack update may only contain one frame and scope")
