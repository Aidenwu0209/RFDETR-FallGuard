"""Bounded online frame buffer and auditable before/during/after selection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fallguard.config import EventConfig, PrivacyConfig
from fallguard.schemas import (
    FallEvent,
    FrameMetadata,
    ImageRef,
    KeyframeRole,
    KeyframeSelection,
)
from fallguard.video import write_image


@dataclass(frozen=True)
class BufferedFrame:
    metadata: FrameMetadata
    image_bgr: np.ndarray
    person_bbox_xyxy: tuple[float, float, float, float] | None = None
    motion_score: float = 0.0


class BoundedFrameBuffer:
    def __init__(self, event_config: EventConfig, privacy_config: PrivacyConfig) -> None:
        self.event_config = event_config
        self.privacy_config = privacy_config
        self._frames: deque[BufferedFrame] = deque(maxlen=event_config.ring_buffer_frames)

    def add(self, frame: BufferedFrame) -> None:
        if (
            self._frames
            and frame.metadata.timestamp_seconds <= self._frames[-1].metadata.timestamp_seconds
        ):
            raise ValueError("buffered frame timestamps must strictly increase")
        self._frames.append(frame)

    def select_and_persist(
        self, event: FallEvent, output_dir: str | Path
    ) -> list[KeyframeSelection]:
        if not self._frames:
            return []
        anchor_value = event.metadata.get(
            "lying_started_at_seconds",
            event.end_time if event.end_time is not None else event.start_time,
        )
        event_anchor = (
            float(anchor_value) if isinstance(anchor_value, int | float) else event.start_time
        )
        targets = {
            KeyframeRole.BEFORE: max(
                0.0, event.start_time - self.event_config.before_offset_seconds
            ),
            KeyframeRole.DURING: event_anchor,
            KeyframeRole.AFTER: event_anchor + self.event_config.after_offset_seconds,
        }
        selected: list[KeyframeSelection] = []
        used_frames: set[int] = set()
        for role, target in targets.items():
            candidates = [
                item for item in self._frames if item.metadata.frame_id not in used_frames
            ]
            if not candidates:
                break
            if role == KeyframeRole.DURING:
                item = max(candidates, key=lambda candidate: candidate.motion_score)
                reason = "maximum buffered motion score during candidate event"
            else:
                item = min(
                    candidates,
                    key=lambda candidate: abs(candidate.metadata.timestamp_seconds - target),
                )
                reason = f"nearest buffered timestamp to {role.value} target {target:.3f}s"
            used_frames.add(item.metadata.frame_id)
            full_ref, crop_ref = self._persist(event, role, item, Path(output_dir))
            selected.append(
                KeyframeSelection(
                    frame_id=item.metadata.frame_id,
                    timestamp_seconds=item.metadata.timestamp_seconds,
                    role=role,
                    reason=reason,
                    score=item.motion_score,
                    features={"target_timestamp_seconds": target},
                    full_frame=full_ref,
                    person_crop=crop_ref,
                )
            )
        return selected

    def _persist(
        self,
        event: FallEvent,
        role: KeyframeRole,
        item: BufferedFrame,
        output_dir: Path,
    ) -> tuple[ImageRef | None, ImageRef | None]:
        base = output_dir / event.event_id
        full_ref = None
        if self.privacy_config.retain_full_frames:
            full_path = base / f"{role.value}-frame-{item.metadata.frame_id}-full.jpg"
            digest = write_image(full_path, item.image_bgr)
            height, width = item.image_bgr.shape[:2]
            full_ref = ImageRef(
                path=full_path,
                sha256=digest,
                width=width,
                height=height,
                kind="full_frame",
            )
        crop_ref = None
        if self.privacy_config.retain_person_crops and item.person_bbox_xyxy is not None:
            x1, y1, x2, y2 = (round(value) for value in item.person_bbox_xyxy)
            height, width = item.image_bgr.shape[:2]
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 > x1 and y2 > y1:
                crop = item.image_bgr[y1:y2, x1:x2]
                crop_path = base / f"{role.value}-frame-{item.metadata.frame_id}-person.jpg"
                digest = write_image(crop_path, crop)
                crop_ref = ImageRef(
                    path=crop_path,
                    sha256=digest,
                    width=x2 - x1,
                    height=y2 - y1,
                    kind="person_crop",
                )
        return full_ref, crop_ref
