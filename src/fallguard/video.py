"""Bounded video/image utilities; NumPy frames never enter persisted business schemas."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fallguard.exceptions import FallGuardError, NonMonotonicTimestampError
from fallguard.schemas import FrameMetadata


@dataclass(frozen=True)
class FramePacket:
    metadata: FrameMetadata
    image_bgr: np.ndarray


class VideoReader:
    def __init__(self, path: str | Path, *, source_id: str, session_id: str) -> None:
        self.path = Path(path)
        self.source_id = source_id
        self.session_id = session_id
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> VideoReader:
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise FallGuardError(f"cannot open video: {self.path}")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    @property
    def fps(self) -> float:
        if self._capture is None:
            raise RuntimeError("VideoReader must be opened")
        value = float(self._capture.get(cv2.CAP_PROP_FPS))
        return value if value > 0 else 0.0

    def frames(self) -> Iterator[FramePacket]:
        if self._capture is None:
            raise RuntimeError("VideoReader must be opened")
        frame_id = 0
        previous_timestamp = -1.0
        fps = self.fps
        while True:
            ok, image = self._capture.read()
            if not ok:
                break
            timestamp = float(self._capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if timestamp <= previous_timestamp:
                if fps <= 0:
                    raise NonMonotonicTimestampError(
                        "video reports non-monotonic timestamps and no valid FPS fallback"
                    )
                timestamp = frame_id / fps
            if timestamp <= previous_timestamp:
                raise NonMonotonicTimestampError("video timestamps are not strictly increasing")
            height, width = image.shape[:2]
            yield FramePacket(
                metadata=FrameMetadata(
                    frame_id=frame_id,
                    timestamp_seconds=timestamp,
                    frame_width=width,
                    frame_height=height,
                    source_id=self.source_id,
                    session_id=self.session_id,
                ),
                image_bgr=image,
            )
            previous_timestamp = timestamp
            frame_id += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def write_image(path: str | Path, image_bgr: np.ndarray) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image_bgr):
        raise FallGuardError(f"failed to write image: {target}")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def read_frame_at(path: str | Path, frame_id: int) -> np.ndarray:
    """Random-access reread for offline keyframe workflows."""
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise FallGuardError(f"cannot open video: {path}")
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id):
            raise FallGuardError(f"video backend cannot seek to frame {frame_id}: {path}")
        ok, image = capture.read()
        if not ok:
            raise FallGuardError(f"cannot read frame {frame_id}: {path}")
        return image
    finally:
        capture.release()
