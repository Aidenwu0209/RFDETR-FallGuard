from __future__ import annotations

import cv2
import numpy as np
import pytest

from fallguard.schemas import Detection
from fallguard.video import read_frame_at
from fallguard.visualization import draw_detections


def one_detection() -> Detection:
    return Detection(
        frame_id=0,
        timestamp_seconds=0,
        bbox_xyxy=(5, 5, 20, 25),
        frame_width=32,
        frame_height=32,
        class_id=1,
        class_name="falling",
        confidence=0.9,
        source_id="source",
        session_id="session",
    )


@pytest.mark.unit
def test_visualization_does_not_mutate_input() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    annotated = draw_detections(image, [one_detection()])
    assert not np.array_equal(annotated, image)
    assert np.count_nonzero(image) == 0


@pytest.mark.integration
def test_offline_video_random_access(tmp_path) -> None:
    video_path = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5,
        (16, 16),
    )
    assert writer.isOpened()
    for value in (0, 80, 160):
        writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()
    frame = read_frame_at(video_path, 1)
    assert frame.shape == (16, 16, 3)
    assert 60 < float(frame.mean()) < 100
