"""Simple deterministic annotation for image/video engineering review."""

from __future__ import annotations

import cv2
import numpy as np

from fallguard.schemas import Detection


def draw_detections(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection.bbox_xyxy)
        color = _class_color(detection.class_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def _class_color(class_id: int) -> tuple[int, int, int]:
    return (
        64 + (class_id * 53) % 192,
        64 + (class_id * 97) % 192,
        64 + (class_id * 149) % 192,
    )
