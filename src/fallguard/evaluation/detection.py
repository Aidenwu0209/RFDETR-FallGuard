"""Detection evaluation delegates AP calculation to official RF-DETR/COCO code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fallguard.detection.rfdetr_adapter import RFDETRDetector


def evaluate_with_official_rfdetr(
    detector: RFDETRDetector,
    dataset_dir: str | Path,
    *,
    split: str = "test",
    **official_kwargs: Any,
) -> Any:
    """Return metrics produced by the pinned official package; no local AP clone exists."""
    return detector.evaluate(dataset_dir=dataset_dir, split=split, **official_kwargs)
