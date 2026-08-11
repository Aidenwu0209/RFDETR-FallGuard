from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fallguard.config import load_config
from fallguard.exceptions import ConfigurationError, FormalBenchmarkRejectedError
from fallguard.schemas import Detection
from fallguard.session import make_session_id

pytestmark = pytest.mark.unit


def test_generated_session_ids_are_human_readable_and_unique() -> None:
    first = make_session_id("input clip")
    second = make_session_id("input clip")
    assert first.startswith("input-clip-")
    assert first != second


def valid_detection() -> dict[str, object]:
    return {
        "frame_id": 0,
        "timestamp_seconds": 0.0,
        "bbox_xyxy": (1.0, 2.0, 10.0, 20.0),
        "frame_width": 100,
        "frame_height": 100,
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.9,
        "source_id": "source",
        "session_id": "session",
    }


def test_detection_rejects_invalid_box_and_frame_bounds() -> None:
    values = valid_detection()
    values["bbox_xyxy"] = (10.0, 2.0, 1.0, 20.0)
    with pytest.raises(ValidationError, match="x2 > x1"):
        Detection(**values)
    values = valid_detection()
    values["bbox_xyxy"] = (1.0, 2.0, 101.0, 20.0)
    with pytest.raises(ValidationError, match="frame dimensions"):
        Detection(**values)


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    overlay = tmp_path / "bad.yaml"
    overlay.write_text("runtime:\n  unknown_option: true\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown_option"):
        load_config(overlay)


def test_experiment_profile_fails_fast_without_validated_thresholds() -> None:
    config = load_config("configs/profiles/experiment.yaml")
    with pytest.raises(FormalBenchmarkRejectedError, match="unvalidated temporal thresholds"):
        config.assert_formal_ready()
