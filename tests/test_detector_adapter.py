from __future__ import annotations

import numpy as np
import pytest

from fallguard.detection.rfdetr_adapter import RFDETRDetector
from fallguard.exceptions import ConfigurationError, UnsupportedConfigurationError
from fallguard.schemas import DetectionMode, FrameMetadata
from tests.fakes.detector import FakeRFDETRModel

pytestmark = pytest.mark.unit


class FakeCheckpointFactory:
    checkpoint_call: tuple[str, dict[str, object]] | None = None
    class_names = ("standing", "fallen", "sitting", "lying")

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs: object) -> FakeCheckpointFactory:
        cls.checkpoint_call = (path, kwargs)
        return cls()


class FakeMismatchedCheckpointFactory(FakeCheckpointFactory):
    class_names = ("fallen", "standing", "sitting", "lying")


def metadata(frame_id: int = 0) -> FrameMetadata:
    return FrameMetadata(
        frame_id=frame_id,
        timestamp_seconds=float(frame_id),
        frame_width=100,
        frame_height=100,
        source_id="source",
        session_id="session",
    )


def test_person_only_conversion_filters_non_person(development_config) -> None:
    config = development_config.detector.model_copy(update={"allow_weight_download": True})
    detector = RFDETRDetector(config, model_factory=FakeRFDETRModel)
    detector.load()
    detections = detector.predict_frame(np.zeros((100, 100, 3)), metadata())
    assert len(detections) == 1
    assert detections[0].class_name == "person"
    assert detections[0].bbox_xyxy == (10.0, 10.0, 50.0, 90.0)


def test_configured_class_alias_is_applied(development_config) -> None:
    config = development_config.detector.model_copy(
        update={
            "allow_weight_download": True,
            "class_aliases": {"person": "human"},
            "person_class_names": ["human"],
        }
    )
    detector = RFDETRDetector(config, model_factory=FakeRFDETRModel)
    detector.load()
    detections = detector.predict_frame(np.zeros((100, 100, 3)), metadata())
    assert detections[0].class_name == "human"


def test_batch_preserves_frame_metadata(development_config) -> None:
    config = development_config.detector.model_copy(update={"allow_weight_download": True})
    detector = RFDETRDetector(config, model_factory=FakeRFDETRModel)
    detector.load()
    batch = detector.predict_batch(
        [np.zeros((100, 100, 3)), np.zeros((100, 100, 3))],
        [metadata(0), metadata(1)],
    )
    assert [items[0].frame_id for items in batch] == [0, 1]


def test_train_aliases_are_explicit_and_gradient_checkpointing_rejected() -> None:
    resolved = RFDETRDetector.resolve_train_configuration(
        {"learning_rate": 0.001, "gradient_accumulation_steps": 2, "gpu_count": 1}
    )
    assert resolved == {"lr": 0.001, "grad_accum_steps": 2, "devices": 1}
    with pytest.raises(UnsupportedConfigurationError, match="no gradient_checkpointing"):
        RFDETRDetector.resolve_train_configuration({"gradient_checkpointing": True})


def test_detection_evaluation_is_delegated_to_official_model(development_config, tmp_path) -> None:
    model = FakeRFDETRModel()
    config = development_config.detector.model_copy(update={"allow_weight_download": True})
    detector = RFDETRDetector(config, model_factory=lambda **kwargs: model)
    detector.load()
    result = detector.evaluate(dataset_dir=tmp_path, split="test")
    assert result == {
        "delegated": True,
        "kwargs": {"dataset_dir": str(tmp_path), "split": "test"},
    }


def test_posture_checkpoint_uses_official_checkpoint_loader(
    development_config, tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "checkpoint_best_total.pth"
    checkpoint.touch()
    config = development_config.detector.model_copy(
        update={
            "mode": DetectionMode.POSTURE_MULTICLASS,
            "weights_path": checkpoint,
            "class_names": {0: "standing", 1: "fallen", 2: "sitting", 3: "lying"},
        }
    )
    detector = RFDETRDetector(config, device="cuda:0")
    monkeypatch.setattr(detector, "_official_factory", lambda: FakeCheckpointFactory)

    detector.load()

    assert isinstance(detector._model, FakeCheckpointFactory)
    assert FakeCheckpointFactory.checkpoint_call == (
        str(checkpoint),
        {"num_classes": 4, "device": "cuda:0"},
    )


def test_posture_checkpoint_rejects_non_contiguous_class_ids(
    development_config, tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "checkpoint_best_total.pth"
    checkpoint.touch()
    config = development_config.detector.model_copy(
        update={
            "mode": DetectionMode.POSTURE_MULTICLASS,
            "weights_path": checkpoint,
            "class_names": {0: "standing", 2: "fallen"},
        }
    )
    detector = RFDETRDetector(config)
    monkeypatch.setattr(detector, "_official_factory", lambda: FakeCheckpointFactory)

    with pytest.raises(ConfigurationError, match="contiguous"):
        detector.load()


def test_posture_checkpoint_rejects_class_order_mismatch(
    development_config, tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "checkpoint_best_total.pth"
    checkpoint.touch()
    config = development_config.detector.model_copy(
        update={
            "mode": DetectionMode.POSTURE_MULTICLASS,
            "weights_path": checkpoint,
            "class_names": {0: "standing", 1: "fallen", 2: "sitting", 3: "lying"},
        }
    )
    detector = RFDETRDetector(config)
    monkeypatch.setattr(detector, "_official_factory", lambda: FakeMismatchedCheckpointFactory)

    with pytest.raises(ConfigurationError, match="class_names differ"):
        detector.load()
