"""Strict adapter around the official RF-DETR Python implementation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from fallguard.config import DetectorConfig
from fallguard.exceptions import (
    ConfigurationError,
    DependencyUnavailableError,
    ModelUnavailableError,
    UnsupportedConfigurationError,
)
from fallguard.schemas import Detection, DetectionMode, FrameMetadata

PINNED_RFDETR_VERSION = "1.9.1"
TRAIN_ALIASES = {
    "learning_rate": "lr",
    "gradient_accumulation_steps": "grad_accum_steps",
    "gpu_count": "devices",
}


class RFDETRDetector:
    component_kind = "real"

    def __init__(
        self,
        config: DetectorConfig,
        *,
        device: str = "auto",
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self._model_factory = model_factory
        self._model: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.loaded:
            return
        if self.config.mode == DetectionMode.POSTURE_MULTICLASS:
            if self.config.weights_path is None:
                raise ModelUnavailableError("posture_multiclass requires a fine-tuned weights_path")
            if not self.config.class_names:
                raise ConfigurationError(
                    "posture_multiclass requires class_names from dataset/checkpoint metadata"
                )
        if self.config.weights_path is not None and not self.config.weights_path.is_file():
            raise ModelUnavailableError(f"RF-DETR weights do not exist: {self.config.weights_path}")
        if self.config.weights_path is None and not self.config.allow_weight_download:
            raise ModelUnavailableError(
                "no weights_path was provided and implicit official-weight download is disabled"
            )

        official_factory = self._official_factory() if self._model_factory is None else None
        factory = self._model_factory or official_factory
        assert factory is not None
        kwargs: dict[str, Any] = {}
        if self.device != "auto":
            kwargs["device"] = self.device
        if (
            self._model_factory is None
            and self.config.mode == DetectionMode.POSTURE_MULTICLASS
            and self.config.weights_path is not None
        ):
            class_ids = sorted(self.config.class_names)
            if class_ids != list(range(len(class_ids))):
                raise ConfigurationError(
                    "posture_multiclass class_names IDs must be contiguous and start at zero"
                )
            assert official_factory is not None
            loaded = official_factory.from_checkpoint(
                str(self.config.weights_path),
                num_classes=len(class_ids),
                **kwargs,
            )
            if not isinstance(loaded, official_factory):
                raise ConfigurationError(
                    "checkpoint architecture does not match selected "
                    f"{self.config.model_variant} variant"
                )
            checkpoint_names = getattr(loaded, "class_names", None)
            expected_names = [self.config.class_names[index] for index in class_ids]
            if checkpoint_names is not None and list(checkpoint_names) != expected_names:
                raise ConfigurationError(
                    f"checkpoint class_names differ from config: {checkpoint_names} != "
                    f"{expected_names}"
                )
            self._model = loaded
            return

        if self.config.weights_path is not None:
            kwargs["pretrain_weights"] = str(self.config.weights_path)
        self._model = factory(**kwargs)

    def _official_factory(self) -> type[Any]:
        try:
            installed = version("rfdetr")
        except PackageNotFoundError as exc:
            raise DependencyUnavailableError(
                "RF-DETR is optional; install the pinned integration with .[rfdetr]"
            ) from exc
        if installed != PINNED_RFDETR_VERSION:
            raise DependencyUnavailableError(
                f"unsupported rfdetr version {installed}; expected {PINNED_RFDETR_VERSION}"
            )
        try:
            from rfdetr import RFDETRNano, RFDETRSmall
        except ImportError as exc:
            raise DependencyUnavailableError(f"failed to import rfdetr: {exc}") from exc
        factories: dict[str, type[Any]] = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
        }
        return factories[self.config.model_variant]

    def predict_image(
        self,
        image: str | Path | np.ndarray[Any, Any],
        metadata: FrameMetadata,
    ) -> list[Detection]:
        model = self._model_or_raise()
        raw = model.predict(
            str(image) if isinstance(image, Path) else image,
            threshold=self.config.confidence_threshold,
            include_source_image=False,
        )
        return self._convert(raw, metadata)

    def predict_frame(
        self,
        image: np.ndarray[Any, Any],
        metadata: FrameMetadata,
    ) -> list[Detection]:
        return self.predict_image(image, metadata)

    def predict_batch(
        self,
        images: Sequence[str | Path | np.ndarray[Any, Any]],
        metadata: Sequence[FrameMetadata],
    ) -> list[list[Detection]]:
        if len(images) != len(metadata):
            raise ValueError("images and metadata must have equal length")
        model = self._model_or_raise()
        normalized = [str(image) if isinstance(image, Path) else image for image in images]
        raw_batch = model.predict(
            normalized,
            threshold=self.config.confidence_threshold,
            include_source_image=False,
        )
        if len(images) == 1 and not isinstance(raw_batch, list):
            raw_batch = [raw_batch]
        return [self._convert(raw, meta) for raw, meta in zip(raw_batch, metadata, strict=True)]

    def train(self, **configuration: Any) -> dict[str, Any]:
        model = self._model_or_raise()
        resolved = self.resolve_train_configuration(configuration)
        model.train(**resolved)
        return resolved

    @staticmethod
    def resolve_train_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
        if "gradient_checkpointing" in configuration:
            raise UnsupportedConfigurationError(
                "rfdetr==1.9.1 TrainConfig has no gradient_checkpointing field"
            )
        resolved: dict[str, Any] = {}
        for key, value in configuration.items():
            official = TRAIN_ALIASES.get(key, key)
            if official in resolved:
                raise ConfigurationError(f"duplicate RF-DETR parameter after aliasing: {official}")
            resolved[official] = value
        return resolved

    def evaluate(self, *, dataset_dir: str | Path, split: str = "test", **kwargs: Any) -> Any:
        model = self._model_or_raise()
        if split not in {"test", "val"}:
            raise ConfigurationError("RF-DETR official evaluation split must be 'test' or 'val'")
        return model.evaluate(dataset_dir=str(dataset_dir), split=split, **kwargs)

    def close(self) -> None:
        self._model = None

    def _model_or_raise(self) -> Any:
        if self._model is None:
            raise ModelUnavailableError("RFDETRDetector.load() must succeed before use")
        return self._model

    def _convert(self, raw: Any, metadata: FrameMetadata) -> list[Detection]:
        xyxy = np.asarray(raw.xyxy)
        confidences = np.asarray(raw.confidence)
        class_ids = np.asarray(raw.class_id)
        data = getattr(raw, "data", {}) or {}
        raw_names = data.get("class_name")
        converted: list[Detection] = []
        for index, (box, confidence, class_id) in enumerate(
            zip(xyxy, confidences, class_ids, strict=True)
        ):
            numeric_id = int(class_id)
            class_name = (
                str(raw_names[index])
                if raw_names is not None
                else self.config.class_names.get(numeric_id, f"class_{numeric_id}")
            )
            class_name = self.config.class_aliases.get(class_name, class_name)
            if (
                self.config.mode == DetectionMode.PERSON_ONLY
                and class_name not in self.config.person_class_names
            ):
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            converted.append(
                Detection(
                    frame_id=metadata.frame_id,
                    timestamp_seconds=metadata.timestamp_seconds,
                    bbox_xyxy=(x1, y1, x2, y2),
                    frame_width=metadata.frame_width,
                    frame_height=metadata.frame_height,
                    class_id=numeric_id,
                    class_name=class_name,
                    confidence=float(confidence),
                    source_id=metadata.source_id,
                    session_id=metadata.session_id,
                )
            )
        return converted
