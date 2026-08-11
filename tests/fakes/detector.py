from __future__ import annotations

from typing import Any

import numpy as np


class RawDetections:
    def __init__(self) -> None:
        self.xyxy = np.asarray([[10, 10, 50, 90], [0, 0, 20, 20]], dtype=float)
        self.confidence = np.asarray([0.9, 0.8], dtype=float)
        self.class_id = np.asarray([0, 1], dtype=int)
        self.data = {"class_name": np.asarray(["person", "dog"])}


class FakeRFDETRModel:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.train_kwargs: dict[str, Any] | None = None

    def predict(self, images: Any, **kwargs: Any) -> Any:
        if isinstance(images, list):
            return [RawDetections() for _ in images]
        return RawDetections()

    def train(self, **kwargs: Any) -> None:
        self.train_kwargs = kwargs

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        return {"delegated": True, "kwargs": kwargs}
