"""Detector interface used by the pipeline."""

from __future__ import annotations

from typing import Any, Protocol

from fallguard.schemas import Detection, FrameMetadata


class Detector(Protocol):
    component_kind: str

    def load(self) -> None: ...

    def predict_frame(self, image: Any, metadata: FrameMetadata) -> list[Detection]: ...

    def close(self) -> None: ...
