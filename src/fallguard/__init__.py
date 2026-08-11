"""RFDETR-FallGuard public package."""

from fallguard.schemas import (
    AlertDecision,
    Detection,
    FallEvent,
    SemanticAssessment,
    Track,
)

__all__ = ["AlertDecision", "Detection", "FallEvent", "SemanticAssessment", "Track"]
__version__ = "0.1.0"
