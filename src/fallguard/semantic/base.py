"""Semantic provider contract and shared strict payload."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fallguard.schemas import ProviderCapabilities, SemanticAssessment, SemanticReviewRequest


class ProviderPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["fall", "not_fall", "uncertain"]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    attempt_to_stand: bool | None = None
    risk_level: Literal["low", "medium", "high", "unknown"] = "unknown"
    model_recommends_alert: bool | None = None


class SemanticProvider(ABC):
    name: str
    model: str
    capabilities: ProviderCapabilities
    is_cloud: bool = False
    component_kind: str = "real"

    @abstractmethod
    def health_check(self, *, live: bool = False) -> dict[str, str | bool]:
        """Return diagnostics. The default live=False path must not contact a service."""

    @abstractmethod
    def review(self, request: SemanticReviewRequest) -> SemanticAssessment:
        """Review an event and return a schema-valid assessment."""
