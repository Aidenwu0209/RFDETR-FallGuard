"""Deterministic provider for tests and demos; never scientific evidence."""

from __future__ import annotations

import time
from typing import Literal

from fallguard.schemas import ProviderCapabilities, SemanticAssessment, SemanticReviewRequest
from fallguard.semantic.base import SemanticProvider


class MockProvider(SemanticProvider):
    name = "mock"
    model = "deterministic-rule-v1"
    component_kind = "mock"
    capabilities = ProviderCapabilities(
        supports_images=True,
        supports_structured_output=True,
        max_images=16,
        input_mode="images_and_text",
    )

    def __init__(self, decision: Literal["fall", "not_fall", "uncertain"] = "fall") -> None:
        self.decision = decision

    def health_check(self, *, live: bool = False) -> dict[str, str | bool]:
        return {"available": True, "live_call": False, "detail": "deterministic mock"}

    def review(self, request: SemanticReviewRequest) -> SemanticAssessment:
        started = time.perf_counter()
        return SemanticAssessment(
            decision=self.decision,
            confidence=1.0 if self.decision != "uncertain" else None,
            reason="MOCK deterministic assessment; not model or benchmark evidence",
            attempt_to_stand=None,
            risk_level="high" if self.decision == "fall" else "unknown",
            provider=self.name,
            model=self.model,
            input_mode="images_and_text" if request.image_refs else "text",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            schema_valid=True,
            provider_success=True,
            model_recommends_alert=self.decision == "fall",
            ground_truth_verified=False,
        )
