from __future__ import annotations

import logging

import pytest

from fallguard.alert import AlertManager
from fallguard.exceptions import PrivacyConsentRequiredError, ProviderUnavailableError
from fallguard.logging import RedactingFilter, redact_text
from fallguard.schemas import (
    FallEvent,
    MotionState,
    ProviderCapabilities,
    SemanticAssessment,
    SemanticReviewRequest,
)
from fallguard.semantic.base import SemanticProvider
from fallguard.semantic.providers.mock import MockProvider
from fallguard.semantic.router import SemanticReviewRouter

pytestmark = pytest.mark.unit


def event() -> FallEvent:
    return FallEvent(
        track_id=1,
        source_id="source",
        session_id="session",
        start_frame=0,
        start_time=0,
    )


class FailingProvider(SemanticProvider):
    name = "openai"
    model = "failure"
    is_cloud = True
    capabilities = ProviderCapabilities(
        supports_images=True,
        supports_structured_output=True,
        max_images=3,
        input_mode="images_and_text",
    )

    def health_check(self, *, live: bool = False):
        return {"available": False, "live_call": False}

    def review(self, request):
        raise ProviderUnavailableError("intentional test failure")


class BuggyProvider(FailingProvider):
    def review(self, request):
        raise RuntimeError("programming defect must not be treated as provider downtime")


def assessment(decision: str, recommends: bool) -> SemanticAssessment:
    return SemanticAssessment(
        decision=decision,
        confidence=0.9,
        reason="test evidence",
        risk_level="high",
        provider="provider",
        model="model",
        input_mode="text",
        latency_ms=1,
        schema_valid=True,
        provider_success=True,
        model_recommends_alert=recommends,
    )


def test_router_records_fallback_without_claiming_primary_success(development_config) -> None:
    config = development_config.semantic.model_copy(
        update={
            "provider": "openai",
            "fallback_providers": ["mock"],
            "allow_fallback": True,
            "allow_mock": True,
        }
    )
    router = SemanticReviewRouter(
        config,
        {"openai": FailingProvider(), "mock": MockProvider()},
    )
    result = router.review(
        SemanticReviewRequest(event=event(), text_context="test", cloud_image_consent=False)
    )
    assert result.provider == "mock"
    assert "intentional test failure" in result.fallback_reason


def test_router_does_not_mask_unexpected_programming_errors(development_config) -> None:
    config = development_config.semantic.model_copy(
        update={
            "provider": "openai",
            "fallback_providers": ["mock"],
            "allow_fallback": True,
            "allow_mock": True,
        }
    )
    router = SemanticReviewRouter(
        config,
        {"openai": BuggyProvider(), "mock": MockProvider()},
    )
    with pytest.raises(RuntimeError, match="programming defect"):
        router.review(
            SemanticReviewRequest(event=event(), text_context="test", cloud_image_consent=False)
        )


def test_cloud_images_require_two_consent_gates(development_config, tmp_path) -> None:
    from fallguard.schemas import ImageRef

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"not-read-before-gate")
    request = SemanticReviewRequest(
        event=event(),
        text_context="test",
        image_refs=[
            ImageRef(
                path=image,
                sha256="a" * 64,
                width=1,
                height=1,
                kind="person_crop",
            )
        ],
        cloud_image_consent=False,
    )
    config = development_config.semantic.model_copy(
        update={"provider": "openai", "allow_cloud_images": True, "allow_fallback": False}
    )
    with pytest.raises(PrivacyConsentRequiredError):
        SemanticReviewRouter(config, {"openai": FailingProvider()}).review(request)


def test_alert_manager_owns_final_decision(development_config) -> None:
    manager = AlertManager(development_config.alert)
    fall = manager.decide(event(), MotionState.LYING, assessment("fall", False))
    not_fall = manager.decide(event(), MotionState.LYING, assessment("not_fall", True))
    assert fall.should_alert is True
    assert not_fall.should_alert is False


def test_log_redaction_removes_secrets_authorization_and_base64() -> None:
    text = redact_text(
        "Authorization: Bearer abc123 api_key=secret data:image/jpeg;base64,QUJDREVGRw=="
    )
    assert "abc123" not in text
    assert "secret" not in text
    assert "QUJD" not in text
    record = logging.LogRecord("test", logging.INFO, __file__, 1, text, (), None)
    assert RedactingFilter().filter(record) is True
