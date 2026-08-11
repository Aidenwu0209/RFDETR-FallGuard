"""Single business-facing semantic route with privacy and fallback audit gates."""

from __future__ import annotations

from fallguard.config import SemanticConfig
from fallguard.exceptions import (
    FallGuardError,
    PrivacyConsentRequiredError,
    ProviderUnavailableError,
)
from fallguard.schemas import SemanticAssessment, SemanticReviewRequest
from fallguard.semantic.base import SemanticProvider


class SemanticReviewRouter:
    def __init__(self, config: SemanticConfig, providers: dict[str, SemanticProvider]) -> None:
        self.config = config
        self.providers = providers

    def review(self, request: SemanticReviewRequest) -> SemanticAssessment:
        if self.config.provider == "none":
            raise ProviderUnavailableError("semantic provider is disabled")
        names = [self.config.provider]
        if self.config.allow_fallback:
            names.extend(name for name in self.config.fallback_providers if name not in names)
        first_failure: str | None = None
        for name in names:
            provider = self.providers.get(name)
            if provider is None:
                first_failure = first_failure or f"provider not registered: {name}"
                continue
            if provider.component_kind == "mock" and not self.config.allow_mock:
                first_failure = first_failure or "mock provider is disabled"
                continue
            prepared = self._prepare_request(request, provider)
            try:
                assessment = provider.review(prepared)
            except FallGuardError as exc:
                first_failure = first_failure or f"{name}: {type(exc).__name__}: {exc}"
                if not self.config.allow_fallback:
                    raise
                continue
            if first_failure:
                assessment = assessment.model_copy(update={"fallback_reason": first_failure})
            return assessment
        raise ProviderUnavailableError(first_failure or "no semantic provider is available")

    def health(self) -> dict[str, dict[str, str | bool]]:
        return {
            name: provider.health_check(live=False) for name, provider in self.providers.items()
        }

    def _prepare_request(
        self,
        request: SemanticReviewRequest,
        provider: SemanticProvider,
    ) -> SemanticReviewRequest:
        if (
            provider.is_cloud
            and request.image_refs
            and (not self.config.allow_cloud_images or not request.cloud_image_consent)
        ):
            raise PrivacyConsentRequiredError(
                f"cloud image review by {provider.name} requires both configured "
                "and per-request consent"
            )
        if not provider.capabilities.supports_images:
            return request.model_copy(update={"image_refs": []})
        if len(request.image_refs) > min(provider.capabilities.max_images, self.config.max_images):
            return request.model_copy(
                update={
                    "image_refs": request.image_refs[
                        : min(provider.capabilities.max_images, self.config.max_images)
                    ]
                }
            )
        return request
