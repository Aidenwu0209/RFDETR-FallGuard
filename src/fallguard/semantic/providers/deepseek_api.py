"""DeepSeek text-context provider; OpenAI compatibility does not imply vision support."""

from __future__ import annotations

import json
import os
import time

import httpx

from fallguard.exceptions import ProviderUnavailableError
from fallguard.schemas import ProviderCapabilities, SemanticAssessment, SemanticReviewRequest
from fallguard.semantic.base import ProviderPayload, SemanticProvider


class DeepSeekProvider(SemanticProvider):
    name = "deepseek"
    is_cloud = True
    capabilities = ProviderCapabilities(
        supports_images=False,
        supports_structured_output=True,
        max_images=0,
        input_mode="text",
    )

    def __init__(self, model: str, *, base_url: str, timeout_seconds: float = 30) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def health_check(self, *, live: bool = False) -> dict[str, str | bool]:
        return {
            "available": bool(os.getenv("DEEPSEEK_API_KEY")),
            "live_call": False,
            "detail": "local key/configuration check only; provider is text-only in this project",
        }

    def review(self, request: SemanticReviewRequest) -> SemanticAssessment:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ProviderUnavailableError("DEEPSEEK_API_KEY is not configured")
        prompt = (
            "Return JSON matching these keys: decision, confidence, reason, attempt_to_stand, "
            "risk_level, model_recommends_alert. decision must be fall, not_fall, or uncertain.\n"
            + request.text_context
        )
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"},
                        "stream": False,
                    },
                )
                response.raise_for_status()
            body = response.json()
            payload = ProviderPayload.model_validate(
                json.loads(body["choices"][0]["message"]["content"])
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise ProviderUnavailableError(
                f"DeepSeek request failed or returned invalid structured output: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = body.get("usage", {})
        return SemanticAssessment(
            **payload.model_dump(),
            provider=self.name,
            model=self.model,
            input_mode="text",
            latency_ms=latency_ms,
            schema_valid=True,
            provider_success=True,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )
