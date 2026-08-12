"""DeepSeek text-context provider; OpenAI compatibility does not imply vision support."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Literal

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

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        timeout_seconds: float = 30,
        reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
        thinking_enabled: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort
        self.thinking_enabled = thinking_enabled

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
            "Return exactly one JSON object with only these keys and types: "
            '{"decision":"fall|not_fall|uncertain","confidence":0.0,'
            '"reason":"string","attempt_to_stand":false,'
            '"risk_level":"low|medium|high|unknown","model_recommends_alert":false}. '
            "confidence must be a JSON number from 0 through 1, or null; never use low, medium, "
            "or high. attempt_to_stand must be true, false, or null; use null when unavailable, "
            "and never use the string unknown.\n" + request.text_context
        )
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "max_tokens": 4096,
            "stream": False,
        }
        if self.thinking_enabled:
            request_body["thinking"] = {"type": "enabled"}
        if self.reasoning_effort is not None:
            request_body["reasoning_effort"] = self.reasoning_effort
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=request_body,
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
        reasoning_tokens = usage.get("reasoning_tokens")
        if reasoning_tokens is None:
            reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
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
            reasoning_tokens=reasoning_tokens,
        )
