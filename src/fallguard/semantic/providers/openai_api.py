"""OpenAI image/text provider using official Responses structured parsing."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Literal, cast

from fallguard.exceptions import DependencyUnavailableError, ProviderUnavailableError
from fallguard.schemas import ProviderCapabilities, SemanticAssessment, SemanticReviewRequest
from fallguard.semantic.base import ProviderPayload, SemanticProvider


class OpenAIProvider(SemanticProvider):
    name = "openai"
    is_cloud = True
    capabilities = ProviderCapabilities(
        supports_images=True,
        supports_structured_output=True,
        max_images=8,
        input_mode="images_and_text",
    )

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 30,
        reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort

    def health_check(self, *, live: bool = False) -> dict[str, str | bool]:
        if live:
            return {
                "available": False,
                "live_call": False,
                "detail": (
                    "live connectivity is only performed by an explicit paid integration test"
                ),
            }
        try:
            import openai  # noqa: F401
        except ImportError:
            return {"available": False, "live_call": False, "detail": "openai package missing"}
        return {
            "available": bool(os.getenv("OPENAI_API_KEY")),
            "live_call": False,
            "detail": "local configuration only",
        }

    def review(self, request: SemanticReviewRequest) -> SemanticAssessment:
        if not os.getenv("OPENAI_API_KEY"):
            raise ProviderUnavailableError("OPENAI_API_KEY is not configured")
        try:
            from openai import OpenAI, OpenAIError
        except ImportError as exc:
            raise DependencyUnavailableError("install .[cloud] for OpenAI integration") from exc
        try:
            client = OpenAI(base_url=self.base_url, timeout=self.timeout_seconds)
            content: list[dict[str, object]] = [
                {"type": "input_text", "text": request.text_context}
            ]
            for image in request.image_refs[: self.capabilities.max_images]:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": self._data_url(Path(image.path)),
                        "detail": "low",
                    }
                )
            started = time.perf_counter()
            request_options: dict[str, Any] = {
                "model": self.model,
                "input": cast(Any, [{"role": "user", "content": content}]),
                "text_format": ProviderPayload,
                "store": False,
            }
            if self.reasoning_effort is not None:
                request_options["reasoning"] = {"effort": self.reasoning_effort}
            response = client.responses.parse(**request_options)
        except (OpenAIError, OSError, ValueError) as exc:
            raise ProviderUnavailableError(f"OpenAI review failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        payload = response.output_parsed
        if payload is None:
            raise ProviderUnavailableError("OpenAI returned no parsed structured assessment")
        usage = getattr(response, "usage", None)
        output_details = getattr(usage, "output_tokens_details", None)
        return SemanticAssessment(
            **payload.model_dump(),
            provider=self.name,
            model=self.model,
            input_mode="images_and_text" if request.image_refs else "text",
            latency_ms=latency_ms,
            schema_valid=True,
            provider_success=True,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            reasoning_tokens=getattr(output_details, "reasoning_tokens", None),
        )

    @staticmethod
    def _data_url(path: Path) -> str:
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"
