"""Local Qwen-compatible vision-language provider with diagnostic failure states."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fallguard.exceptions import (
    DependencyUnavailableError,
    ModelUnavailableError,
    ProviderUnavailableError,
)
from fallguard.schemas import ProviderCapabilities, SemanticAssessment, SemanticReviewRequest
from fallguard.semantic.base import ProviderPayload, SemanticProvider


def parse_provider_payload(decoded: str) -> ProviderPayload:
    """Accept raw JSON or one conventional JSON code fence, then validate strictly."""

    stripped = decoded.strip()
    if "```" in stripped and not stripped.startswith("```"):
        raise ValueError("extra text appears outside Markdown fences")
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) < 3 or lines[0].lower() not in {"```", "```json"} or lines[-1] != "```":
            raise ValueError("invalid or multiple Markdown fences around JSON payload")
        stripped = "\n".join(lines[1:-1]).strip()
    return ProviderPayload.model_validate_json(stripped)


class LocalQwenProvider(SemanticProvider):
    name = "local_qwen"
    is_cloud = False
    capabilities = ProviderCapabilities(
        supports_images=True,
        supports_structured_output=False,
        max_images=8,
        input_mode="images_and_text",
    )

    def __init__(self, model_path: str | Path, *, model_name: str | None = None) -> None:
        self.model_path = Path(model_path)
        self.model = model_name or self.model_path.name
        self._processor: Any | None = None
        self._model: Any | None = None

    def health_check(self, *, live: bool = False) -> dict[str, str | bool]:
        try:
            import transformers  # noqa: F401
        except ImportError:
            return {"available": False, "live_call": False, "detail": "transformers missing"}
        return {
            "available": self.model_path.exists(),
            "live_call": False,
            "detail": "local path/dependency check only; model is not loaded",
        }

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise ModelUnavailableError(f"local Qwen model path does not exist: {self.model_path}")
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise DependencyUnavailableError("install .[local-vlm] for Local Qwen") from exc
        try:
            self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                self.model_path,
                local_files_only=True,
                trust_remote_code=False,
            )
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=False,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelUnavailableError(f"failed to load Local Qwen: {exc}") from exc

    def review(self, request: SemanticReviewRequest) -> SemanticAssessment:
        self.load()
        if self._processor is None or self._model is None:
            raise ModelUnavailableError("Local Qwen load returned no processor or model")
        processor: Any = self._processor
        model: Any = self._model
        schema_instruction = (
            "Return exactly one JSON object and no markdown or extra text. Use these exact "
            "fields and JSON types: decision is one of fall, not_fall, uncertain; confidence "
            "is a number from 0 to 1 or null; reason is a non-empty string; "
            "attempt_to_stand is true, false, or null; risk_level is one of low, medium, "
            "high, unknown; model_recommends_alert is true, false, or null."
        )
        started = time.perf_counter()
        try:
            content: list[dict[str, object]] = [
                {"type": "image", "path": str(item.path.resolve())} for item in request.image_refs
            ]
            content.append(
                {"type": "text", "text": schema_instruction + "\n" + request.text_context}
            )
            messages: list[dict[str, object]] = [{"role": "user", "content": content}]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            if hasattr(model, "device") and hasattr(inputs, "to"):
                inputs = inputs.to(model.device)
            elif hasattr(model, "device"):
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
            generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            prompt_length = inputs["input_ids"].shape[1]
            decoded = processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
            )[0]
            payload = parse_provider_payload(decoded)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            raise ProviderUnavailableError(
                "Local Qwen review failed or returned invalid JSON: "
                f"{exc}; decoded_prefix={locals().get('decoded', '')[:1000]!r}"
            ) from exc
        return SemanticAssessment(
            **payload.model_dump(),
            provider=self.name,
            model=self.model,
            input_mode="images_and_text" if request.image_refs else "text",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            schema_valid=True,
            provider_success=True,
        )
