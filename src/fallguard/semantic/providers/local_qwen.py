"""Local Qwen-compatible vision-language provider with diagnostic failure states."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from fallguard.exceptions import (
    DependencyUnavailableError,
    ModelUnavailableError,
    ProviderUnavailableError,
)
from fallguard.schemas import ProviderCapabilities, SemanticAssessment, SemanticReviewRequest
from fallguard.semantic.base import ProviderPayload, SemanticProvider


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
            self._processor = AutoProcessor.from_pretrained(
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
        images: list[Image.Image] = []
        schema_instruction = (
            "Return only JSON with decision (fall/not_fall/uncertain), confidence, reason, "
            "attempt_to_stand, risk_level (low/medium/high/unknown), model_recommends_alert."
        )
        started = time.perf_counter()
        try:
            for item in request.image_refs:
                with Image.open(item.path) as raw_image:
                    images.append(raw_image.convert("RGB"))
            content: list[dict[str, str]] = [{"type": "image"} for _ in images]
            content.append(
                {"type": "text", "text": schema_instruction + "\n" + request.text_context}
            )
            messages: list[dict[str, object]] = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(text=[text], images=images, return_tensors="pt")
            if hasattr(model, "device"):
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
            generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            prompt_length = inputs["input_ids"].shape[1]
            decoded = processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
            )[0]
            payload = ProviderPayload.model_validate(json.loads(decoded))
        except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ProviderUnavailableError(
                f"Local Qwen review failed or returned invalid JSON: {exc}"
            ) from exc
        finally:
            for image in images:
                image.close()
        return SemanticAssessment(
            **payload.model_dump(),
            provider=self.name,
            model=self.model,
            input_mode="images_and_text" if request.image_refs else "text",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            schema_valid=True,
            provider_success=True,
        )
