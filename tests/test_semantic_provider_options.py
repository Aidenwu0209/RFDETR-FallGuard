from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fallguard.schemas import FallEvent, ImageRef, SemanticReviewRequest
from fallguard.semantic.providers.deepseek_api import DeepSeekProvider
from fallguard.semantic.providers.local_qwen import LocalQwenProvider, parse_provider_payload
from fallguard.semantic.providers.openai_api import OpenAIProvider

pytestmark = pytest.mark.unit


def request() -> SemanticReviewRequest:
    return SemanticReviewRequest(
        event=FallEvent(
            track_id=1,
            source_id="unit-test",
            session_id="unit-test",
            start_frame=0,
            start_time=0,
        ),
        text_context="Synthetic event evidence.",
    )


def test_openai_disables_storage_and_passes_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    payload = SimpleNamespace(
        model_dump=lambda: {
            "decision": "uncertain",
            "confidence": 0.4,
            "reason": "insufficient evidence",
            "attempt_to_stand": None,
            "risk_level": "unknown",
            "model_recommends_alert": False,
        }
    )

    class FakeResponses:
        def parse(self, **kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                output_parsed=payload,
                usage=SimpleNamespace(
                    input_tokens=12,
                    output_tokens=7,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=3),
                ),
            )

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-secret")
    monkeypatch.setattr("openai.OpenAI", FakeClient)
    result = OpenAIProvider("gateway-model", reasoning_effort="xhigh").review(request())
    assert captured["store"] is False
    assert captured["reasoning"] == {"effort": "xhigh"}
    assert result.reasoning_tokens == 3


def test_deepseek_v4_uses_typed_contract_and_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    response_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "decision": "uncertain",
                            "confidence": 0.4,
                            "reason": "insufficient evidence",
                            "attempt_to_stand": None,
                            "risk_level": "unknown",
                            "model_recommends_alert": False,
                        }
                    )
                }
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "reasoning_tokens": 4,
        },
    }

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return response_payload

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            return None

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, url: str, **kwargs: Any) -> FakeResponse:
            captured["url"] = url
            captured.update(kwargs)
            return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-secret")
    monkeypatch.setattr("httpx.Client", FakeClient)
    result = DeepSeekProvider(
        "deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        reasoning_effort="high",
    ).review(request())
    body = captured["json"]
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 4096
    assert "confidence must be a JSON number" in body["messages"][0]["content"]
    assert result.reasoning_tokens == 4


def test_local_qwen_uses_official_multimodal_chat_template(tmp_path: Path) -> None:
    image_path = tmp_path / "during.jpg"
    image_path.write_bytes(b"unit-test-placeholder")
    captured: dict[str, Any] = {}

    class FakeInputs(dict[str, torch.Tensor]):
        def to(self, device: object) -> FakeInputs:
            captured["device"] = device
            return self

    class FakeProcessor:
        def apply_chat_template(self, messages: object, **kwargs: Any) -> FakeInputs:
            captured["messages"] = messages
            captured.update(kwargs)
            return FakeInputs(input_ids=torch.tensor([[1, 2, 3]]))

        def batch_decode(self, *_args: object, **_kwargs: object) -> list[str]:
            return [
                json.dumps(
                    {
                        "decision": "fall",
                        "confidence": 0.9,
                        "reason": "person transitions to the floor",
                        "attempt_to_stand": False,
                        "risk_level": "high",
                        "model_recommends_alert": True,
                    }
                )
            ]

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, **kwargs: Any) -> torch.Tensor:
            assert kwargs["max_new_tokens"] == 512
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 2, 3, 4]])

    provider = LocalQwenProvider(tmp_path, model_name="Qwen3.5-4B")
    provider._processor = FakeProcessor()
    provider._model = FakeModel()
    semantic_request = request().model_copy(
        update={
            "image_refs": [
                ImageRef(
                    path=image_path,
                    sha256="0" * 64,
                    width=1,
                    height=1,
                    kind="person_crop",
                )
            ]
        }
    )

    result = provider.review(semantic_request)

    assert captured["tokenize"] is True
    assert captured["return_dict"] is True
    assert captured["return_tensors"] == "pt"
    assert captured["enable_thinking"] is False
    messages = captured["messages"]
    assert messages[0]["content"][0] == {
        "type": "image",
        "path": str(image_path.resolve()),
    }
    assert result.decision == "fall"
    assert result.provider_success is True


def test_local_qwen_accepts_one_json_code_fence() -> None:
    payload = parse_provider_payload(
        """```json
{"decision":"not_fall","confidence":0.95,"reason":"controlled movement",\
"attempt_to_stand":false,"risk_level":"low","model_recommends_alert":false}
```"""
    )
    assert payload.decision == "not_fall"
    assert payload.confidence == 0.95


def test_local_qwen_rejects_extra_text_around_json_fence() -> None:
    with pytest.raises(ValueError, match="fences"):
        parse_provider_payload("prefix\n```json\n{}\n```")
