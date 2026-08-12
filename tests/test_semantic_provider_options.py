from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from fallguard.schemas import FallEvent, SemanticReviewRequest
from fallguard.semantic.providers.deepseek_api import DeepSeekProvider
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
