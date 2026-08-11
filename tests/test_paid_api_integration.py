from __future__ import annotations

import os

import pytest

from fallguard.schemas import FallEvent, SemanticReviewRequest
from fallguard.semantic.providers.deepseek_api import DeepSeekProvider
from fallguard.semantic.providers.openai_api import OpenAIProvider

pytestmark = [pytest.mark.api, pytest.mark.integration]


def paid_tests_enabled() -> bool:
    return os.getenv("RUN_PAID_API_INTEGRATION_TESTS") == "1"


def request() -> SemanticReviewRequest:
    return SemanticReviewRequest(
        event=FallEvent(
            track_id=1,
            source_id="paid-test",
            session_id="paid-test",
            start_frame=0,
            start_time=0,
        ),
        text_context="Synthetic test context. No image is attached. Return uncertain.",
    )


@pytest.mark.skipif(
    not paid_tests_enabled(), reason="paid API integration was not explicitly enabled"
)
def test_openai_real_structured_response() -> None:
    model = os.environ["OPENAI_TEST_MODEL"]
    result = OpenAIProvider(model).review(request())
    assert result.provider_success and result.schema_valid


@pytest.mark.skipif(
    not paid_tests_enabled(), reason="paid API integration was not explicitly enabled"
)
def test_deepseek_real_json_response() -> None:
    model = os.environ["DEEPSEEK_TEST_MODEL"]
    result = DeepSeekProvider(model, base_url="https://api.deepseek.com").review(request())
    assert result.provider_success and result.schema_valid
