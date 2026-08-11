from __future__ import annotations

import pytest


@pytest.mark.integration
def test_current_openai_sdk_exposes_responses_structured_parse() -> None:
    openai = pytest.importorskip("openai", reason="optional cloud extra is not installed")
    client = openai.OpenAI(api_key="sk-local-contract-test")
    assert callable(client.responses.parse)
