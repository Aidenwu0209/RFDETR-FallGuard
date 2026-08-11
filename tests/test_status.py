from __future__ import annotations

import pytest

from fallguard.status import environment_status

pytestmark = pytest.mark.unit


def test_environment_status_does_not_reveal_key_values(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "never-render-this-secret")
    status = environment_status(tmp_path)
    rendered = str(status)
    assert status["api_keys"]["OPENAI_API_KEY"] == {"present": True}
    assert "never-render-this-secret" not in rendered
    assert status["api_keys"]["network_or_paid_call_performed"] is False
