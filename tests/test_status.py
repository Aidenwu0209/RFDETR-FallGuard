from __future__ import annotations

import json

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


def test_threshold_status_requires_explicit_lock_and_confirmation(tmp_path) -> None:
    validation = tmp_path / "artifacts/validation"
    validation.mkdir(parents=True)
    (validation / "candidate.json").write_text(
        json.dumps({"validation_kind": "GROUPED_CLIP_LEVEL_INTERNAL_VALIDATION"}),
        encoding="utf-8",
    )
    status = environment_status(tmp_path)
    assert status["validation"]["grouped_reports"] == [str(validation / "candidate.json")]
    assert status["validation"]["thresholds_frozen"] is False
    assert status["validation"]["thresholds_confirmed_on_s3"] is False

    (validation / "threshold-lock.json").write_text(
        json.dumps({"lock_kind": "THRESHOLD_LOCK_PENDING_S3_CONFIRMATION"}),
        encoding="utf-8",
    )
    (validation / "threshold-confirmation.json").write_text(
        json.dumps({"confirmation_kind": "THRESHOLD_LOCK_CONFIRMED_ON_S3"}),
        encoding="utf-8",
    )
    status = environment_status(tmp_path)
    assert status["validation"]["thresholds_frozen"] is True
    assert status["validation"]["thresholds_confirmed_on_s3"] is True
