from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
@pytest.mark.parametrize(
    "script",
    [
        "infer_image.py",
        "infer_video.py",
        "track_video.py",
        "run_pipeline.py",
        "train_detector.py",
        "evaluate_detector.py",
        "benchmark.py",
        "train_semantic_adapter.py",
        "validate_official_model.py",
        "prepare_gmdcsa24.py",
        "validate_grouped_pipeline.py",
        "check_api_configuration.py",
        "normalize_fallen_person.py",
        "prepare_fallen_person.py",
        "generate_threshold_candidates.py",
        "select_thresholds.py",
        "confirm_thresholds.py",
    ],
)
def test_required_cli_help_is_runnable(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.integration
def test_environment_check_is_local_and_successful() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/check_environment.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert '"network_or_paid_call_performed": false' in result.stdout


@pytest.mark.integration
def test_gradio_app_builds_without_weights_or_api_keys() -> None:
    pytest.importorskip("gradio", reason="optional UI extra is not installed")
    from app.gradio_app import build_app

    app = build_app()
    assert app is not None
