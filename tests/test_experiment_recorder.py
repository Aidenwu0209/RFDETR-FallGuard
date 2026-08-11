from __future__ import annotations

import json

import pytest

from fallguard.experiment import ExperimentRecorder

pytestmark = pytest.mark.integration


def test_experiment_recorder_captures_reproducibility_and_mock_exclusion(
    development_config,
    tmp_path,
) -> None:
    run_dir = ExperimentRecorder(tmp_path).create(
        name="mock-fixture",
        config=development_config,
        mock_components=["MockProvider"],
        notes="synthetic fixture only",
    )
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["formal_result_eligible"] is False
    assert environment["mock_components"] == ["MockProvider"]
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "git.json").is_file()
    assert (run_dir / "environment-freeze.txt").is_file()
