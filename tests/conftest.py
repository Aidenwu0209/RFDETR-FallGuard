from __future__ import annotations

from pathlib import Path

import pytest

from fallguard.config import AppConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def development_config() -> AppConfig:
    return load_config(
        PROJECT_ROOT / "configs/profiles/development.yaml",
        base_path=PROJECT_ROOT / "configs/base.yaml",
    )
