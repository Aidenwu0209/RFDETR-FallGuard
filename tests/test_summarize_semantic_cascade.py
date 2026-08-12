from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_semantic_cascade",
    PROJECT_ROOT / "scripts/summarize_semantic_cascade.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: dict[str, object]) -> tuple[Path, dict[str, object]]:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def frontend(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "model_variant": "small",
        "weights_sha256": "weights",
        "config_sha256": "config",
        "pipeline_implementation_sha256": "implementation",
        "rows": rows,
    }


def test_summary_includes_clips_that_bypass_semantic_stage(tmp_path: Path) -> None:
    report_path, report = write_json(
        tmp_path / "frontend.json",
        frontend(
            [
                {
                    "video_id": "fall",
                    "subject_id": 1,
                    "expected_fall": True,
                    "predicted_fall": True,
                    "predicted_event_count": 1,
                },
                {
                    "video_id": "adl",
                    "subject_id": 1,
                    "expected_fall": False,
                    "predicted_fall": False,
                    "predicted_event_count": 0,
                },
            ]
        ),
    )
    semantic_path, semantic = write_json(
        tmp_path / "semantic.json",
        {
            "model": "Qwen3.5-4B",
            "model_revision": "revision",
            "rows": [
                {
                    "candidate_id": "fall#event-1",
                    "video_id": "fall",
                    "expected": "fall",
                    "predicted": "fall",
                }
            ],
        },
    )
    summary = MODULE.summarize([(report_path, report)], semantic_path, semantic)
    assert summary["metrics_against_weak_clip_labels"] == {
        "clips": 2,
        "true_positive_clips": 1,
        "false_positive_clips": 0,
        "false_negative_clips": 0,
        "true_negative_clips": 1,
        "precision": 1.0,
        "recall": 1.0,
        "specificity": 1.0,
        "f1": 1.0,
    }
    assert summary["stage_counts"]["frontend_input_clips"] == 2


def test_summary_rejects_missing_event_review(tmp_path: Path) -> None:
    report_path, report = write_json(
        tmp_path / "frontend.json",
        frontend(
            [
                {
                    "video_id": "fall",
                    "subject_id": 1,
                    "expected_fall": True,
                    "predicted_fall": True,
                    "predicted_event_count": 2,
                }
            ]
        ),
    )
    semantic_path, semantic = write_json(
        tmp_path / "semantic.json",
        {
            "rows": [
                {
                    "candidate_id": "fall#event-1",
                    "video_id": "fall",
                    "expected": "fall",
                    "predicted": "fall",
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="candidate count"):
        MODULE.summarize([(report_path, report)], semantic_path, semantic)
