from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_semantic_provider_matrix",
    PROJECT_ROOT / "scripts/evaluate_semantic_provider_matrix.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_matrix_has_balanced_decisions_and_exact_requested_models() -> None:
    assert len(MODULE.CASES) == 8
    assert {case.expected_decision for case in MODULE.CASES} == {
        "fall",
        "not_fall",
        "uncertain",
    }
    assert [(spec.model, spec.reasoning_effort) for spec in MODULE.DEFAULT_MODELS] == [
        ("gpt-5.6-luna", "max"),
        ("gpt-5.6-terra", "xhigh"),
        ("gpt-5.6-sol", "high"),
        ("deepseek-v4-flash", "high"),
    ]


def test_score_respects_uncertain_attempt_as_unscored() -> None:
    case = next(case for case in MODULE.CASES if case.case_id == "track_gap_during_descent")
    judgment = MODULE.Judgment(
        decision="uncertain",
        confidence=0.5,
        reason="Identity continuity is unavailable.",
        attempt_to_stand=True,
        risk_level="unknown",
        model_recommends_alert=False,
    )
    assert MODULE.score(case, judgment) == {
        "decision_correct": True,
        "alert_correct": True,
        "attempt_correct": True,
        "risk_correct": True,
    }


def test_summarize_counts_schema_and_field_accuracy() -> None:
    rows = [
        {
            "status": "success",
            "latency_ms": 10.0,
            "usage": {"input_tokens": 4, "output_tokens": 2, "reasoning_tokens": 1},
            "score": {
                "decision_correct": True,
                "alert_correct": True,
                "attempt_correct": True,
                "risk_correct": False,
            },
        },
        {"status": "error", "error_type": "TimeoutError", "error": "timed out"},
    ]
    summary = MODULE.summarize(rows)
    assert summary["schema_success_rate"] == 0.5
    assert summary["decision_accuracy"] == 1.0
    assert summary["risk_accuracy"] == 0.0
    assert summary["all_scored_fields_accuracy"] == 0.0
    assert summary["tokens"] == {
        "input_tokens": 4,
        "output_tokens": 2,
        "reasoning_tokens": 1,
    }
