from __future__ import annotations

import pytest

from fallguard.evaluation.deployment import DeploymentBenchmark
from fallguard.evaluation.event import EventEvaluator, EventMatchProtocol, TimeWindow
from fallguard.exceptions import FormalBenchmarkRejectedError
from fallguard.schemas import DetectionMode, GroundTruthEvent, PredictedEvent

pytestmark = pytest.mark.unit


def ground_truth(event_id: str, start: float, end: float, track: int = 1):
    return GroundTruthEvent(
        event_id=event_id,
        source_id="source",
        session_id="session",
        start_time=start,
        end_time=end,
        track_id=track,
    )


def prediction(event_id: str, start: float, end: float, track: int = 1):
    return PredictedEvent(
        event_id=event_id,
        source_id="source",
        session_id="session",
        start_time=start,
        end_time=end,
        track_id=track,
    )


def test_event_matching_is_one_to_one_and_reports_unavailable_metrics() -> None:
    evaluator = EventEvaluator(
        EventMatchProtocol(temporal_iou_min=0.3, start_tolerance_seconds=0.5)
    )
    truth = [ground_truth("g1", 1, 3), ground_truth("g2", 10, 12)]
    predicted = [
        prediction("p1", 1.1, 3.1),
        prediction("duplicate", 1.2, 2.9),
        prediction("p2", 10.1, 12.1),
    ]
    metrics = evaluator.evaluate(truth, predicted, monitored_hours=0.5)
    assert metrics["true_positive_events"] == 2
    assert metrics["false_positive_events"] == 1
    assert metrics["false_alarms_per_hour"] == 2
    assert metrics["specificity"] is None
    assert metrics["specificity_unavailable_reason"] == "negative event windows were not defined"


def test_specificity_requires_explicit_negative_windows() -> None:
    evaluator = EventEvaluator(EventMatchProtocol(0.3, None))
    windows = [
        TimeWindow("source", "session", 20, 25),
        TimeWindow("source", "session", 30, 35),
    ]
    metrics = evaluator.evaluate(
        [],
        [prediction("fp", 21, 22)],
        negative_windows=windows,
    )
    assert metrics["specificity"] == 0.5


class MockComponent:
    component_kind = "mock"


def test_formal_benchmark_rejects_mock_component(development_config) -> None:
    config = development_config.model_copy(deep=True)
    config.benchmark.formal = True
    config.detector.mode = DetectionMode.POSTURE_MULTICLASS
    config.detector.class_names = {0: "standing", 1: "falling"}
    config.semantic.provider = "deepseek"
    config.semantic.model = "configured-model"
    config.semantic.allow_mock = False
    config.semantic.allow_fallback = False
    benchmark = DeploymentBenchmark(config, [MockComponent()])
    with pytest.raises(FormalBenchmarkRejectedError, match="mock components"):
        benchmark.run(lambda: None)
