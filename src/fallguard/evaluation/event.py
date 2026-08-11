"""Protocol-first, one-to-one fall event evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from fallguard.exceptions import EvaluationProtocolError
from fallguard.schemas import GroundTruthEvent, PredictedEvent


@dataclass(frozen=True)
class TimeWindow:
    source_id: str
    session_id: str
    start_time: float
    end_time: float

    def __post_init__(self) -> None:
        if self.start_time < 0 or self.end_time <= self.start_time:
            raise ValueError("time window must have non-negative start and positive duration")


@dataclass(frozen=True)
class EventMatchProtocol:
    temporal_iou_min: float | None
    start_tolerance_seconds: float | None
    threshold_rule: Literal["and", "or"] = "or"
    track_handling: Literal["same_track", "ignore_track"] = "same_track"

    def __post_init__(self) -> None:
        if self.temporal_iou_min is None and self.start_tolerance_seconds is None:
            raise EvaluationProtocolError("at least one event matching threshold is required")
        if self.temporal_iou_min is not None and not 0 <= self.temporal_iou_min <= 1:
            raise EvaluationProtocolError("temporal_iou_min must be in [0, 1]")
        if self.start_tolerance_seconds is not None and self.start_tolerance_seconds < 0:
            raise EvaluationProtocolError("start_tolerance_seconds cannot be negative")


@dataclass(frozen=True)
class EventMatch:
    ground_truth_id: str
    predicted_id: str
    temporal_iou: float
    start_delay_seconds: float


def temporal_iou(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    intersection = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    union = max(first_end, second_end) - min(first_start, second_start)
    return intersection / union if union > 0 else 0.0


class EventEvaluator:
    def __init__(self, protocol: EventMatchProtocol) -> None:
        self.protocol = protocol

    def match(
        self,
        ground_truth: list[GroundTruthEvent],
        predictions: list[PredictedEvent],
    ) -> list[EventMatch]:
        candidates: list[tuple[float, float, GroundTruthEvent, PredictedEvent]] = []
        for truth in ground_truth:
            for prediction in predictions:
                if (truth.source_id, truth.session_id) != (
                    prediction.source_id,
                    prediction.session_id,
                ):
                    continue
                if (
                    self.protocol.track_handling == "same_track"
                    and truth.track_id is not None
                    and prediction.track_id is not None
                    and truth.track_id != prediction.track_id
                ):
                    continue
                tiou = temporal_iou(
                    truth.start_time,
                    truth.end_time,
                    prediction.start_time,
                    prediction.end_time,
                )
                start_error = abs(prediction.start_time - truth.start_time)
                if self._eligible(tiou, start_error):
                    candidates.append((tiou, -start_error, truth, prediction))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        used_truth: set[str] = set()
        used_predictions: set[str] = set()
        matches: list[EventMatch] = []
        for tiou, _, truth, prediction in candidates:
            if truth.event_id in used_truth or prediction.event_id in used_predictions:
                continue
            used_truth.add(truth.event_id)
            used_predictions.add(prediction.event_id)
            matches.append(
                EventMatch(
                    ground_truth_id=truth.event_id,
                    predicted_id=prediction.event_id,
                    temporal_iou=tiou,
                    start_delay_seconds=prediction.start_time - truth.start_time,
                )
            )
        return matches

    def evaluate(
        self,
        ground_truth: list[GroundTruthEvent],
        predictions: list[PredictedEvent],
        *,
        monitored_hours: float | None = None,
        negative_windows: list[TimeWindow] | None = None,
    ) -> dict[str, object]:
        if monitored_hours is not None and monitored_hours <= 0:
            raise EvaluationProtocolError("monitored_hours must be positive")
        matches = self.match(ground_truth, predictions)
        true_positive = len(matches)
        false_positive = len(predictions) - true_positive
        false_negative = len(ground_truth) - true_positive
        precision = self._safe_div(true_positive, true_positive + false_positive)
        recall = self._safe_div(true_positive, true_positive + false_negative)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        specificity, specificity_reason = self._specificity(predictions, negative_windows)
        delays = [match.start_delay_seconds for match in matches]
        return {
            "protocol": asdict(self.protocol),
            "true_positive_events": true_positive,
            "false_positive_events": false_positive,
            "false_negative_events": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "miss_rate": self._safe_div(false_negative, len(ground_truth)),
            "specificity": specificity,
            "specificity_unavailable_reason": specificity_reason,
            "false_alarms_per_hour": (
                false_positive / monitored_hours if monitored_hours is not None else None
            ),
            "false_alarms_per_hour_unavailable_reason": (
                None if monitored_hours is not None else "monitored duration was not supplied"
            ),
            "mean_detection_delay_seconds": sum(delays) / len(delays) if delays else None,
            "matches": [asdict(match) for match in matches],
        }

    def _eligible(self, tiou: float, start_error: float) -> bool:
        checks = []
        if self.protocol.temporal_iou_min is not None:
            checks.append(tiou >= self.protocol.temporal_iou_min)
        if self.protocol.start_tolerance_seconds is not None:
            checks.append(start_error <= self.protocol.start_tolerance_seconds)
        return all(checks) if self.protocol.threshold_rule == "and" else any(checks)

    @staticmethod
    def _safe_div(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    @staticmethod
    def _specificity(
        predictions: list[PredictedEvent],
        negative_windows: list[TimeWindow] | None,
    ) -> tuple[float | None, str | None]:
        if negative_windows is None:
            return None, "negative event windows were not defined"
        false_positive_windows = 0
        for window in negative_windows:
            hit = any(
                prediction.source_id == window.source_id
                and prediction.session_id == window.session_id
                and temporal_iou(
                    prediction.start_time,
                    prediction.end_time,
                    window.start_time,
                    window.end_time,
                )
                > 0
                for prediction in predictions
            )
            false_positive_windows += int(hit)
        true_negative_windows = len(negative_windows) - false_positive_windows
        denominator = true_negative_windows + false_positive_windows
        return (true_negative_windows / denominator if denominator else None), None
