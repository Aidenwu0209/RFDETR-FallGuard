"""Stage timing utilities based on a monotonic high-resolution clock."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass(frozen=True)
class TimingSummary:
    count: int
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    maximum_ms: float | None


class _Timer(AbstractContextManager["_Timer"]):
    def __init__(self, collector: TimingCollector, stage: str) -> None:
        self.collector = collector
        self.stage = stage
        self.started = 0.0

    def __enter__(self) -> _Timer:
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.collector.record(self.stage, (time.perf_counter() - self.started) * 1000.0)


class TimingCollector:
    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)

    def time(self, stage: str) -> _Timer:
        return _Timer(self, stage)

    def record(self, stage: str, latency_ms: float) -> None:
        if latency_ms < 0:
            raise ValueError("latency cannot be negative")
        self._values[stage].append(latency_ms)

    def reset(self) -> None:
        self._values.clear()

    def summary(self) -> dict[str, TimingSummary]:
        result: dict[str, TimingSummary] = {}
        for stage, values in self._values.items():
            result[stage] = TimingSummary(
                count=len(values),
                mean_ms=sum(values) / len(values) if values else None,
                p50_ms=percentile(values, 0.50),
                p95_ms=percentile(values, 0.95),
                maximum_ms=max(values) if values else None,
            )
        return result
