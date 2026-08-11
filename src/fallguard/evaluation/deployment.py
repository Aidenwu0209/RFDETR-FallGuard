"""Deployment benchmark gates and resource-aware timing."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any, Protocol

import psutil

from fallguard.config import AppConfig
from fallguard.exceptions import FormalBenchmarkRejectedError
from fallguard.timing import TimingCollector


class BenchmarkComponent(Protocol):
    component_kind: str


class DeploymentBenchmark:
    def __init__(self, config: AppConfig, components: Sequence[BenchmarkComponent]) -> None:
        self.config = config
        self.components = components

    def run(
        self,
        operation: Callable[[], Any],
        *,
        before_measurement: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        if self.config.benchmark.formal:
            self.config.assert_formal_ready()
            mocks = [
                type(item).__name__ for item in self.components if item.component_kind == "mock"
            ]
            if mocks:
                raise FormalBenchmarkRejectedError(
                    "formal benchmark contains mock components: " + ", ".join(mocks)
                )
        torch = self._optional_torch()
        use_cuda = bool(torch is not None and torch.cuda.is_available())
        if use_cuda and torch is not None:
            torch.cuda.reset_peak_memory_stats()
        for _ in range(self.config.benchmark.warmup_iterations):
            operation()
        if use_cuda and torch is not None:
            torch.cuda.synchronize()
        if before_measurement is not None:
            before_measurement()
        process = psutil.Process()
        rss_before = process.memory_info().rss
        timings = TimingCollector()
        started = time.perf_counter()
        for _ in range(self.config.benchmark.measured_iterations):
            iteration = time.perf_counter()
            operation()
            if use_cuda and torch is not None:
                torch.cuda.synchronize()
            timings.record("iteration", (time.perf_counter() - iteration) * 1000.0)
        total_seconds = time.perf_counter() - started
        rss_after = process.memory_info().rss
        summary = timings.summary()["iteration"]
        return {
            "formal": self.config.benchmark.formal,
            "mock_excluded": all(item.component_kind != "mock" for item in self.components),
            "iterations": self.config.benchmark.measured_iterations,
            "sustained_fps": self.config.benchmark.measured_iterations / total_seconds,
            "iteration_latency": asdict(summary),
            "decode_included": self.config.benchmark.include_decode,
            "ui_included": self.config.benchmark.include_ui,
            "network_included": self.config.benchmark.include_network,
            "cpu_rss_delta_bytes": rss_after - rss_before,
            "gpu_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if use_cuda and torch is not None else None
            ),
            "gpu_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if use_cuda and torch is not None else None
            ),
        }

    @staticmethod
    def _optional_torch() -> Any | None:
        try:
            import torch
        except ImportError:
            return None
        return torch
