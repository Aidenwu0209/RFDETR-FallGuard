#!/usr/bin/env python3
"""Run an engineering mock timing or a gated real-video formal benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fallguard.config import load_config
from fallguard.evaluation.deployment import DeploymentBenchmark
from fallguard.exceptions import FormalBenchmarkRejectedError
from fallguard.factory import build_pipeline
from fallguard.pipeline import AblationMode
from fallguard.session import make_session_id
from fallguard.timing import TimingCollector
from fallguard.video import FramePacket, VideoReader


class _MockWorkload:
    component_kind = "mock"

    @staticmethod
    def operation() -> int:
        return sum(value * value for value in range(1000))


def engineering_mock(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    workload = _MockWorkload()
    benchmark = DeploymentBenchmark(config, [workload])
    result = benchmark.run(workload.operation)
    result["label"] = "MOCK_ENGINEERING_ONLY"
    result["formal_benchmark_eligible"] = False
    return result


def formal_video(
    config_path: str,
    video: Path,
    keyframe_dir: Path,
    ablation_mode: AblationMode = "full",
) -> dict[str, Any]:
    config = load_config(config_path)
    config.assert_formal_ready()
    pipeline = build_pipeline(
        config,
        with_real_frontend=True,
        keyframe_output_dir=keyframe_dir,
        ablation_mode=ablation_mode,
    )
    assert pipeline.detector is not None and pipeline.tracker is not None
    pipeline.detector.load()
    pipeline.tracker.load()
    total = config.benchmark.warmup_iterations + config.benchmark.measured_iterations
    session_id = config.runtime.session_id or make_session_id(video.stem)
    reader = VideoReader(video, source_id=config.runtime.source_id, session_id=session_id)
    reader.__enter__()
    stream = reader.frames()
    buffered: list[FramePacket] | None = None
    if not config.benchmark.include_decode:
        buffered = []
        for _ in range(total):
            try:
                buffered.append(next(stream))
            except StopIteration as exc:
                reader.close()
                raise ValueError(f"video needs at least {total} frames") from exc
        reader.close()
    index = 0
    semantic_latency = TimingCollector()
    input_tokens_total = 0
    output_tokens_total = 0
    semantic_calls = 0

    def operation() -> None:
        nonlocal index, input_tokens_total, output_tokens_total, semantic_calls
        if buffered is not None:
            packet = buffered[index]
        else:
            try:
                packet = next(stream)
            except StopIteration as exc:
                raise ValueError(f"video needs at least {total} frames") from exc
        index += 1
        pipeline_result = pipeline.process_frame(packet.image_bgr, packet.metadata)
        for assessment in pipeline_result.semantic_assessments:
            semantic_calls += 1
            semantic_latency.record("provider_reported", assessment.latency_ms)
            input_tokens_total += assessment.input_tokens or 0
            output_tokens_total += assessment.output_tokens or 0

    def reset_measurement_state() -> None:
        nonlocal input_tokens_total, output_tokens_total, semantic_calls
        pipeline.timing.reset()
        semantic_latency.reset()
        input_tokens_total = 0
        output_tokens_total = 0
        semantic_calls = 0

    try:
        result = DeploymentBenchmark(
            config,
            [pipeline, pipeline.detector, pipeline.tracker],
        ).run(operation, before_measurement=reset_measurement_state)
    finally:
        reader.close()
    result["label"] = "FORMAL_REAL_COMPONENTS"
    result["ablation_mode"] = ablation_mode
    stage_summary = pipeline.timing.summary()
    result["stage_latency"] = {name: summary.__dict__ for name, summary in stage_summary.items()}
    result["stage_throughput_fps"] = {
        name: 1000.0 / summary.mean_ms
        for name, summary in stage_summary.items()
        if name in {"detection", "tracking", "temporal"}
        and summary.mean_ms is not None
        and summary.mean_ms > 0
    }
    provider_summary = semantic_latency.summary().get("provider_reported")
    result["semantic_provider"] = {
        "calls": semantic_calls,
        "reported_latency": provider_summary.__dict__ if provider_summary else None,
        "input_tokens_total_when_reported": input_tokens_total,
        "output_tokens_total_when_reported": output_tokens_total,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("engineering-mock", "formal"), required=True)
    parser.add_argument("--config")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--keyframe-dir", type=Path, default=Path("artifacts/benchmark-keyframes"))
    parser.add_argument(
        "--ablation",
        choices=("detector_only", "detector_tracking", "temporal", "full"),
        default="full",
    )
    args = parser.parse_args()
    if args.mode == "engineering-mock":
        result = engineering_mock(args.config or "configs/profiles/development.yaml")
    else:
        if args.video is None:
            parser.error("--video is required for formal mode")
        try:
            result = formal_video(
                args.config or "configs/profiles/experiment.yaml",
                args.video,
                args.keyframe_dir,
                args.ablation,
            )
        except FormalBenchmarkRejectedError as exc:
            print(
                json.dumps(
                    {"state": "BLOCKED_EXTERNAL", "formal_benchmark_rejected": str(exc)},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from None
    rendered = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
