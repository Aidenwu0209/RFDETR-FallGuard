#!/usr/bin/env python3
"""Run a repeatable real-GPU smoke benchmark for one official RF-DETR model."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2

from fallguard.config import load_config
from fallguard.detection.rfdetr_adapter import RFDETRDetector
from fallguard.schemas import FrameMetadata
from fallguard.video import write_image
from fallguard.visualization import draw_detections

OFFICIAL_MD5 = {
    "nano": "fb6504cce7fbdc783f7a46991f07639f",
    "small": "fb37061c1af7bace359c91b723a8d5c1",
}


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * proportion)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--variant", required=True, choices=("nano", "small"))
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--config", default="configs/profiles/development.yaml")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/real-smoke"))
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be >= 0 and iterations must be >= 1")
    if not args.image.is_file() or not args.weights.is_file():
        parser.error("image and weights must exist")

    md5 = _digest(args.weights, "md5")
    if md5 != OFFICIAL_MD5[args.variant]:
        expected_md5 = OFFICIAL_MD5[args.variant]
        parser.error(f"official weight MD5 mismatch for {args.variant}: {md5} != {expected_md5}")

    try:
        import torch
    except ImportError as exc:
        parser.error(f"PyTorch unavailable: {exc}")
    if not torch.cuda.is_available():
        parser.error("this validation requires a real CUDA device")

    config = load_config(args.config)
    detector_config = config.detector.model_copy(
        update={
            "model_variant": args.variant,
            "weights_path": args.weights,
            "allow_weight_download": False,
        }
    )
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"cannot read image: {args.image}")
    height, width = image.shape[:2]
    metadata = FrameMetadata(
        frame_id=0,
        timestamp_seconds=0,
        frame_width=width,
        frame_height=height,
        source_id="public-smoke-image",
        session_id=f"official-{args.variant}-smoke",
    )

    detector = RFDETRDetector(detector_config, device="cuda")
    load_started = time.perf_counter()
    detector.load()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    for _ in range(args.warmup):
        detector.predict_frame(image, metadata)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    durations_ms: list[float] = []
    detections = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        detections = detector.predict_frame(image, metadata)
        torch.cuda.synchronize()
        durations_ms.append((time.perf_counter() - started) * 1000)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated = args.output_dir / f"{args.variant}-bus-annotated.jpg"
    prediction_json = args.output_dir / f"{args.variant}-bus-detections.json"
    report_json = args.output_dir / f"{args.variant}-smoke-report.json"
    annotation_sha256 = write_image(annotated, draw_detections(image, detections))
    prediction_payload = [item.model_dump(mode="json") for item in detections]
    prediction_json.write_text(
        json.dumps(prediction_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report: dict[str, Any] = {
        "validation_kind": "REAL_SINGLE_IMAGE_GPU_SMOKE_NOT_FORMAL_BENCHMARK",
        "model_variant": args.variant,
        "weight_path": str(args.weights),
        "weight_bytes": args.weights.stat().st_size,
        "weight_md5": md5,
        "weight_sha256": _digest(args.weights, "sha256"),
        "image_path": str(args.image),
        "image_sha256": _digest(args.image, "sha256"),
        "image_width": width,
        "image_height": height,
        "confidence_threshold": detector_config.confidence_threshold,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "load_seconds": round(load_seconds, 4),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "latency_ms": {
            "mean": round(statistics.fmean(durations_ms), 3),
            "median": round(statistics.median(durations_ms), 3),
            "p95_nearest_rank": round(_percentile(durations_ms, 0.95), 3),
            "minimum": round(min(durations_ms), 3),
            "maximum": round(max(durations_ms), 3),
        },
        "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "detection_count": len(detections),
        "class_counts": {
            name: sum(item.class_name == name for item in detections)
            for name in sorted({item.class_name for item in detections})
        },
        "prediction_json": str(prediction_json),
        "annotated_image": str(annotated),
        "annotated_image_sha256": annotation_sha256,
        "paid_api_call_performed": False,
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
