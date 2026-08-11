# RFDETR-FallGuard

An auditable fall-monitoring cascade built around official RF-DETR, class-agnostic person
tracking, temporal event confirmation, provider-agnostic semantic review, and application-owned
alerts.

> Current scope: engineering infrastructure and deterministic mock validation. No trained
> posture checkpoint, dataset, final threshold, or scientific metric is bundled.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/check_environment.py
python scripts/run_pipeline.py \
  --mode mock \
  --config configs/profiles/development.yaml \
  --output-dir artifacts/mock-run
```

Optional integrations are installed explicitly:

```bash
python -m pip install -e '.[rfdetr,tracking]'
python -m pip install -e '.[ui]'
python -m pip install -e '.[cloud]'
python -m pip install -e '.[local-vlm]'
```

Do not install/download a model merely to make a command appear successful. See
`BLOCKERS.md`, `IMPLEMENTATION_STATUS.md`, and `docs/PROJECT_SPEC.md` for evidence boundaries.

## What the mock command proves

The command exercises strict tracked detections, track history, second-based temporal features,
state transitions, one event lifecycle, bounded keyframe selection, Mock semantic assessment,
and application-owned alert output. It writes `events.jsonl`, `alerts.jsonl`, and selected person
crops. Output is visibly labeled `MOCK` and is not eligible for a formal benchmark.

## Real inference

Real commands never download weights implicitly. Supply an existing approved checkpoint:

```bash
python scripts/infer_image.py input.jpg \
  --weights weights/rf-detr-small.pth \
  --output-json artifacts/image-detections.json \
  --output-image artifacts/image-annotated.jpg

python scripts/infer_video.py input.mp4 \
  --weights weights/rf-detr-small.pth \
  --output-jsonl artifacts/video-detections.jsonl \
  --output-video artifacts/video-annotated.mp4

python scripts/track_video.py input.mp4 \
  --weights weights/rf-detr-small.pth \
  --output-jsonl artifacts/tracks.jsonl

python scripts/run_pipeline.py \
  --mode real \
  --video input.mp4 \
  --weights weights/rf-detr-small.pth \
  --config configs/profiles/development.yaml \
  --output-dir artifacts/real-run \
  --ablation full
```

Official COCO-pretrained weights belong in `person_only` mode and only prove person detection.
`posture_multiclass` requires a fine-tuned checkpoint plus class metadata.

## Training and evaluation entry points

Detector training prints its resolved official parameter names by default. It performs training
only with `--execute`:

```bash
python scripts/train_detector.py \
  --dataset-dir datasets/processed/fall-coco \
  --config configs/profiles/experiment.yaml

python scripts/evaluate_detector.py \
  --dataset-dir datasets/processed/fall-coco \
  --weights weights/fallguard-small.pth \
  --split test

python scripts/train_semantic_adapter.py \
  --config configs/qlora.yaml \
  --allow-external-blockers
```

Detection AP is delegated to official RF-DETR evaluation. Event matching is available through
`fallguard.evaluation.event` and requires explicit temporal thresholds. The experiment profile
will reject formal benchmarking until validation-derived thresholds and posture class metadata
are supplied:

```bash
python scripts/benchmark.py \
  --mode formal \
  --config configs/profiles/experiment.yaml \
  --video input.mp4
```

For non-scientific timing of the benchmark harness itself:

```bash
python scripts/benchmark.py --mode engineering-mock
```

Formal ablations use `--ablation detector_only`, `detector_tracking`, `temporal`, or `full`.
The output separates sustained end-to-end FPS, per-stage latency/derived stage throughput,
video-time event-trigger latency, provider-reported latency/tokens, memory, and decode/UI/network
inclusion. Warm-up observations are reset before measured stage summaries.

## Gradio prototype

```bash
python -m pip install -e '.[ui]'
python app/gradio_app.py --host 127.0.0.1 --port 7860
python scripts/smoke_gradio.py
```

Upload and webcam inputs are processed as finite clips. This repository does not call that
continuous real-time monitoring. Real mode returns a friendly configuration error when no
checkpoint is present. Cloud images require both configuration and a per-session consent box.

## Quality commands

```bash
python -m pytest -q
python scripts/check_environment.py
python -m compileall -q src app scripts tests datasets
ruff check .
mypy src

python scripts/infer_image.py --help
python scripts/infer_video.py --help
python scripts/track_video.py --help
python scripts/run_pipeline.py --help
python scripts/train_detector.py --help
python scripts/evaluate_detector.py --help
python scripts/benchmark.py --help
python scripts/train_semantic_adapter.py --help
```

Real cloud integration tests are excluded by default and additionally gated by:

```bash
RUN_PAID_API_INTEGRATION_TESTS=1 python -m pytest -m api
```

This opt-in may incur provider charges. It is never enabled by a health check.

## Documentation

- `docs/architecture.md`: module and dependency boundaries.
- `docs/temporal_design.md`: coordinate, unit, smoothing, timeout, and transition definitions.
- `docs/semantic_review.md`: provider capabilities, privacy, fallback, and alert ownership.
- `docs/evaluation_protocol.md`: one-to-one event matching and metric availability.
- `docs/rfdetr_adapter.md`: official API mapping and unsupported settings.
- `docs/qlora.md`: manifests, dry-run, and explicit execution boundary.
- `docs/privacy.md`: retention, logging, UI, and cloud consent.
- `docs/third_party_versions.md`: current API/version verification evidence.
