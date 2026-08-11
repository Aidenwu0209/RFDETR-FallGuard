# RFDETR-FallGuard

An auditable fall-monitoring cascade built around official RF-DETR, class-agnostic person
tracking, temporal event confirmation, provider-agnostic semantic review, and application-owned
alerts.

> Current scope: real official Nano/Small person-detection smoke validation, audited posture-data
> and short-training entry points, plus grouped GMDCSA-24 preparation. No real-data posture
> checkpoint, final threshold, or scientific accuracy claim is bundled.

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

On the validated Legion runtime, install detector training support with:

```bash
python -m pip install 'rfdetr[train]==1.9.1'
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
  --model-variant small \
  --weights weights/official/rf-detr-small.pth \
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

Repeat the hash-gated real GPU smoke check for each official model:

```bash
python scripts/validate_official_model.py data/smoke/bus.jpg \
  --variant nano --weights weights/official/rf-detr-nano.pth
python scripts/validate_official_model.py data/smoke/bus.jpg \
  --variant small --weights weights/official/rf-detr-small.pth
```

## Training and evaluation entry points

First audit an extracted Roboflow COCO export. The audit verifies images, annotations, class IDs,
hashes, bounding boxes, and cross-split exact/near duplicates, then emits the only class-order
profile accepted by the training gate:

```bash
python scripts/prepare_fallen_person.py \
  --dataset-dir data/raw/fallen-person \
  --output-dir data/processed/fallen-person
```

Detector training prints its resolved official parameter names by default and performs training
only with `--execute`. Nano and Small use the same audited class mapping and data:

```bash
python scripts/train_detector.py \
  --dataset-dir data/raw/fallen-person \
  --dataset-audit data/processed/fallen-person/audit.json \
  --config data/processed/fallen-person/posture_profile.yaml \
  --weights weights/official/rf-detr-nano.pth \
  --model-variant nano \
  --output-dir checkpoints/nano \
  --epochs 2 --batch-size 1 --grad-accum-steps 4 --execute

python scripts/train_detector.py \
  --dataset-dir data/raw/fallen-person \
  --dataset-audit data/processed/fallen-person/audit.json \
  --config data/processed/fallen-person/posture_profile.yaml \
  --weights weights/official/rf-detr-small.pth \
  --model-variant small \
  --output-dir checkpoints/small \
  --epochs 2 --batch-size 1 --grad-accum-steps 4 --execute

python scripts/evaluate_detector.py \
  --dataset-dir data/raw/fallen-person \
  --weights checkpoints/small/checkpoint_best_total.pth \
  --split test

python scripts/train_semantic_adapter.py \
  --config configs/qlora.yaml \
  --allow-external-blockers
```

Prepare the subject-isolated GMDCSA-24 manifests after its official archive is downloaded:

```bash
python scripts/prepare_gmdcsa24.py \
  --archive data/raw/gmdcsa24/GMDCSA24-v2.0.zip \
  --extract-dir data/raw/gmdcsa24/extracted \
  --output-dir data/processed/gmdcsa24
```

The grouped runner requires an actual posture checkpoint and a config whose class order matches
checkpoint metadata. Generate the four provisional candidates before looking at grouped results:

```bash
python scripts/generate_threshold_candidates.py \
  --base-config data/processed/fallen-person/posture_profile.yaml \
  --output-dir artifacts/validation/candidates
```

Run every generated candidate against the same deterministic S1-S2 subset for both Nano and
Small. The example below shows one of the eight runs; change only the candidate, variant, weights,
and output filename:

```bash
python scripts/validate_grouped_pipeline.py \
  --config artifacts/validation/candidates/high_recall.yaml \
  --manifest data/processed/gmdcsa24/manifest.json \
  --dataset-root data/raw/gmdcsa24/extracted/REPOSITORY_ROOT \
  --weights checkpoints/nano/checkpoint_best_total.pth \
  --model-variant nano \
  --partition threshold_development \
  --output-json artifacts/validation/development/nano-high_recall.json
```

Freeze one candidate using S1-S2 only. Supply every development report with a repeated
`--development-report`; the strict defaults require recall 1.0 and zero false-positive clips and
will fail rather than silently relax the gate:

```bash
python scripts/select_thresholds.py \
  --development-report artifacts/validation/development/nano-high_recall.json \
  --development-report artifacts/validation/development/nano-balanced_short.json \
  --development-report artifacts/validation/development/nano-balanced_duration.json \
  --development-report artifacts/validation/development/nano-high_precision.json \
  --development-report artifacts/validation/development/small-high_recall.json \
  --development-report artifacts/validation/development/small-balanced_short.json \
  --development-report artifacts/validation/development/small-balanced_duration.json \
  --development-report artifacts/validation/development/small-high_precision.json \
  --output-lock artifacts/validation/threshold-lock.json \
  --output-config artifacts/validation/frozen-profile.yaml
```

Run the frozen winner once on S3, then confirm without retuning:

```bash
python scripts/validate_grouped_pipeline.py \
  --config artifacts/validation/frozen-profile.yaml \
  --manifest data/processed/gmdcsa24/manifest.json \
  --dataset-root data/raw/gmdcsa24/extracted/REPOSITORY_ROOT \
  --partition threshold_validation \
  --output-json artifacts/validation/frozen-s3.json

python scripts/confirm_thresholds.py \
  --threshold-lock artifacts/validation/threshold-lock.json \
  --validation-report artifacts/validation/frozen-s3.json \
  --output-json artifacts/validation/threshold-confirmation.json
```

S4 remains inaccessible without the explicit unlock flag, the matching S3 confirmation artifact,
and `--all-videos`. Grouped reports contain clip-level metrics only; no detection-delay claim is
made without human-confirmed onset timestamps.

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
The UI also exposes read-only model/GPU smoke reports, grouped dataset status, training/threshold
readiness, and API-key presence. API values are never rendered and refresh performs no network or
paid call.

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
python scripts/prepare_fallen_person.py --help
python scripts/generate_threshold_candidates.py --help
python scripts/select_thresholds.py --help
python scripts/confirm_thresholds.py --help
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
