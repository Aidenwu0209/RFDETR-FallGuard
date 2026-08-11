# Engineering Decisions

## D-001: empty-origin bootstrap

The public GitHub origin was empty on 2026-08-11. The implementation therefore starts
from a minimal `src/` package instead of integrating code that does not exist. The supplied
execution objective and graduation-project attachment define the initial specification.

## D-002: authoritative dependency source

`pyproject.toml` is the only hand-maintained dependency source. There is no independent
`requirements.txt`. A machine-generated environment freeze may be stored with experiment
artifacts, but it must not become a second editable dependency list.

## D-003: pinned adapters

The RF-DETR adapter targets `rfdetr==1.9.1`; the tracking adapter targets
`supervision==0.30.0`. Their wheel source/signatures were inspected before implementation.
The project never rewrites either model or tracker.

Supervision 0.30.0 emits an upstream deprecation warning for `ByteTrack` and announces removal in
0.31.0. The exact tested pin is retained for reproducibility; migration is a separate deliberate
compatibility task, not a silent dependency upgrade.

## D-004: model and posture semantics

`person_only` is an engineering smoke mode for official COCO-pretrained weights and cannot
produce the class `falling`. `posture_multiclass` requires class metadata from configuration
or a dataset. Person identity association is class-agnostic; posture remains an observation.

## D-005: provisional thresholds

Values in `configs/profiles/development.yaml` exist only for tests and UI integration.
`configs/profiles/experiment.yaml` leaves all unvalidated temporal thresholds null. Formal
evaluation or benchmarking must fail fast until a validation protocol supplies them.

## D-006: privacy and external services

Cloud image transmission, paid connectivity tests, model downloads, and fallback are opt-in.
Health checks are local by default. Full frames are not retained by default, and secrets,
authorization headers, Base64 images, and full request bodies must not be logged.

## D-007: edit and validation surfaces

Files are authored in the local workspace with patches, synchronized without `.git` to the
Legion clone, and validated on the Legion. The Legion is the runtime evidence source.

## D-008: run identity

An explicitly configured `runtime.session_id` is preserved for reproducible experiments.
Otherwise each CLI run combines the input stem with a random suffix. This prevents repeated runs
of the same video and tracker counters restarting at `1` from sharing event identity.

## D-009: benchmark measurement boundary

Warm-up frames are processed and CUDA is synchronized before measurement. Stage collectors are
then reset. Sustained end-to-end FPS is kept distinct from throughput derived from mean stage
latency, and event-trigger latency is labeled as video time rather than compute time.

## D-010: provider failure boundary

Providers normalize expected dependency, model, HTTP, OOM/runtime, file, and structured-output
failures into the project exception hierarchy. Router fallback and Pipeline alert degradation
catch only those expected failures; an unexpected programming error propagates and fails tests.

## D-011: runtime versus planned hardware

The supplied project attachment discusses an RTX 5070 Ti plan. The actual Legion audit showed an
RTX 4060 Laptop GPU with 8188 MiB driver-visible memory, but no PyTorch installation. Reports use
the observed runtime and do not treat driver visibility as CUDA execution proof.

## D-012: Git publication

The user explicitly authorized publication to `main`. Initial commit
`63f999e3ccd612c4a48e46ca729f4821e873516b` was pushed after secret/large-file scope checks.
Weights, datasets, artifacts, checkpoints, uploads, videos, and environment files remain ignored.

## D-013: official-model evidence boundary

Nano and Small use separately MD5-validated official COCO weights. The same public image is used
for real RTX 4060 smoke inference. Latency, memory, and detections are labeled single-image smoke
evidence, not AP, fall accuracy, sustained video FPS, or deployment performance. The upstream
`_kp_active_mask` partial-load warning is retained in logs.

## D-014: grouped fall validation

GMDCSA-24 is split only by subject: S1-S2 threshold development, S3 threshold validation, and S4
locked test. The deterministic small subset selects two Fall and two ADL videos per subject.
Thresholds must not be selected on S3 or S4. Clip labels support clip-level event metrics but do
not justify detection-delay claims without separate human-confirmed onset timestamps.
