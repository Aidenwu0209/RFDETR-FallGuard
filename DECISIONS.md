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

The origin is empty and all project files are currently untracked. No local commit or remote push
is made because the controlling execution instruction requires explicit user confirmation before
push.
