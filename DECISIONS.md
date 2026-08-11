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
RTX 4060 Laptop GPU with 8188 MiB memory. The project environment now verifies
`torch==2.13.0+cu130`, CUDA availability, and real model execution on that GPU. Reports use this
observed runtime rather than the planned hardware and never treat driver visibility alone as CUDA
execution proof.

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

## D-015: posture-data audit gate

Fallen Person is accepted only through a local COCO audit that verifies all image/annotation
hashes, dimensions, boxes, the exact four-class schema, and cross-split duplicates. RF-DETR class
indices are derived by sorting COCO category IDs and must remain contiguous from zero. The source
export does not contain person/video group identifiers, so its original validation split cannot
substitute for the subject-isolated GMDCSA-24 cascade protocol.

The supplied archive has SHA-256
`352d0ba25c5b82307749aaaba3f8da34874892fe45e5c9a1197d6e55c7566892`. Its duplicate unused
`fallen` category is normalized deterministically, while source files remain immutable. Only
Roboflow rounding overflow up to 0.01 pixel is clipped. The audit found 21,575 cross-split dHash
near-duplicate pairs, so its detector test metrics remain engineering-only.

## D-016: fine-tuned checkpoint loading and smoke boundary

Locally trained RF-DETR checkpoints are loaded with the upstream `from_checkpoint` path, not as
official starter weights. The adapter pins the four-class head and rejects a selected Nano/Small
architecture mismatch. Two-epoch real-data CUDA runs produced independently reloadable Nano and
Small checkpoints with SHA-256 `3d190d2450a1dffcd2340e198f66d14475c9f1f1f0cd3fcc0246d7edba77cc6f`
and `7c843e2570a077317c95e3458d50a402f36f6ac8ebcd8bffe7fa8da28f898e71`.
RF-DETR 1.9.1 omits query-layout fields from these best-total checkpoint args, so the generic
reload warning is retained; both were separately verified to carry the expected default 300 x 13
query rows.

## D-017: threshold freeze and locked-test gate

Four stage-1 configurations and a bounded four-confidence stage-2 precision grid are declared
before grouped evaluation. Nano and Small candidates must use the same S1-S2 video IDs and
manifest. Selection recomputes clip metrics and rejects subject leakage, mixed partitions,
parameter/config drift, duplicate candidates, implementation-fingerprint drift, and silently
relaxed quality gates. Nano confidence/tracker activation 0.75 was frozen after perfect S1-S2
clip metrics, then failed its one-time S3 gate with 1 TP, 0 FP, 1 FN, and 2 TN. No formal threshold
is confirmed. S4 remains locked and cannot influence checkpoint or threshold selection.

## D-018: temporal candidate and event semantics

Static lying or a wide aspect ratio cannot independently initiate a fall candidate; initiation
requires downward motion or the configured fall posture class. Lying and aspect ratio remain
supporting evidence after initiation. A SUSPECTED transition is cached as a pending candidate and
does not create a formal event until the track reaches FALLING or LYING. This makes the declared
duration thresholds observable in event-presence metrics and prevents transient suspects from
being counted as completed falls.

## D-019: fresh Fall29 subject protocol

The failed historical GMDCSA S3 result remains immutable and is not reused for tuning. A new
cycle uses the public Figshare Video-Based Fall Detection Dataset with 29 subjects. The archive
MD5 is `c784167d08f2fa94e3afd36cec758e1f`; the raw archive is preserved. Preparation excludes two
auxiliary timelapse videos and one exact duplicate from the derived manifest, leaving 2,013
unique subject videos. Development subjects, validation subjects, locked-test subjects, the
six-value detector/tracker confidence grid, and pass gates are declared in
`configs/validation/figshare_fall29_v1.yaml` before model results are inspected.

The Small checkpoint at confidence/tracker activation 0.40 is selected from development only.
Its frozen validation gate requires recall at least 0.90 and zero false-positive clips. The
locked test is usable only after a matching confirmation and explicit `--unlock-locked-test
--all-videos`; it cannot influence model or threshold selection.

The one-time full locked test used all 374 declared clips and produced 139 TP, 19 FP, 64 FN, and
152 TN (precision 0.8797, recall 0.6847, specificity 0.8889, F1 0.7701). The threshold remains the
confirmed internal profile; the lower unseen-subject recall is reported as a limitation and does
not trigger post-test retuning.

## D-020: audit-only confirmation migration

The first locked-test invocation stopped before processing a video because JSON reload changes
integer class-name mapping keys to strings. The equality check now compares canonical JSON. An
existing confirmation can cross that control-plane-only change only through an explicit migration
artifact proving the original confirmation hash, both full implementation fingerprints, equal
runtime-core hashes, unchanged model/threshold parameters, and no reuse of the validation
partition. The Fall29 migration records runtime-core SHA-256
`31110189fbd6f3818e75e8c20bae13a19c09cd8a87fca7a9c330467c4a2625bc` on both sides. It does not
authorize retuning or change the confirmed pipeline behavior.
