# RFDETR-FallGuard Project Specification

## 1. Purpose

Build an auditable cascade for fall monitoring:

```text
RF-DETR candidate detection
  -> class-agnostic person identity tracking
  -> temporal candidate confirmation
  -> before/during/after keyframes
  -> provider-agnostic semantic review
  -> application-owned alert decision and event log
```

The engineering deliverable must run a deterministic mock vertical slice without real
weights, private uploads, paid APIs, manual labeling, or long training. Mock evidence is not
scientific or deployment evidence.

## 2. Research alignment

The graduation-project attachment specifies RF-DETR-Small as the main detector and Nano as
a lightweight comparison; ByteTrack identity association; aspect ratio, center displacement,
confidence history, and duration as temporal evidence; before/during/after semantic context;
and detection-, event-, and deployment-level evaluation. UP-Fall, Le2i, and self-collected
scenes are future experimental inputs, not bundled data.

## 3. Required modes

- `person_only`: an official pretrained-weight smoke mode. Output means only that a person
  was detected; it must never be converted to `falling`.
- `posture_multiclass`: a fine-tuned-checkpoint mode whose classes and aliases come from
  dataset metadata or explicit configuration.
- Development profile: contains visibly provisional thresholds for deterministic tests.
- Experiment profile: leaves unvalidated thresholds null and refuses formal benchmarks.

## 4. Data and identity rules

- `frame_id` starts at zero.
- `timestamp_seconds` is monotonic elapsed video time. UTC alert time is separate.
- `bbox_xyxy` is pixel-space `[x1, y1, x2, y2]`, with positive area.
- Frames carry width and height metadata.
- Track identity is scoped by `session_id` and `source_id`.
- ByteTrack association is class-agnostic; per-frame posture remains observation history.
- Persisted business schemas contain image references/hashes, not NumPy arrays or Base64.

## 5. Semantic and alert boundaries

Semantic providers return `fall`, `not_fall`, or `uncertain` plus confidence, reason,
attempt-to-stand, risk, provider/model/input mode, latency, schema validity, and provider
success. A provider may recommend an alert but does not own the final decision. The
`AlertManager` combines semantic and temporal evidence.

Providers are routed exclusively through `SemanticReviewRouter`. Real cloud calls are opt-in;
image uploads require separate user consent. Local health checks perform configuration checks
only. Formal experiments disable fallback. Unit tests never call a cloud API.

## 6. Temporal and event rules

Image y increases downward. Aspect ratio is `width / height`. Displacement is recorded in
pixels and normalized by frame dimensions and person height. Speed uses elapsed seconds and
rejects non-increasing timestamps. Every state transition records a reason.

One continuing fall produces one event. Merge gap, cooldown, timeout, and frame-ring capacity
are configured. Keyframe metadata records whether a frame is before, during, or after, its
selection reason, and full-frame/person-crop references according to privacy policy.

## 7. Evaluation boundary

Detection AP delegates to the official RF-DETR/COCO evaluation path. Event evaluation defines
one-to-one matching before computing counts. Specificity is null without defined negative
windows. False alarms per hour is null without monitored duration. Formal thresholds must be
validation-derived.

Deployment reports sustained loop FPS; stage and end-to-end latency; semantic latency/token
usage when returned; CPU memory; and CUDA allocated/reserved peaks when available. Decode,
UI, and network inclusion are explicit. Mock components are rejected by formal benchmarks.

## 8. Acceptance boundary

Default tests require no real data, GPU, weights, or API. Required external assets are recorded
as `BLOCKED_EXTERNAL`; implemented but unexecuted integrations are
`CODE_COMPLETE_UNVALIDATED`. No final metric is invented.
