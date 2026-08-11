# Evaluation Protocol

## Detection level

Detection AP is delegated to the pinned official RF-DETR evaluation method, which uses its COCO
evaluation path. This project does not maintain a second AP implementation. Before comparison,
record dataset revision, split, class metadata, input resolution, batch size, precision, device,
and checkpoint hash.

## Event level

Ground truth and predictions use half-open elapsed-time intervals scoped by source and session.
Matching candidates must satisfy configured temporal-IoU and/or start-time tolerance. Candidate
pairs are sorted by temporal IoU and then smaller start error; greedy assignment is one-to-one.
After one prediction claims a ground-truth event, additional predictions for that event are false
positives. `same_track` rejects different known track IDs; `ignore_track` must be an explicit
protocol choice for cross-tracker comparison.

```text
temporal IoU = intersection duration / union duration
Precision = TP events / predicted events
Recall = TP events / ground-truth events
F1 = harmonic mean of event precision and recall
Miss rate = FN events / ground-truth events
Detection delay = predicted start - ground-truth start
False alarms per hour = false-positive events / monitored hours
```

Specificity is `null` unless the evaluator receives explicit negative time windows. False alarms
per hour is `null` unless monitored duration is supplied. A formal run must not select matching
or temporal thresholds on the test set.

## Deployment level

Report the detection/tracking/temporal/keyframe/semantic/alert stage distributions and the main
loop sustained FPS. P50/P95 use recorded per-iteration latency. GPU timing synchronizes CUDA
after warm-up and each measured iteration. CPU RSS and CUDA peak allocated/reserved memory are
reported when available. The result records whether decoding, UI, and network time are included.

Formal benchmarking rejects null experimental thresholds, `person_only`, missing class metadata,
fallback, and mock components. Mock timing is labeled `MOCK_ENGINEERING_ONLY`.

## Grouped threshold freeze

The small cascade protocol is fixed before execution:

1. Four provisional configs are generated before grouped results are inspected. If that stage
   fails the declared gate, a bounded 0.60/0.70/0.75/0.80 precision grid may be declared as a
   separate S1-S2-only development stage; S3 remains unread.
2. Every Nano/Small candidate runs on the identical subject-isolated S1-S2 video list.
3. `select_thresholds.py` recomputes metrics from clip rows, rejects mixed subjects, mismatched
   manifests, different video lists, duplicate candidates, or config/report parameter drift.
4. Candidate eligibility uses a declared minimum recall and maximum false-positive clip count.
   Ranking then uses fewer false positives, higher F1/recall/specificity, and Nano only as an exact
   metric tie-break, then lower detector confidence within the same preferred model to preserve
   recall margin. A failed gate is reported; it is never relaxed silently.
5. The winner's detector, ByteTrack, temporal, and event parameters are frozen in a complete config.
6. Reports record both the Git revision and a SHA-256 fingerprint over the executable pipeline
   sources. S3 is evaluated once with the same pipeline fingerprint, checkpoint hash, and frozen
   parameters. Confirmation rejects any overlap with S1-S2 or any retuning.
7. S4 requires the matching confirmation artifact, an explicit unlock flag, and full-subject
   evaluation. S4 never participates in model or threshold selection.

This produces confirmed internal thresholds, not a generalization or deployment claim. GMDCSA-24
clip labels still cannot support detection delay without human-confirmed onset timestamps.

Warm-up frames are consumed before measurement and all stage collectors are reset at the
measurement boundary. `event_trigger_video_time` is elapsed video time from event start to alert
decision; it is deliberately not mislabeled as wall-clock compute latency. Per-stage throughput
is derived from measured mean stage latency and is distinct from sustained end-to-end loop FPS.
