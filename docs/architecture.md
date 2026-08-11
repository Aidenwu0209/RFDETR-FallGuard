# Architecture

## Runtime flow

```text
VideoReader / image
  -> RFDETRDetector (official rfdetr adapter)
  -> Detection schema
  -> ByteTrackAdapter (class_id is deliberately replaced by 0 for association)
  -> TrackedDetection (original posture class restored)
  -> TrackManager history
  -> TemporalFeatureExtractor
  -> TemporalStateMachine + transition reason
  -> EventManager + bounded frame buffer
  -> SemanticReviewRouter
  -> SemanticAssessment
  -> AlertManager
  -> event and alert JSONL
```

The business pipeline never imports a semantic provider directly. It receives only a
`SemanticReviewRouter`. Likewise, the detector network and ByteTrack algorithm remain in their
official packages; local code owns normalization, lifecycle, and evidence boundaries.

## Identity and posture

A fall changes appearance and often changes the fine-tuned posture class. Using that class as
an association constraint can split one person into several tracks. The ByteTrack adapter sends
all detections as class `0` to association, while `TrackedDetection` keeps the detector's original
class name and confidence. A track key is:

```text
(source_id, session_id, track_id)
```

This prevents a tracker counter that restarts at `1` for a new video from colliding with an old
event. Unless an explicit reproducibility session is configured, every CLI run appends a random
suffix to the input stem so repeated processing of the same filename also remains isolated.

## Real and mock surfaces

There is no production `MockDetector`. The deterministic vertical slice feeds already tracked
synthetic observations through `process_tracked()`. `MockProvider.component_kind` is `mock`, and
formal benchmarking rejects any component with that kind. Mock artifacts are labeled `MOCK`.

## Dependency boundaries

- Core: Pydantic, YAML, NumPy, OpenCV, Pillow, HTTPX, psutil.
- RF-DETR: exact `rfdetr==1.9.1`, installed only with `.[rfdetr]`.
- Tracking: exact `supervision==0.30.0`, installed only with `.[tracking]`.
- UI: exact `gradio==6.22.0`, installed only with `.[ui]`.
- Cloud: exact `openai==2.53.0`, installed only with `.[cloud]`.
- Local VLM: exact Transformers/PEFT versions plus explicit optional GPU dependencies.

`pyproject.toml` is the only hand-maintained dependency source.
