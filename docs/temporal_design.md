# Temporal Design

## Coordinates and units

- Image x increases rightward and image y increases downward.
- A positive `center_dy_pixels` is downward motion.
- Aspect ratio is always `bbox_width / bbox_height`.
- Displacement is stored in pixels and normalized by frame width, frame height, and previous
  person height.
- Speed is displacement divided by `timestamp_seconds` difference. It is never implicitly
  "pixels per frame".
- Non-increasing timestamps raise `NonMonotonicTimestampError`. Video reading may use
  `frame_id / FPS` only when the container timestamp is unusable and valid FPS exists.

For consecutive observations `p` and `c`:

```text
dt = c.timestamp_seconds - p.timestamp_seconds
dy_pixels = c.center_y - p.center_y
dy_frame_height = dy_pixels / c.frame_height
vertical_speed_frame_height_per_second = dy_frame_height / dt
aspect_ratio_width_over_height = c.width / c.height
```

## Smoothing and missing observations

The extractor stores at most `smoothing_window` recent feature records for each scoped track and
uses a simple arithmetic mean for geometric signals. Raw posture class and confidence remain the
current observation; a majority-vote class is not silently invented.

Short occlusion is handled by ByteTrack's `lost_track_buffer`. The local `TrackManager` retains a
bounded observation history. If a candidate track remains missing longer than
`track_timeout_seconds`, the state machine emits a `RESOLVED` transition whose reason contains
the observed timeout. A new source/session must reset ByteTrack.

## State transitions

```text
UPRIGHT -> SUSPECTED
  downward-speed or configured falling class
  (ratio and configured lying class are support evidence, not event initiators)

SUSPECTED -> FALLING
  evidence persists for suspect_duration_seconds

SUSPECTED -> UPRIGHT
  upright class plus upright aspect ratio and no fall signal

FALLING -> LYING
  configured lying class or falling duration reaches lying_duration_seconds

LYING -> RECOVERING
  upright class plus upright aspect ratio

RECOVERING -> RESOLVED
  upright evidence persists for suspect_duration_seconds

RECOVERING -> FALLING
  fall evidence returns
```

Every actual change produces `TransitionRecord.reason`. Floating-duration comparisons use a
`1e-9` numerical tolerance so an exact configured boundary is not delayed by binary rounding.
`SUSPECTED` is an internal candidate and does not create a `FallEvent`; the event is promoted
only when persistent evidence reaches `FALLING` (or a direct `LYING` confirmation). This keeps
ordinary static lying and a single wide bounding box from becoming completed fall events.

## Profiles

`configs/profiles/development.yaml` values are provisional and exist for unit tests and UI
integration. They must not be reported as validated thresholds. The experiment profile keeps
all thresholds null, so the state machine and formal benchmark fail before execution. Source
code does not fill those null values.
