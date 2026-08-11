"""Strict serializable schemas shared by every pipeline stage."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DetectionMode(str, Enum):
    PERSON_ONLY = "person_only"
    POSTURE_MULTICLASS = "posture_multiclass"


class FrameMetadata(StrictModel):
    frame_id: Annotated[int, Field(ge=0)]
    timestamp_seconds: NonNegativeFloat
    frame_width: PositiveInt
    frame_height: PositiveInt
    source_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class Detection(StrictModel):
    detection_id: str = Field(default_factory=lambda: str(uuid4()))
    frame_id: Annotated[int, Field(ge=0)]
    timestamp_seconds: NonNegativeFloat
    bbox_xyxy: tuple[float, float, float, float]
    frame_width: PositiveInt
    frame_height: PositiveInt
    class_id: Annotated[int, Field(ge=0)]
    class_name: str = Field(min_length=1)
    confidence: UnitFloat
    source_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)

    @field_validator("bbox_xyxy")
    @classmethod
    def validate_bbox(cls, value: tuple[float, float, float, float]) -> tuple[float, ...]:
        x1, y1, x2, y2 = value
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must satisfy x2 > x1 and y2 > y1")
        if x1 < 0 or y1 < 0:
            raise ValueError("bbox_xyxy pixel coordinates must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> Detection:
        _, _, x2, y2 = self.bbox_xyxy
        if x2 > self.frame_width or y2 > self.frame_height:
            raise ValueError("bbox_xyxy must lie within the declared frame dimensions")
        return self

    @property
    def width(self) -> float:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> float:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class TrackedDetection(Detection):
    track_id: Annotated[int, Field(ge=0)]


class Track(StrictModel):
    track_id: Annotated[int, Field(ge=0)]
    source_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    observations: list[TrackedDetection] = Field(default_factory=list)
    last_seen_seconds: NonNegativeFloat
    active: bool = True

    @model_validator(mode="after")
    def validate_scope(self) -> Track:
        for observation in self.observations:
            if observation.track_id != self.track_id:
                raise ValueError("track observation has a different track_id")
            if (observation.source_id, observation.session_id) != (
                self.source_id,
                self.session_id,
            ):
                raise ValueError("track observation has a different source/session scope")
        if self.observations and self.last_seen_seconds != self.observations[-1].timestamp_seconds:
            raise ValueError("last_seen_seconds must match the latest observation")
        return self


class MotionState(str, Enum):
    UPRIGHT = "upright"
    SUSPECTED = "suspected"
    FALLING = "falling"
    LYING = "lying"
    RECOVERING = "recovering"
    RESOLVED = "resolved"


class TemporalFeatures(StrictModel):
    frame_id: Annotated[int, Field(ge=0)]
    timestamp_seconds: NonNegativeFloat
    aspect_ratio_width_over_height: Annotated[float, Field(gt=0.0)]
    center_dx_pixels: float
    center_dy_pixels: float
    center_dx_frame_width: float
    center_dy_frame_height: float
    center_dy_person_height: float
    vertical_speed_pixels_per_second: float
    vertical_speed_frame_height_per_second: float
    posture_class_name: str
    posture_confidence: UnitFloat
    elapsed_seconds: Annotated[float, Field(gt=0.0)]


class TransitionRecord(StrictModel):
    track_id: Annotated[int, Field(ge=0)]
    source_id: str
    session_id: str
    frame_id: Annotated[int, Field(ge=0)]
    timestamp_seconds: NonNegativeFloat
    previous_state: MotionState
    next_state: MotionState
    reason: str = Field(min_length=1)


class EventStatus(str, Enum):
    ACTIVE = "active"
    COOLDOWN = "cooldown"
    RESOLVED = "resolved"
    TIMED_OUT = "timed_out"


class ImageRef(StrictModel):
    path: Path
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    width: PositiveInt
    height: PositiveInt
    kind: Literal["full_frame", "person_crop"]
    retained: bool = True


class KeyframeRole(str, Enum):
    BEFORE = "before"
    DURING = "during"
    AFTER = "after"


class KeyframeSelection(StrictModel):
    frame_id: Annotated[int, Field(ge=0)]
    timestamp_seconds: NonNegativeFloat
    role: KeyframeRole
    reason: str = Field(min_length=1)
    score: float | None = None
    features: dict[str, float | str | bool | None] = Field(default_factory=dict)
    full_frame: ImageRef | None = None
    person_crop: ImageRef | None = None


class FallEvent(StrictModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    track_id: Annotated[int, Field(ge=0)]
    source_id: str
    session_id: str
    start_frame: Annotated[int, Field(ge=0)]
    start_time: NonNegativeFloat
    end_frame: Annotated[int, Field(ge=0)] | None = None
    end_time: NonNegativeFloat | None = None
    status: EventStatus = EventStatus.ACTIVE
    transition_reasons: list[str] = Field(default_factory=list)
    keyframes: list[KeyframeSelection] = Field(default_factory=list)
    metadata: dict[str, str | float | int | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_end(self) -> FallEvent:
        if (self.end_frame is None) != (self.end_time is None):
            raise ValueError("end_frame and end_time must both be present or both be None")
        if self.end_frame is not None and self.end_frame < self.start_frame:
            raise ValueError("end_frame cannot precede start_frame")
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("end_time cannot precede start_time")
        return self


SemanticDecision = Literal["fall", "not_fall", "uncertain"]
RiskLevel = Literal["low", "medium", "high", "unknown"]


class ProviderCapabilities(StrictModel):
    supports_images: bool
    supports_structured_output: bool
    max_images: Annotated[int, Field(ge=0)]
    input_mode: Literal["images", "text", "images_and_text"]


class SemanticReviewRequest(StrictModel):
    event: FallEvent
    text_context: str = Field(min_length=1)
    image_refs: list[ImageRef] = Field(default_factory=list)
    cloud_image_consent: bool = False


class SemanticAssessment(StrictModel):
    decision: SemanticDecision
    confidence: UnitFloat | None = None
    reason: str = Field(min_length=1)
    attempt_to_stand: bool | None = None
    risk_level: RiskLevel = "unknown"
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    input_mode: Literal["images", "text", "images_and_text"]
    latency_ms: NonNegativeFloat
    schema_valid: bool
    provider_success: bool
    model_recommends_alert: bool | None = None
    fallback_reason: str | None = None
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    ground_truth_verified: bool = False


class AlertDecision(StrictModel):
    event_id: str
    should_alert: bool
    decision_time_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reasons: list[str] = Field(min_length=1)
    semantic_assessment: SemanticAssessment | None = None
    temporal_state: MotionState
    delivered: bool = False

    @field_validator("decision_time_utc")
    @classmethod
    def utc_only(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decision_time_utc must be timezone-aware")
        return value.astimezone(timezone.utc)


class GroundTruthEvent(StrictModel):
    event_id: str
    source_id: str
    session_id: str
    start_time: NonNegativeFloat
    end_time: NonNegativeFloat
    track_id: int | None = None

    @model_validator(mode="after")
    def positive_duration(self) -> GroundTruthEvent:
        if self.end_time <= self.start_time:
            raise ValueError("ground-truth event duration must be positive")
        return self


class PredictedEvent(StrictModel):
    event_id: str
    source_id: str
    session_id: str
    start_time: NonNegativeFloat
    end_time: NonNegativeFloat
    track_id: int | None = None
    confidence: UnitFloat | None = None

    @model_validator(mode="after")
    def positive_duration(self) -> PredictedEvent:
        if self.end_time <= self.start_time:
            raise ValueError("predicted event duration must be positive")
        return self


class SemanticTrainingSample(StrictModel):
    sample_id: str
    source_id: str
    session_id: str
    event_id: str
    person_id: str | None = None
    image_refs: list[ImageRef] = Field(min_length=1)
    text_context: str = Field(min_length=1)
    target: SemanticAssessment
    split_group: str = Field(min_length=1)

    @model_validator(mode="after")
    def target_must_be_verified(self) -> SemanticTrainingSample:
        if not self.target.ground_truth_verified:
            raise ValueError("semantic training targets must be ground-truth verified")
        return self
