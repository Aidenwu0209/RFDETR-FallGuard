"""Strict YAML configuration and formal-run gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fallguard.exceptions import ConfigurationError, FormalBenchmarkRejectedError
from fallguard.schemas import DetectionMode


class ConfigSection(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RuntimeConfig(ConfigSection):
    profile: Literal["development", "experiment"]
    source_id: str
    session_id: str | None = None
    log_level: str = "INFO"
    log_json: bool = True
    device: str = "auto"


class DetectorConfig(ConfigSection):
    backend: Literal["rfdetr"] = "rfdetr"
    model_variant: Literal["nano", "small"] = "small"
    mode: DetectionMode = DetectionMode.PERSON_ONLY
    weights_path: Path | None = None
    allow_weight_download: bool = False
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    person_class_names: list[str] = Field(min_length=1)
    class_names: dict[int, str]
    class_aliases: dict[str, str] = Field(default_factory=dict)
    posture_groups: dict[str, list[str]] = Field(default_factory=dict)


class TrackingConfig(ConfigSection):
    backend: Literal["bytetrack"] = "bytetrack"
    track_activation_threshold: float = Field(ge=0.0, le=1.0)
    lost_track_buffer: int = Field(gt=0)
    minimum_matching_threshold: float = Field(ge=0.0, le=1.0)
    minimum_consecutive_frames: int = Field(gt=0)
    frame_rate: int = Field(gt=0)


class TemporalConfig(ConfigSection):
    smoothing_window: int = Field(gt=0)
    candidate_on_lying_posture: bool = False
    confirm_on_falling_posture: bool = False
    confirm_on_lying_posture: bool = False
    aspect_ratio_fall_min: float | None = Field(default=None, gt=0.0)
    vertical_speed_frame_height_per_second_min: float | None = None
    suspect_duration_seconds: float | None = Field(default=None, gt=0.0)
    lying_duration_seconds: float | None = Field(default=None, gt=0.0)
    upright_aspect_ratio_max: float | None = Field(default=None, gt=0.0)
    track_timeout_seconds: float | None = Field(default=None, gt=0.0)

    def missing_thresholds(self) -> list[str]:
        return [
            name
            for name in (
                "aspect_ratio_fall_min",
                "vertical_speed_frame_height_per_second_min",
                "suspect_duration_seconds",
                "lying_duration_seconds",
                "upright_aspect_ratio_max",
                "track_timeout_seconds",
            )
            if getattr(self, name) is None
        ]


class EventConfig(ConfigSection):
    merge_gap_seconds: float = Field(ge=0.0)
    cooldown_seconds: float = Field(ge=0.0)
    timeout_seconds: float = Field(gt=0.0)
    ring_buffer_frames: int = Field(gt=0)
    before_offset_seconds: float = Field(ge=0.0)
    after_offset_seconds: float = Field(ge=0.0)


class SemanticConfig(ConfigSection):
    provider: Literal["none", "mock", "openai", "deepseek", "local_qwen"]
    model: str | None
    fallback_providers: list[Literal["mock", "openai", "deepseek", "local_qwen"]]
    allow_fallback: bool
    allow_mock: bool
    allow_cloud_images: bool
    max_images: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0.0)
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    deepseek_thinking: bool = True
    openai_base_url: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    local_model_path: Path | None = None


class AlertConfig(ConfigSection):
    alert_on_semantic_fall: bool = True
    alert_on_semantic_uncertain: bool = False
    high_temporal_risk_can_alert: bool = False


class PrivacyConfig(ConfigSection):
    retain_full_frames: bool = False
    retain_person_crops: bool = True
    hash_algorithm: Literal["sha256"] = "sha256"


class BenchmarkConfig(ConfigSection):
    formal: bool
    warmup_iterations: int = Field(ge=0)
    measured_iterations: int = Field(gt=0)
    include_decode: bool
    include_ui: bool
    include_network: bool


class AppConfig(ConfigSection):
    schema_version: Literal[1]
    runtime: RuntimeConfig
    detector: DetectorConfig
    tracking: TrackingConfig
    temporal: TemporalConfig
    event: EventConfig
    semantic: SemanticConfig
    alert: AlertConfig
    privacy: PrivacyConfig
    benchmark: BenchmarkConfig

    def assert_formal_ready(self) -> None:
        issues: list[str] = []
        if not self.benchmark.formal:
            issues.append("benchmark.formal must be true")
        missing = self.temporal.missing_thresholds()
        if missing:
            issues.append("unvalidated temporal thresholds: " + ", ".join(missing))
        if self.detector.mode != DetectionMode.POSTURE_MULTICLASS:
            issues.append("detector.mode must be posture_multiclass")
        if not self.detector.class_names:
            issues.append("detector.class_names must come from dataset/checkpoint metadata")
        if self.semantic.provider == "mock" or self.semantic.allow_mock:
            issues.append("mock semantic provider is forbidden")
        if self.semantic.allow_fallback:
            issues.append("semantic fallback must be disabled")
        if issues:
            raise FormalBenchmarkRejectedError("; ".join(issues))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigurationError(f"configuration root must be a mapping: {path}")
    return value


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    profile_path: str | Path,
    *,
    base_path: str | Path = "configs/base.yaml",
) -> AppConfig:
    base = _read_yaml(Path(base_path))
    profile = _read_yaml(Path(profile_path))
    try:
        return AppConfig.model_validate(_deep_merge(base, profile))
    except ValidationError as exc:
        raise ConfigurationError(f"invalid configuration: {exc}") from exc
