"""Application object construction without hidden provider or model fallback."""

from __future__ import annotations

from pathlib import Path

from fallguard.alert import AlertManager
from fallguard.config import AppConfig
from fallguard.detection.rfdetr_adapter import RFDETRDetector
from fallguard.events.keyframes import BoundedFrameBuffer
from fallguard.events.manager import EventManager
from fallguard.exceptions import ConfigurationError
from fallguard.pipeline import AblationMode, FallGuardPipeline
from fallguard.semantic.base import SemanticProvider
from fallguard.semantic.providers import (
    DeepSeekProvider,
    LocalQwenProvider,
    MockProvider,
    OpenAIProvider,
)
from fallguard.semantic.router import SemanticReviewRouter
from fallguard.temporal.features import TemporalFeatureExtractor
from fallguard.temporal.state_machine import TemporalStateMachine
from fallguard.tracking.bytetrack_adapter import ByteTrackAdapter
from fallguard.tracking.manager import TrackManager


def build_providers(config: AppConfig) -> dict[str, SemanticProvider]:
    names = {config.semantic.provider, *config.semantic.fallback_providers}
    providers: dict[str, SemanticProvider] = {}
    if "mock" in names:
        providers["mock"] = MockProvider()
    if "openai" in names:
        if not config.semantic.model:
            raise ConfigurationError("semantic.model is required for OpenAI")
        providers["openai"] = OpenAIProvider(
            config.semantic.model,
            base_url=config.semantic.openai_base_url,
            timeout_seconds=config.semantic.timeout_seconds,
        )
    if "deepseek" in names:
        if not config.semantic.model:
            raise ConfigurationError("semantic.model is required for DeepSeek")
        providers["deepseek"] = DeepSeekProvider(
            config.semantic.model,
            base_url=config.semantic.deepseek_base_url,
            timeout_seconds=config.semantic.timeout_seconds,
        )
    if "local_qwen" in names:
        if config.semantic.local_model_path is None:
            raise ConfigurationError("semantic.local_model_path is required for Local Qwen")
        providers["local_qwen"] = LocalQwenProvider(
            config.semantic.local_model_path,
            model_name=config.semantic.model,
        )
    return providers


def build_pipeline(
    config: AppConfig,
    *,
    with_real_frontend: bool,
    keyframe_output_dir: str | Path = "keyframes",
    cloud_image_consent: bool = False,
    ablation_mode: AblationMode = "full",
) -> FallGuardPipeline:
    detector = (
        RFDETRDetector(config.detector, device=config.runtime.device)
        if with_real_frontend
        else None
    )
    tracker = ByteTrackAdapter(config.tracking) if with_real_frontend else None
    return FallGuardPipeline(
        detector=detector,
        tracker=tracker,
        track_manager=TrackManager(),
        feature_extractor=TemporalFeatureExtractor(
            smoothing_window=config.temporal.smoothing_window
        ),
        state_machine=TemporalStateMachine(config.temporal, config.detector.posture_groups),
        event_manager=EventManager(config.event),
        semantic_router=SemanticReviewRouter(config.semantic, build_providers(config)),
        alert_manager=AlertManager(config.alert),
        frame_buffer=BoundedFrameBuffer(config.event, config.privacy),
        keyframe_output_dir=keyframe_output_dir,
        cloud_image_consent=cloud_image_consent,
        ablation_mode=ablation_mode,
    )
