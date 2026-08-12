"""Auditable per-track fall state machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from fallguard.config import TemporalConfig
from fallguard.exceptions import ConfigurationError
from fallguard.schemas import MotionState, TemporalFeatures, TrackedDetection, TransitionRecord

TrackKey = tuple[str, str, int]


@dataclass
class _TrackState:
    state: MotionState
    entered_seconds: float
    last_seen_seconds: float


class TemporalStateMachine:
    def __init__(self, config: TemporalConfig, posture_groups: dict[str, list[str]]) -> None:
        missing = config.missing_thresholds()
        if missing:
            raise ConfigurationError(
                "temporal state machine needs explicit thresholds: " + ", ".join(missing)
            )
        self.config = config
        self.aspect_ratio_fall_min = cast(float, config.aspect_ratio_fall_min)
        self.vertical_speed_min = cast(float, config.vertical_speed_frame_height_per_second_min)
        self.suspect_duration_seconds = cast(float, config.suspect_duration_seconds)
        self.lying_duration_seconds = cast(float, config.lying_duration_seconds)
        self.upright_aspect_ratio_max = cast(float, config.upright_aspect_ratio_max)
        self.track_timeout_seconds = cast(float, config.track_timeout_seconds)
        self.posture_groups = {
            key: {name.lower() for name in value} for key, value in posture_groups.items()
        }
        self._states: dict[TrackKey, _TrackState] = {}

    def state_for(self, observation: TrackedDetection) -> MotionState:
        key = (observation.source_id, observation.session_id, observation.track_id)
        return self._states.get(
            key,
            _TrackState(
                MotionState.UPRIGHT, observation.timestamp_seconds, observation.timestamp_seconds
            ),
        ).state

    def update(
        self,
        observation: TrackedDetection,
        features: TemporalFeatures,
    ) -> TransitionRecord | None:
        key = (observation.source_id, observation.session_id, observation.track_id)
        current = self._states.setdefault(
            key,
            _TrackState(
                MotionState.UPRIGHT, observation.timestamp_seconds, observation.timestamp_seconds
            ),
        )
        current.last_seen_seconds = observation.timestamp_seconds
        elapsed_in_state = observation.timestamp_seconds - current.entered_seconds
        posture = observation.class_name.lower()
        falling_posture = posture in self.posture_groups.get("fall", set())
        lying_posture = posture in self.posture_groups.get("lying", set())
        upright_posture = posture in self.posture_groups.get("upright", set())
        ratio_signal = features.aspect_ratio_width_over_height >= self.aspect_ratio_fall_min
        speed_signal = features.vertical_speed_frame_height_per_second >= self.vertical_speed_min
        fall_signal = ratio_signal or speed_signal or falling_posture or lying_posture
        candidate_trigger = (
            speed_signal
            or falling_posture
            or (self.config.candidate_on_lying_posture and lying_posture)
        )
        upright_signal = (
            upright_posture
            and features.aspect_ratio_width_over_height <= self.upright_aspect_ratio_max
        )

        next_state = current.state
        reason = ""
        if current.state == MotionState.UPRIGHT:
            if self.config.confirm_on_lying_posture and lying_posture:
                next_state = MotionState.LYING
                reason = "configured lying posture immediately confirmed"
            elif self.config.confirm_on_falling_posture and falling_posture:
                next_state = MotionState.FALLING
                reason = "configured falling posture immediately confirmed"
            elif candidate_trigger:
                next_state = MotionState.SUSPECTED
                reason = self._signal_reason(
                    False,
                    speed_signal,
                    falling_posture,
                    self.config.candidate_on_lying_posture and lying_posture,
                )
        elif current.state == MotionState.SUSPECTED:
            if self.config.confirm_on_lying_posture and lying_posture:
                next_state = MotionState.LYING
                reason = "configured lying posture immediately confirmed"
            elif self.config.confirm_on_falling_posture and falling_posture:
                next_state = MotionState.FALLING
                reason = "configured falling posture immediately confirmed"
            elif upright_signal and not fall_signal:
                next_state = MotionState.UPRIGHT
                reason = "candidate cleared by upright posture and aspect ratio"
            elif elapsed_in_state + 1e-9 >= self.suspect_duration_seconds:
                next_state = MotionState.FALLING
                reason = f"fall evidence persisted for {elapsed_in_state:.3f}s"
        elif current.state == MotionState.FALLING:
            if lying_posture or elapsed_in_state + 1e-9 >= self.lying_duration_seconds:
                next_state = MotionState.LYING
                reason = (
                    "configured lying posture observed"
                    if lying_posture
                    else f"falling state persisted for {elapsed_in_state:.3f}s"
                )
        elif current.state == MotionState.LYING and upright_signal:
            next_state = MotionState.RECOVERING
            reason = "upright evidence observed after lying state"
        elif current.state == MotionState.RECOVERING:
            if fall_signal:
                next_state = MotionState.FALLING
                reason = "fall evidence returned during recovery"
            elif elapsed_in_state + 1e-9 >= self.suspect_duration_seconds and upright_signal:
                next_state = MotionState.RESOLVED
                reason = f"upright recovery persisted for {elapsed_in_state:.3f}s"
        elif current.state == MotionState.RESOLVED and fall_signal:
            next_state = MotionState.SUSPECTED
            reason = "new fall evidence after resolved state"

        if next_state == current.state:
            return None
        transition = TransitionRecord(
            track_id=observation.track_id,
            source_id=observation.source_id,
            session_id=observation.session_id,
            frame_id=observation.frame_id,
            timestamp_seconds=observation.timestamp_seconds,
            previous_state=current.state,
            next_state=next_state,
            reason=reason,
        )
        current.state = next_state
        current.entered_seconds = observation.timestamp_seconds
        return transition

    def expire(
        self,
        *,
        source_id: str,
        session_id: str,
        track_id: int,
        frame_id: int,
        now_seconds: float,
    ) -> TransitionRecord | None:
        key = (source_id, session_id, track_id)
        current = self._states.get(key)
        if current is None or current.state in {MotionState.UPRIGHT, MotionState.RESOLVED}:
            return None
        if now_seconds - current.last_seen_seconds <= self.track_timeout_seconds:
            return None
        transition = TransitionRecord(
            track_id=track_id,
            source_id=source_id,
            session_id=session_id,
            frame_id=frame_id,
            timestamp_seconds=now_seconds,
            previous_state=current.state,
            next_state=MotionState.RESOLVED,
            reason=f"track timed out after {now_seconds - current.last_seen_seconds:.3f}s",
        )
        current.state = MotionState.RESOLVED
        current.entered_seconds = now_seconds
        return transition

    @staticmethod
    def _signal_reason(ratio: bool, speed: bool, falling: bool, lying: bool) -> str:
        signals = []
        if ratio:
            signals.append("width/height ratio threshold")
        if speed:
            signals.append("downward speed threshold")
        if falling:
            signals.append("configured falling posture")
        if lying:
            signals.append("configured lying posture")
        return "candidate triggered by " + ", ".join(signals)
