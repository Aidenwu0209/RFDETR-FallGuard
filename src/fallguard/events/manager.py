"""One-event-per-continuous-episode lifecycle manager."""

from __future__ import annotations

from fallguard.config import EventConfig
from fallguard.schemas import (
    EventStatus,
    FallEvent,
    KeyframeSelection,
    MotionState,
    TransitionRecord,
)

TrackKey = tuple[str, str, int]


class EventManager:
    def __init__(self, config: EventConfig) -> None:
        self.config = config
        self._events: dict[str, FallEvent] = {}
        self._active_by_track: dict[TrackKey, str] = {}
        self._last_by_track: dict[TrackKey, str] = {}

    def _replace(self, event: FallEvent, **updates: object) -> FallEvent:
        """Apply coupled lifecycle fields as one validated state transition."""
        payload = event.model_dump(mode="python")
        payload.update(updates)
        replacement = FallEvent.model_validate(payload)
        self._events[event.event_id] = replacement
        return replacement

    def on_transition(self, transition: TransitionRecord) -> FallEvent | None:
        key = (transition.source_id, transition.session_id, transition.track_id)
        active_id = self._active_by_track.get(key)
        if active_id is not None:
            event = self._events[active_id]
            event.transition_reasons.append(transition.reason)
            if transition.next_state == MotionState.LYING:
                event.metadata["lying_started_at_seconds"] = transition.timestamp_seconds
            if transition.next_state == MotionState.RESOLVED:
                event = self._replace(
                    event,
                    end_frame=transition.frame_id,
                    end_time=transition.timestamp_seconds,
                    status=EventStatus.RESOLVED,
                )
                del self._active_by_track[key]
                self._last_by_track[key] = event.event_id
            return event

        if transition.next_state != MotionState.SUSPECTED:
            return None

        previous_id = self._last_by_track.get(key)
        if previous_id is not None:
            previous = self._events[previous_id]
            assert previous.end_time is not None
            gap = transition.timestamp_seconds - previous.end_time
            if gap <= self.config.merge_gap_seconds:
                previous = self._replace(
                    previous,
                    end_frame=None,
                    end_time=None,
                    status=EventStatus.ACTIVE,
                    transition_reasons=[
                        *previous.transition_reasons,
                        "merged: " + transition.reason,
                    ],
                )
                self._active_by_track[key] = previous.event_id
                return previous
            if gap <= self.config.cooldown_seconds:
                previous.metadata["cooldown_suppressed_candidate_at"] = transition.timestamp_seconds
                return previous

        event = FallEvent(
            track_id=transition.track_id,
            source_id=transition.source_id,
            session_id=transition.session_id,
            start_frame=transition.frame_id,
            start_time=transition.timestamp_seconds,
            transition_reasons=[transition.reason],
        )
        self._events[event.event_id] = event
        self._active_by_track[key] = event.event_id
        return event

    def tick(self, now_seconds: float, frame_id: int) -> list[FallEvent]:
        timed_out: list[FallEvent] = []
        for key, event_id in list(self._active_by_track.items()):
            event = self._events[event_id]
            if now_seconds - event.start_time > self.config.timeout_seconds:
                event = self._replace(
                    event,
                    end_time=now_seconds,
                    end_frame=frame_id,
                    status=EventStatus.TIMED_OUT,
                    transition_reasons=[
                        *event.transition_reasons,
                        "event timeout reached",
                    ],
                )
                del self._active_by_track[key]
                self._last_by_track[key] = event_id
                timed_out.append(event)
        return timed_out

    def add_keyframes(self, event_id: str, selections: list[KeyframeSelection]) -> FallEvent:
        event = self._events[event_id]
        existing = {(item.role, item.frame_id) for item in event.keyframes}
        event.keyframes.extend(
            item for item in selections if (item.role, item.frame_id) not in existing
        )
        return event

    def get(self, event_id: str) -> FallEvent:
        return self._events[event_id]

    def all_events(self) -> list[FallEvent]:
        return list(self._events.values())

    def active_events(self) -> list[FallEvent]:
        return [self._events[event_id] for event_id in self._active_by_track.values()]
