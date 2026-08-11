"""Track history keyed by source, session, and tracker identity."""

from __future__ import annotations

from fallguard.schemas import Track, TrackedDetection

TrackKey = tuple[str, str, int]


class TrackManager:
    def __init__(self, *, max_history: int = 120) -> None:
        if max_history <= 1:
            raise ValueError("max_history must be greater than one")
        self.max_history = max_history
        self._tracks: dict[TrackKey, Track] = {}

    def update(self, observations: list[TrackedDetection]) -> list[Track]:
        updated: list[Track] = []
        for item in observations:
            key = (item.source_id, item.session_id, item.track_id)
            track = self._tracks.get(key)
            history = [] if track is None else list(track.observations)
            if history and item.timestamp_seconds <= history[-1].timestamp_seconds:
                raise ValueError("track observation timestamps must strictly increase")
            history.append(item)
            history = history[-self.max_history :]
            track = Track(
                track_id=item.track_id,
                source_id=item.source_id,
                session_id=item.session_id,
                observations=history,
                last_seen_seconds=item.timestamp_seconds,
                active=True,
            )
            self._tracks[key] = track
            updated.append(track)
        return updated

    def expire(self, now_seconds: float, timeout_seconds: float) -> list[Track]:
        expired: list[Track] = []
        for key, track in list(self._tracks.items()):
            if track.active and now_seconds - track.last_seen_seconds > timeout_seconds:
                inactive = track.model_copy(update={"active": False})
                self._tracks[key] = inactive
                expired.append(inactive)
        return expired

    def get(self, source_id: str, session_id: str, track_id: int) -> Track | None:
        return self._tracks.get((source_id, session_id, track_id))

    def active(self) -> list[Track]:
        return [track for track in self._tracks.values() if track.active]
