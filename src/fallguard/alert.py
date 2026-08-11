"""Final alert ownership; semantic providers only contribute evidence."""

from __future__ import annotations

from fallguard.config import AlertConfig
from fallguard.schemas import AlertDecision, FallEvent, MotionState, SemanticAssessment


class AlertManager:
    def __init__(self, config: AlertConfig) -> None:
        self.config = config

    def decide(
        self,
        event: FallEvent,
        temporal_state: MotionState,
        semantic: SemanticAssessment | None,
    ) -> AlertDecision:
        reasons: list[str] = []
        should_alert = False
        if semantic is None or not semantic.provider_success or not semantic.schema_valid:
            reasons.append("no successful schema-valid semantic assessment")
            if self.config.high_temporal_risk_can_alert and temporal_state in {
                MotionState.FALLING,
                MotionState.LYING,
            }:
                should_alert = True
                reasons.append("application policy allows high temporal risk without semantics")
        elif semantic.decision == "fall" and self.config.alert_on_semantic_fall:
            should_alert = True
            reasons.append("schema-valid semantic decision is fall")
        elif semantic.decision == "uncertain" and self.config.alert_on_semantic_uncertain:
            should_alert = True
            reasons.append("application policy alerts on semantic uncertainty")
        else:
            reasons.append(f"application policy does not alert for semantic {semantic.decision}")
        if semantic is not None and semantic.model_recommends_alert is not None:
            reasons.append(
                "provider recommendation recorded as evidence="
                + str(semantic.model_recommends_alert).lower()
            )
        return AlertDecision(
            event_id=event.event_id,
            should_alert=should_alert,
            reasons=reasons,
            semantic_assessment=semantic,
            temporal_state=temporal_state,
        )
