from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_false_negatives",
    PROJECT_ROOT / "scripts/diagnose_false_negatives.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("detections", "tracks", "states", "events", "expected"),
    [
        (0, 0, [], 0, "detector_no_production_detection"),
        (2, 0, [], 0, "tracking_no_output"),
        (2, 2, [], 0, "temporal_never_suspected"),
        (2, 2, ["suspected", "resolved"], 0, "temporal_candidate_not_confirmed"),
        (2, 2, ["suspected", "falling"], 0, "event_not_created"),
        (2, 2, ["suspected", "falling"], 1, "event_reproduced_during_diagnostic"),
    ],
)
def test_stage_attribution(
    detections: int,
    tracks: int,
    states: list[str],
    events: int,
    expected: str,
) -> None:
    assert (
        MODULE.stage_attribution(
            production_detections=detections,
            tracked_detections=tracks,
            transition_states=states,
            event_count=events,
        )
        == expected
    )
