from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_gmdcsa24_recovery_cv",
    PROJECT_ROOT / "scripts/prepare_gmdcsa24_recovery_cv.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def source_manifest() -> dict[str, object]:
    records = [
        {
            "subject_id": subject,
            "label": label,
            "partition": "original",
            "video_id": f"s{subject}/{label}",
        }
        for subject in range(1, 5)
        for label in ("fall", "adl")
    ]
    return {"dataset": "GMDCSA-24", "audit": {"videos": 8}, "records": records}


@pytest.mark.parametrize("validation_subject", [1, 2, 3])
def test_recovery_fold_is_subject_isolated_and_keeps_subject4_locked(
    validation_subject: int,
) -> None:
    fold = MODULE.build_fold(source_manifest(), validation_subject)
    protocol = fold["protocol"]
    assert protocol["threshold_validation"] == [validation_subject]
    assert protocol["locked_test"] == [4]
    subject_partitions: dict[int, set[str]] = {}
    for record in fold["records"]:
        subject_partitions.setdefault(record["subject_id"], set()).add(record["partition"])
    assert all(len(partitions) == 1 for partitions in subject_partitions.values())
    assert subject_partitions[4] == {"locked_test"}
