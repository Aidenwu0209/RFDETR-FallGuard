from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_figshare_fall29", PROJECT_ROOT / "scripts/prepare_figshare_fall29.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_manifest_preserves_subject_groups_and_excludes_auxiliary_videos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "VideoDataset"
    both_labels = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 20}
    fall_only = {11, 12, 15, 16, 17, 18, 21, 22}
    for subject in range(1, 30):
        for label_dir in ("ADL", "Fall"):
            if label_dir == "ADL" and subject in fall_only:
                continue
            if label_dir == "Fall" and subject not in both_labels | fall_only:
                continue
            for index in range(2):
                path = root / label_dir / f"SBJ_{subject:02d}_LOC1" / f"ACT{index}" / "clip.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{subject}-{label_dir}-{index}".encode())
    auxiliary = root / "Fall" / "timelapse_20x.mp4"
    auxiliary.write_bytes(b"auxiliary")
    duplicate = root / "Fall" / "SBJ_02_LOC1" / "ACT9" / "duplicate.mp4"
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    duplicate.write_bytes(b"2-Fall-0")

    def fake_digest(path: Path, algorithm: str = "md5") -> str:
        return hashlib.new(algorithm, path.read_bytes()).hexdigest()

    monkeypatch.setattr(MODULE, "digest", fake_digest)
    monkeypatch.setattr(
        MODULE,
        "video_metadata",
        lambda _path: {
            "fps": 25.0,
            "frames": 50,
            "duration_seconds": 2.0,
            "width": 640,
            "height": 480,
        },
    )

    manifest = MODULE.build_manifest(root, per_subject_label=2)
    records = manifest["records"]
    assert len(records) == 84
    assert all(record["small_validation_subset"] for record in records)
    assert manifest["protocol"]["threshold_development"] == MODULE.DEVELOPMENT_SUBJECTS
    assert manifest["protocol"]["threshold_validation"] == MODULE.VALIDATION_SUBJECTS
    assert manifest["protocol"]["locked_test"] == MODULE.LOCKED_TEST_SUBJECTS
    assert manifest["audit"]["labels"] == {"adl": 42, "fall": 42}
    assert manifest["audit"]["partitions"] == {
        "locked_test": 16,
        "threshold_development": 48,
        "threshold_validation": 20,
    }
    assert manifest["audit"]["subject_leakage"] is False
    assert manifest["audit"]["auxiliary_videos_excluded"] == ["Fall/timelapse_20x.mp4"]
    assert manifest["audit"]["source_subject_videos"] == 85
    assert len(manifest["audit"]["duplicate_videos_excluded"]) == 1
    assert manifest["audit"]["duplicate_content_hashes"] == []
