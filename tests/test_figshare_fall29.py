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
    for subject in range(1, 30):
        for label_dir in ("ADL", "Fall"):
            for index in range(2):
                path = root / label_dir / f"SBJ_{subject:02d}_LOC1" / f"ACT{index}" / "clip.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{subject}-{label_dir}-{index}".encode())
    auxiliary = root / "Fall" / "timelapse_20x.mp4"
    auxiliary.write_bytes(b"auxiliary")

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
    assert len(records) == 116
    assert all(record["small_validation_subset"] for record in records)
    assert manifest["protocol"]["threshold_development"] == list(range(1, 18))
    assert manifest["protocol"]["threshold_validation"] == list(range(18, 24))
    assert manifest["protocol"]["locked_test"] == list(range(24, 30))
    assert manifest["audit"]["subject_leakage"] is False
    assert manifest["audit"]["auxiliary_videos_excluded"] == ["Fall/timelapse_20x.mp4"]
