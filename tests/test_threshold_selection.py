from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from fallguard.config import AppConfig, load_config
from fallguard.exceptions import ConfigurationError
from fallguard.schemas import DetectionMode
from fallguard.threshold_selection import (
    CANDIDATE_PRESETS,
    clip_metrics,
    confirm_thresholds,
    frozen_config_from_lock,
    generate_candidate_configs,
    select_thresholds,
    validate_locked_test_confirmation,
)

pytestmark = pytest.mark.unit
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def config_snapshot(development_config, variant: str) -> dict[str, Any]:
    config = development_config.model_copy(deep=True)
    config.runtime.profile = "experiment"
    config.detector = config.detector.model_copy(
        update={
            "model_variant": variant,
            "mode": DetectionMode.POSTURE_MULTICLASS,
            "weights_path": f"checkpoints/{variant}/checkpoint_best_total.pth",
            "class_names": {0: "standing", 1: "fallen", 2: "sitting", 3: "lying"},
            "person_class_names": ["standing", "fallen", "sitting", "lying"],
            "posture_groups": {
                "upright": ["standing", "sitting"],
                "fall": ["fallen"],
                "lying": ["lying"],
            },
        }
    )
    config.semantic = config.semantic.model_copy(
        update={
            "provider": "none",
            "model": None,
            "allow_fallback": False,
            "allow_mock": False,
            "fallback_providers": [],
        }
    )
    config.benchmark.formal = False
    return config.model_dump(mode="json")


def rows(partition: str, *, false_positive: bool = False) -> list[dict[str, Any]]:
    prefix = "dev" if partition == "threshold_development" else "s3"
    return [
        {
            "video_id": f"{prefix}-fall-1",
            "subject_id": 1 if prefix == "dev" else 3,
            "partition": partition,
            "expected_fall": True,
            "predicted_fall": True,
            "predicted_event_count": 1,
            "first_event_start_seconds": 1.0,
        },
        {
            "video_id": f"{prefix}-fall-2",
            "subject_id": 2 if prefix == "dev" else 3,
            "partition": partition,
            "expected_fall": True,
            "predicted_fall": True,
            "predicted_event_count": 1,
            "first_event_start_seconds": 1.0,
        },
        {
            "video_id": f"{prefix}-adl-1",
            "subject_id": 1 if prefix == "dev" else 3,
            "partition": partition,
            "expected_fall": False,
            "predicted_fall": false_positive,
            "predicted_event_count": int(false_positive),
            "first_event_start_seconds": 1.0 if false_positive else None,
        },
        {
            "video_id": f"{prefix}-adl-2",
            "subject_id": 2 if prefix == "dev" else 3,
            "partition": partition,
            "expected_fall": False,
            "predicted_fall": False,
            "predicted_event_count": 0,
            "first_event_start_seconds": None,
        },
    ]


def grouped_report(
    development_config,
    variant: str,
    partition: str = "threshold_development",
    *,
    false_positive: bool = False,
) -> dict[str, Any]:
    snapshot = config_snapshot(development_config, variant)
    report_rows = rows(partition, false_positive=false_positive)
    return {
        "validation_kind": "GROUPED_CLIP_LEVEL_INTERNAL_VALIDATION",
        "formal_generalization_claim": False,
        "partition": partition,
        "subset": "deterministic_small_subset",
        "weights_sha256": ("a" if variant == "nano" else "b") * 64,
        "weights_path": f"checkpoints/{variant}/checkpoint_best_total.pth",
        "config_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
        "model_variant": variant,
        "pipeline_parameters": {
            "detector": {
                "confidence_threshold": snapshot["detector"]["confidence_threshold"],
                "class_names": deepcopy(snapshot["detector"]["class_names"]),
                "posture_groups": deepcopy(snapshot["detector"]["posture_groups"]),
            },
            "tracking": deepcopy(snapshot["tracking"]),
            "temporal": deepcopy(snapshot["temporal"]),
            "event": deepcopy(snapshot["event"]),
        },
        "config_snapshot": snapshot,
        "protocol": {
            "partition_unit": "subject",
            "threshold_development": [1, 2],
            "threshold_validation": [3],
            "locked_test": [4],
        },
        "selected_video_ids": sorted(row["video_id"] for row in report_rows),
        "metrics_by_partition": {partition: clip_metrics(report_rows)},
        "rows": report_rows,
        "paid_api_call_performed": False,
    }


def test_predeclared_candidates_fill_thresholds_without_claiming_validation(
    development_config,
) -> None:
    snapshot = config_snapshot(development_config, "nano")
    base = AppConfig.model_validate(snapshot)
    candidates = generate_candidate_configs(base)
    assert set(candidates) == set(CANDIDATE_PRESETS)
    assert len({str(value["temporal"]) for value in candidates.values()}) == len(candidates)
    for raw in candidates.values():
        candidate = AppConfig.model_validate(raw)
        assert candidate.temporal.missing_thresholds() == []
        assert candidate.benchmark.formal is False
        candidate.benchmark.formal = True
        candidate.assert_formal_ready()


def test_selection_prefers_nano_on_exact_metric_tie_and_emits_formal_config(
    development_config,
) -> None:
    nano = grouped_report(development_config, "nano")
    small = grouped_report(development_config, "small")
    lock = select_thresholds(
        [("nano.json", nano), ("small.json", small)],
        minimum_recall=1.0,
        maximum_false_positive_clips=0,
    )
    assert lock["selected"]["model_variant"] == "nano"
    assert lock["locked_test"]["status"] == "locked"
    frozen = AppConfig.model_validate(frozen_config_from_lock(lock))
    frozen.assert_formal_ready()
    assert frozen.detector.weights_path is not None
    assert frozen.benchmark.formal is True


def test_selection_rejects_non_development_or_noncomparable_reports(development_config) -> None:
    development = grouped_report(development_config, "nano")
    validation = grouped_report(development_config, "small", partition="threshold_validation")
    with pytest.raises(ConfigurationError, match="partition must be threshold_development"):
        select_thresholds(
            [("dev.json", development), ("s3.json", validation)],
            minimum_recall=1.0,
            maximum_false_positive_clips=0,
        )

    mismatched = grouped_report(development_config, "small")
    mismatched["rows"][0]["video_id"] = "different-video"
    mismatched["selected_video_ids"] = sorted(row["video_id"] for row in mismatched["rows"])
    with pytest.raises(ConfigurationError, match="same development videos"):
        select_thresholds(
            [("nano.json", development), ("small.json", mismatched)],
            minimum_recall=1.0,
            maximum_false_positive_clips=0,
        )


def test_selection_fails_when_no_candidate_meets_declared_gate(development_config) -> None:
    report = grouped_report(development_config, "nano", false_positive=True)
    with pytest.raises(ConfigurationError, match="no development candidate"):
        select_thresholds(
            [("nano.json", report)],
            minimum_recall=1.0,
            maximum_false_positive_clips=0,
        )


def test_selection_rejects_subject_leakage_and_snapshot_parameter_drift(
    development_config,
) -> None:
    leaked = grouped_report(development_config, "nano")
    leaked["rows"][0]["subject_id"] = 3
    with pytest.raises(ConfigurationError, match="subject outside"):
        select_thresholds(
            [("leaked.json", leaked)],
            minimum_recall=1.0,
            maximum_false_positive_clips=0,
        )

    drifted = grouped_report(development_config, "nano")
    drifted["pipeline_parameters"]["temporal"]["suspect_duration_seconds"] = 7.0
    with pytest.raises(ConfigurationError, match="parameters differ"):
        select_thresholds(
            [("drifted.json", drifted)],
            minimum_recall=1.0,
            maximum_false_positive_clips=0,
        )


def test_s3_confirmation_requires_unchanged_parameters_and_disjoint_videos(
    development_config,
) -> None:
    development = grouped_report(development_config, "nano")
    lock = select_thresholds(
        [("nano.json", development)],
        minimum_recall=1.0,
        maximum_false_positive_clips=0,
    )
    validation = grouped_report(development_config, "nano", partition="threshold_validation")
    confirmation = confirm_thresholds(
        lock,
        validation,
        minimum_recall=1.0,
        maximum_false_positive_clips=0,
    )
    assert confirmation["formal_thresholds_confirmed"] is True
    assert confirmation["confirmation_policy"]["parameters_retuned_on_s3"] is False
    assert confirmation["locked_test"]["status"] == "locked_pending_final_evaluation"
    validate_locked_test_confirmation(
        confirmation,
        manifest_sha256=validation["manifest_sha256"],
        protocol=validation["protocol"],
        model_variant=validation["model_variant"],
        weights_sha256=validation["weights_sha256"],
        pipeline_parameters=validation["pipeline_parameters"],
    )

    changed = deepcopy(validation)
    changed["pipeline_parameters"]["temporal"]["suspect_duration_seconds"] = 99.0
    with pytest.raises(ConfigurationError, match="parameters differ"):
        confirm_thresholds(
            lock,
            changed,
            minimum_recall=1.0,
            maximum_false_positive_clips=0,
        )

    overlapping = deepcopy(validation)
    overlapping["rows"][0]["video_id"] = development["rows"][0]["video_id"]
    overlapping["selected_video_ids"] = sorted(row["video_id"] for row in overlapping["rows"])
    with pytest.raises(ConfigurationError, match="reuses a threshold-development video"):
        confirm_thresholds(
            lock,
            overlapping,
            minimum_recall=1.0,
            maximum_false_positive_clips=0,
        )

    with pytest.raises(ConfigurationError, match="locked-test configuration differs"):
        validate_locked_test_confirmation(
            confirmation,
            manifest_sha256=validation["manifest_sha256"],
            protocol=validation["protocol"],
            model_variant="small",
            weights_sha256=validation["weights_sha256"],
            pipeline_parameters=validation["pipeline_parameters"],
        )


@pytest.mark.integration
def test_threshold_candidate_freeze_and_confirmation_clis(development_config, tmp_path) -> None:
    base_config = tmp_path / "posture-profile.yaml"
    base_config.write_text(
        yaml.safe_dump(config_snapshot(development_config, "nano"), sort_keys=False),
        encoding="utf-8",
    )
    candidates_dir = tmp_path / "candidates"
    generated = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/generate_threshold_candidates.py"),
            "--base-config",
            str(base_config),
            "--output-dir",
            str(candidates_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert generated.returncode == 0, generated.stderr
    manifest = json.loads((candidates_dir / "candidate-manifest.json").read_text())
    assert manifest["candidate_count"] == 4

    nano_path = tmp_path / "nano-dev.json"
    small_path = tmp_path / "small-dev.json"
    nano_path.write_text(json.dumps(grouped_report(development_config, "nano")), encoding="utf-8")
    small_path.write_text(json.dumps(grouped_report(development_config, "small")), encoding="utf-8")
    lock_path = tmp_path / "threshold-lock.json"
    frozen_path = tmp_path / "frozen-profile.yaml"
    selected = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/select_thresholds.py"),
            "--development-report",
            str(nano_path),
            "--development-report",
            str(small_path),
            "--output-lock",
            str(lock_path),
            "--output-config",
            str(frozen_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert selected.returncode == 0, selected.stderr
    load_config(frozen_path).assert_formal_ready()

    s3_path = tmp_path / "s3.json"
    s3_path.write_text(
        json.dumps(grouped_report(development_config, "nano", "threshold_validation")),
        encoding="utf-8",
    )
    confirmation_path = tmp_path / "threshold-confirmation.json"
    confirmed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/confirm_thresholds.py"),
            "--threshold-lock",
            str(lock_path),
            "--validation-report",
            str(s3_path),
            "--output-json",
            str(confirmation_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert confirmed.returncode == 0, confirmed.stderr
    confirmation = json.loads(confirmation_path.read_text())
    assert confirmation["formal_thresholds_confirmed"] is True
