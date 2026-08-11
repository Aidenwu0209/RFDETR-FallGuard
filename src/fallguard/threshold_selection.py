"""Leakage-resistant threshold selection and confirmation for grouped fall validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fallguard.config import AppConfig
from fallguard.exceptions import ConfigurationError, FormalBenchmarkRejectedError

DEVELOPMENT_PARTITION = "threshold_development"
VALIDATION_PARTITION = "threshold_validation"
LOCKED_TEST_PARTITION = "locked_test"
GROUPED_REPORT_KIND = "GROUPED_CLIP_LEVEL_INTERNAL_VALIDATION"
THRESHOLD_LOCK_KIND = "THRESHOLD_LOCK_PENDING_S3_CONFIRMATION"
THRESHOLD_CONFIRMATION_KIND = "THRESHOLD_LOCK_CONFIRMED_ON_S3"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
CANDIDATE_PRESETS: dict[str, dict[str, dict[str, int | float]]] = {
    "high_recall": {
        "detector": {"confidence_threshold": 0.18},
        "tracking": {"track_activation_threshold": 0.18},
        "temporal": {
            "smoothing_window": 3,
            "aspect_ratio_fall_min": 0.8,
            "vertical_speed_frame_height_per_second_min": 0.15,
            "suspect_duration_seconds": 0.25,
            "lying_duration_seconds": 0.5,
            "upright_aspect_ratio_max": 0.78,
            "track_timeout_seconds": 1.2,
        },
    },
    "balanced_short": {
        "detector": {"confidence_threshold": 0.25},
        "tracking": {"track_activation_threshold": 0.25},
        "temporal": {
            "smoothing_window": 5,
            "aspect_ratio_fall_min": 0.85,
            "vertical_speed_frame_height_per_second_min": 0.2,
            "suspect_duration_seconds": 0.3,
            "lying_duration_seconds": 0.6,
            "upright_aspect_ratio_max": 0.75,
            "track_timeout_seconds": 1.2,
        },
    },
    "balanced_duration": {
        "detector": {"confidence_threshold": 0.3},
        "tracking": {"track_activation_threshold": 0.25},
        "temporal": {
            "smoothing_window": 5,
            "aspect_ratio_fall_min": 0.95,
            "vertical_speed_frame_height_per_second_min": 0.25,
            "suspect_duration_seconds": 0.4,
            "lying_duration_seconds": 0.8,
            "upright_aspect_ratio_max": 0.72,
            "track_timeout_seconds": 1.0,
        },
    },
    "high_precision": {
        "detector": {"confidence_threshold": 0.4},
        "tracking": {"track_activation_threshold": 0.3},
        "temporal": {
            "smoothing_window": 7,
            "aspect_ratio_fall_min": 1.05,
            "vertical_speed_frame_height_per_second_min": 0.35,
            "suspect_duration_seconds": 0.6,
            "lying_duration_seconds": 1.0,
            "upright_aspect_ratio_max": 0.68,
            "track_timeout_seconds": 0.8,
        },
    },
}
EXPANDED_PRECISION_PRESETS: dict[str, dict[str, dict[str, int | float]]] = {
    name: {
        "detector": {"confidence_threshold": confidence},
        "tracking": {"track_activation_threshold": confidence},
        "temporal": dict(CANDIDATE_PRESETS["high_precision"]["temporal"]),
    }
    for name, confidence in (
        ("precision_060", 0.60),
        ("precision_070", 0.70),
        ("precision_075", 0.75),
        ("precision_080", 0.80),
    )
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ConfigurationError(f"JSON file does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid JSON file: {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"JSON root must be an object: {source}")
    return value


def generate_candidate_configs(
    base_config: AppConfig, *, include_expanded_precision_grid: bool = False
) -> dict[str, dict[str, Any]]:
    if base_config.detector.mode.value != "posture_multiclass":
        raise ConfigurationError("threshold candidates require posture_multiclass mode")
    if not base_config.detector.class_names:
        raise ConfigurationError("threshold candidates require audited posture class names")
    generated: dict[str, dict[str, Any]] = {}
    presets = dict(CANDIDATE_PRESETS)
    if include_expanded_precision_grid:
        presets.update(EXPANDED_PRECISION_PRESETS)
    for name, updates in presets.items():
        candidate = base_config.model_copy(deep=True)
        candidate.runtime.profile = "experiment"
        candidate.detector = candidate.detector.model_copy(update=updates["detector"])
        candidate.tracking = candidate.tracking.model_copy(update=updates["tracking"])
        candidate.temporal = candidate.temporal.model_copy(update=updates["temporal"])
        candidate.semantic = candidate.semantic.model_copy(
            update={
                "provider": "none",
                "model": None,
                "fallback_providers": [],
                "allow_fallback": False,
                "allow_mock": False,
                "allow_cloud_images": False,
            }
        )
        candidate.benchmark.formal = False
        if candidate.temporal.missing_thresholds():
            raise ConfigurationError(f"candidate {name} leaves temporal thresholds unset")
        generated[name] = candidate.model_dump(mode="json")
    return generated


def clip_metrics(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
    tp = sum(bool(row["expected_fall"]) and bool(row["predicted_fall"]) for row in rows)
    fp = sum(not bool(row["expected_fall"]) and bool(row["predicted_fall"]) for row in rows)
    fn = sum(bool(row["expected_fall"]) and not bool(row["predicted_fall"]) for row in rows)
    tn = sum(not bool(row["expected_fall"]) and not bool(row["predicted_fall"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "clips": len(rows),
        "true_positive_clips": tp,
        "false_positive_clips": fp,
        "false_negative_clips": fn,
        "true_negative_clips": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ConfigurationError(f"grouped report has invalid {field}")
    return value


def _metrics_equal(observed: dict[str, Any], expected: dict[str, int | float | None]) -> bool:
    if set(observed) != set(expected):
        return False
    for key, expected_value in expected.items():
        observed_value = observed[key]
        if isinstance(expected_value, float):
            if not isinstance(observed_value, int | float) or not math.isclose(
                float(observed_value), expected_value, rel_tol=1e-12, abs_tol=1e-12
            ):
                return False
        elif observed_value != expected_value:
            return False
    return True


def validate_grouped_report(report: dict[str, Any], *, expected_partition: str) -> dict[str, Any]:
    if report.get("validation_kind") != GROUPED_REPORT_KIND:
        raise ConfigurationError("not a grouped clip-level validation report")
    implementation_revision = report.get("implementation_git_commit")
    if (
        not isinstance(implementation_revision, str)
        or GIT_COMMIT_PATTERN.fullmatch(implementation_revision) is None
    ):
        raise ConfigurationError("grouped report has no valid implementation Git commit")
    pipeline_fingerprint = _require_sha256(
        report.get("pipeline_implementation_sha256"), "pipeline_implementation_sha256"
    )
    if report.get("partition") != expected_partition:
        raise ConfigurationError(
            f"report partition must be {expected_partition}, got {report.get('partition')}"
        )
    rows_raw = report.get("rows")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise ConfigurationError("grouped report has no clip rows")
    rows: list[dict[str, Any]] = []
    video_ids: list[str] = []
    allowed_subjects = {
        DEVELOPMENT_PARTITION: {1, 2},
        VALIDATION_PARTITION: {3},
        LOCKED_TEST_PARTITION: {4},
    }.get(expected_partition)
    if allowed_subjects is None:
        raise ConfigurationError(f"unknown grouped partition: {expected_partition}")
    for raw in rows_raw:
        if not isinstance(raw, dict):
            raise ConfigurationError("grouped report contains an invalid clip row")
        if raw.get("partition") != expected_partition:
            raise ConfigurationError("grouped report mixes protocol partitions")
        video_id = raw.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ConfigurationError("grouped report contains an invalid video_id")
        if not isinstance(raw.get("expected_fall"), bool) or not isinstance(
            raw.get("predicted_fall"), bool
        ):
            raise ConfigurationError("grouped report clip labels must be booleans")
        if raw.get("subject_id") not in allowed_subjects:
            raise ConfigurationError("grouped report contains a subject outside its partition")
        rows.append(raw)
        video_ids.append(video_id)
    if len(set(video_ids)) != len(video_ids):
        raise ConfigurationError("grouped report contains duplicate video IDs")
    selected_video_ids = report.get("selected_video_ids")
    if selected_video_ids != sorted(video_ids):
        raise ConfigurationError("grouped report selected_video_ids do not match its clip rows")

    grouped_metrics = report.get("metrics_by_partition")
    if not isinstance(grouped_metrics, dict) or set(grouped_metrics) != {expected_partition}:
        raise ConfigurationError(
            "grouped report metrics must contain exactly one requested partition"
        )
    saved_metrics = grouped_metrics[expected_partition]
    computed_metrics = clip_metrics(rows)
    if not isinstance(saved_metrics, dict) or not _metrics_equal(saved_metrics, computed_metrics):
        raise ConfigurationError("grouped report metrics do not match its clip rows")

    model_variant = report.get("model_variant")
    if model_variant not in {"nano", "small"}:
        raise ConfigurationError("grouped report has an invalid model variant")
    _require_sha256(report.get("weights_sha256"), "weights_sha256")
    _require_sha256(report.get("config_sha256"), "config_sha256")
    _require_sha256(report.get("manifest_sha256"), "manifest_sha256")
    if not isinstance(report.get("weights_path"), str) or not report["weights_path"]:
        raise ConfigurationError("grouped report has no weights path")
    parameters = report.get("pipeline_parameters")
    if not isinstance(parameters, dict) or set(parameters) != {
        "detector",
        "tracking",
        "temporal",
        "event",
    }:
        raise ConfigurationError("grouped report has incomplete pipeline parameters")
    snapshot = report.get("config_snapshot")
    if not isinstance(snapshot, dict):
        raise ConfigurationError("grouped report has no config snapshot")
    snapshot_detector = snapshot.get("detector")
    if not isinstance(snapshot_detector, dict):
        raise ConfigurationError("grouped report config snapshot has no detector section")
    if snapshot_detector.get("mode") != "posture_multiclass":
        raise ConfigurationError("grouped report was not produced in posture_multiclass mode")
    if snapshot_detector.get("model_variant") != model_variant:
        raise ConfigurationError("grouped report model variant differs from its config snapshot")
    if str(snapshot_detector.get("weights_path")) != report["weights_path"]:
        raise ConfigurationError("grouped report weights path differs from its config snapshot")
    snapshot_parameters = {
        "detector": {
            "confidence_threshold": snapshot_detector.get("confidence_threshold"),
            "class_names": snapshot_detector.get("class_names"),
            "posture_groups": snapshot_detector.get("posture_groups"),
        },
        "tracking": snapshot.get("tracking"),
        "temporal": snapshot.get("temporal"),
        "event": snapshot.get("event"),
    }
    if parameters != snapshot_parameters:
        raise ConfigurationError("grouped report parameters differ from its config snapshot")
    protocol = report.get("protocol")
    if not isinstance(protocol, dict):
        raise ConfigurationError("grouped report has no protocol")
    expected_protocol = {
        "partition_unit": "subject",
        DEVELOPMENT_PARTITION: [1, 2],
        VALIDATION_PARTITION: [3],
        LOCKED_TEST_PARTITION: [4],
    }
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise ConfigurationError("grouped report protocol does not preserve the subject split")
    return {
        "report": report,
        "rows": rows,
        "video_ids": sorted(video_ids),
        "metrics": computed_metrics,
        "model_variant": model_variant,
        "implementation_git_commit": implementation_revision,
        "pipeline_implementation_sha256": pipeline_fingerprint,
    }


def _candidate_sort_key(
    candidate: dict[str, Any], preferred_variant: str
) -> tuple[int, float, float, float, int, float, str, str]:
    metrics = candidate["metrics"]
    f1 = float(metrics["f1"]) if metrics["f1"] is not None else -1.0
    recall = float(metrics["recall"]) if metrics["recall"] is not None else -1.0
    specificity = float(metrics["specificity"]) if metrics["specificity"] is not None else -1.0
    return (
        int(metrics["false_positive_clips"]),
        -f1,
        -recall,
        -specificity,
        0 if candidate["model_variant"] == preferred_variant else 1,
        float(candidate["report"]["pipeline_parameters"]["detector"]["confidence_threshold"]),
        str(candidate["model_variant"]),
        str(candidate["candidate_id"]),
    )


def select_thresholds(
    candidates: list[tuple[str, dict[str, Any]]],
    *,
    minimum_recall: float,
    maximum_false_positive_clips: int,
    preferred_variant: str = "nano",
) -> dict[str, Any]:
    if not 0.0 <= minimum_recall <= 1.0:
        raise ConfigurationError("minimum_recall must be in [0, 1]")
    if maximum_false_positive_clips < 0:
        raise ConfigurationError("maximum_false_positive_clips cannot be negative")
    if preferred_variant not in {"nano", "small"}:
        raise ConfigurationError("preferred_variant must be nano or small")
    if not candidates:
        raise ConfigurationError("at least one development report is required")

    validated: list[dict[str, Any]] = []
    for source, report in candidates:
        item = validate_grouped_report(report, expected_partition=DEVELOPMENT_PARTITION)
        item["source"] = source
        item["report_sha256"] = canonical_sha256(report)
        item["candidate_id"] = canonical_sha256(
            {
                "weights_sha256": report["weights_sha256"],
                "model_variant": report["model_variant"],
                "pipeline_implementation_sha256": report["pipeline_implementation_sha256"],
                "pipeline_parameters": report["pipeline_parameters"],
            }
        )
        validated.append(item)

    reference = validated[0]["report"]
    reference_video_ids = validated[0]["video_ids"]
    for item in validated[1:]:
        report = item["report"]
        if report["manifest_sha256"] != reference["manifest_sha256"]:
            raise ConfigurationError("candidate reports use different dataset manifests")
        if report["protocol"] != reference["protocol"]:
            raise ConfigurationError("candidate reports use different grouped protocols")
        if report["pipeline_implementation_sha256"] != reference["pipeline_implementation_sha256"]:
            raise ConfigurationError("candidate reports use different pipeline implementations")
        if item["video_ids"] != reference_video_ids:
            raise ConfigurationError(
                "candidate reports do not evaluate the same development videos"
            )
    candidate_ids = [str(item["candidate_id"]) for item in validated]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ConfigurationError("duplicate threshold/model candidate reports were supplied")

    eligible = [
        item
        for item in validated
        if item["metrics"]["recall"] is not None
        and float(item["metrics"]["recall"]) >= minimum_recall
        and int(item["metrics"]["false_positive_clips"]) <= maximum_false_positive_clips
    ]
    if not eligible:
        raise ConfigurationError(
            "no development candidate satisfies the declared recall/false-positive gate"
        )
    selected = sorted(eligible, key=lambda item: _candidate_sort_key(item, preferred_variant))[0]
    selected_report = selected["report"]
    return {
        "lock_kind": THRESHOLD_LOCK_KIND,
        "selection_partition": DEVELOPMENT_PARTITION,
        "selection_policy": {
            "minimum_recall": minimum_recall,
            "maximum_false_positive_clips": maximum_false_positive_clips,
            "ranking": [
                "fewest_false_positive_clips",
                "highest_f1",
                "highest_recall",
                "highest_specificity",
                f"prefer_{preferred_variant}_on_exact_metric_tie",
                "lowest_detector_confidence_on_remaining_tie",
                "deterministic_candidate_id",
            ],
        },
        "manifest_sha256": reference["manifest_sha256"],
        "pipeline_implementation_sha256": reference["pipeline_implementation_sha256"],
        "protocol": deepcopy(reference["protocol"]),
        "development_video_ids": reference_video_ids,
        "candidate_count": len(validated),
        "eligible_candidate_count": len(eligible),
        "candidate_reports": [
            {
                "source": item["source"],
                "report_sha256": item["report_sha256"],
                "candidate_id": item["candidate_id"],
                "model_variant": item["model_variant"],
                "metrics": item["metrics"],
            }
            for item in validated
        ],
        "selected": {
            "source": selected["source"],
            "report_sha256": selected["report_sha256"],
            "candidate_id": selected["candidate_id"],
            "model_variant": selected_report["model_variant"],
            "implementation_git_commit": selected_report["implementation_git_commit"],
            "pipeline_implementation_sha256": selected_report["pipeline_implementation_sha256"],
            "weights_path": selected_report["weights_path"],
            "weights_sha256": selected_report["weights_sha256"],
            "config_sha256": selected_report["config_sha256"],
            "pipeline_parameters": deepcopy(selected_report["pipeline_parameters"]),
            "development_metrics": selected["metrics"],
            "config_snapshot": deepcopy(selected_report["config_snapshot"]),
        },
        "next_gate": {
            "partition": VALIDATION_PARTITION,
            "reuse_parameters_without_retuning": True,
            "status": "pending",
        },
        "locked_test": {
            "partition": LOCKED_TEST_PARTITION,
            "status": "locked",
            "must_not_influence_selection": True,
        },
        "formal_generalization_claim": False,
    }


def frozen_config_from_lock(lock: dict[str, Any]) -> dict[str, Any]:
    if lock.get("lock_kind") != THRESHOLD_LOCK_KIND:
        raise ConfigurationError("not a pending S3 threshold lock")
    selected = lock.get("selected")
    if not isinstance(selected, dict) or not isinstance(selected.get("config_snapshot"), dict):
        raise ConfigurationError("threshold lock has no selected config snapshot")
    config = deepcopy(selected["config_snapshot"])
    detector = config.get("detector")
    runtime = config.get("runtime")
    benchmark = config.get("benchmark")
    if (
        not isinstance(detector, dict)
        or not isinstance(runtime, dict)
        or not isinstance(benchmark, dict)
    ):
        raise ConfigurationError("selected config snapshot is incomplete")
    runtime["profile"] = "experiment"
    detector.update(
        {
            "model_variant": selected["model_variant"],
            "weights_path": selected["weights_path"],
            "allow_weight_download": False,
        }
    )
    benchmark["formal"] = True
    try:
        validated = AppConfig.model_validate(config)
        validated.assert_formal_ready()
    except (ValidationError, FormalBenchmarkRejectedError) as exc:
        raise ConfigurationError(
            f"selected config cannot be frozen for formal validation: {exc}"
        ) from exc
    return validated.model_dump(mode="json")


def confirm_thresholds(
    lock: dict[str, Any],
    validation_report: dict[str, Any],
    *,
    minimum_recall: float,
    maximum_false_positive_clips: int,
) -> dict[str, Any]:
    if lock.get("lock_kind") != THRESHOLD_LOCK_KIND:
        raise ConfigurationError("not a pending S3 threshold lock")
    if not 0.0 <= minimum_recall <= 1.0:
        raise ConfigurationError("minimum_recall must be in [0, 1]")
    if maximum_false_positive_clips < 0:
        raise ConfigurationError("maximum_false_positive_clips cannot be negative")
    selected = lock.get("selected")
    if not isinstance(selected, dict):
        raise ConfigurationError("threshold lock has no selected candidate")
    validated = validate_grouped_report(validation_report, expected_partition=VALIDATION_PARTITION)
    for field in ("manifest_sha256", "protocol", "pipeline_implementation_sha256"):
        if validation_report[field] != lock.get(field):
            raise ConfigurationError(f"S3 report {field} differs from the threshold lock")
    for field in ("model_variant", "weights_sha256", "pipeline_parameters"):
        if validation_report[field] != selected.get(field):
            raise ConfigurationError(f"S3 report {field} differs from the selected candidate")
    development_video_ids = lock.get("development_video_ids")
    if not isinstance(development_video_ids, list):
        raise ConfigurationError("threshold lock has no development video list")
    if set(development_video_ids) & set(validated["video_ids"]):
        raise ConfigurationError("S3 report reuses a threshold-development video")
    metrics = validated["metrics"]
    recall = metrics["recall"]
    if (
        recall is None
        or float(recall) < minimum_recall
        or int(metrics["false_positive_clips"]) > maximum_false_positive_clips
    ):
        raise ConfigurationError("S3 report does not satisfy the declared confirmation gate")
    return {
        "confirmation_kind": THRESHOLD_CONFIRMATION_KIND,
        "threshold_lock_sha256": canonical_sha256(lock),
        "validation_report_sha256": canonical_sha256(validation_report),
        "manifest_sha256": lock["manifest_sha256"],
        "protocol": deepcopy(lock["protocol"]),
        "validation_partition": VALIDATION_PARTITION,
        "validation_video_ids": validated["video_ids"],
        "confirmation_policy": {
            "minimum_recall": minimum_recall,
            "maximum_false_positive_clips": maximum_false_positive_clips,
            "parameters_retuned_on_s3": False,
        },
        "selected": {
            key: deepcopy(selected[key])
            for key in (
                "candidate_id",
                "model_variant",
                "implementation_git_commit",
                "pipeline_implementation_sha256",
                "weights_path",
                "weights_sha256",
                "pipeline_parameters",
            )
        },
        "validation_metrics": metrics,
        "validation_implementation_git_commit": validation_report["implementation_git_commit"],
        "locked_test": {
            "partition": LOCKED_TEST_PARTITION,
            "status": "locked_pending_final_evaluation",
            "must_not_influence_thresholds": True,
        },
        "formal_thresholds_confirmed": True,
        "formal_generalization_claim": False,
        "detection_delay_available": False,
        "detection_delay_unavailable_reason": (
            "GMDCSA-24 supplies clip labels but no human-confirmed fall onset timestamps"
        ),
    }


def validate_locked_test_confirmation(
    confirmation: dict[str, Any],
    *,
    manifest_sha256: str,
    protocol: dict[str, Any],
    model_variant: str,
    weights_sha256: str,
    pipeline_parameters: dict[str, Any],
    pipeline_implementation_sha256: str,
) -> None:
    if confirmation.get("confirmation_kind") != THRESHOLD_CONFIRMATION_KIND:
        raise ConfigurationError("not an accepted S3 confirmation artifact")
    selected = confirmation.get("selected")
    if not isinstance(selected, dict):
        raise ConfigurationError("S3 confirmation has no selected candidate")
    if confirmation.get("manifest_sha256") != manifest_sha256:
        raise ConfigurationError("S3 confirmation uses a different GMDCSA-24 manifest")
    if confirmation.get("protocol") != protocol:
        raise ConfigurationError("S3 confirmation uses a different grouped protocol")
    if selected.get("pipeline_implementation_sha256") != pipeline_implementation_sha256:
        raise ConfigurationError("locked-test implementation differs from the S3 confirmation")
    if (
        selected.get("model_variant") != model_variant
        or selected.get("weights_sha256") != weights_sha256
        or selected.get("pipeline_parameters") != pipeline_parameters
    ):
        raise ConfigurationError("locked-test configuration differs from the S3 confirmation")
