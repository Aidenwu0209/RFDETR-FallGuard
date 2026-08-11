"""Read-only local status summaries for the UI; never reveal secret values."""

from __future__ import annotations

import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fallguard.threshold_selection import is_pending_threshold_lock, is_threshold_confirmation


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _file_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _first_json(paths: tuple[Path, ...]) -> tuple[dict[str, Any] | None, Path | None]:
    for path in paths:
        value = _read_json(path)
        if value is not None:
            return value, path
    return None, None


def _manifest_summary(path: Path) -> dict[str, Any] | None:
    manifest = _read_json(path)
    if manifest is None:
        return None
    records = manifest.get("records")
    small_count = (
        sum(bool(record.get("small_validation_subset")) for record in records)
        if isinstance(records, list)
        else None
    )
    return {
        key: value
        for key, value in {
            "dataset": manifest.get("dataset"),
            "version": manifest.get("version"),
            "source": manifest.get("source"),
            "doi": manifest.get("doi"),
            "repository_license": manifest.get("repository_license"),
            "archive": manifest.get("archive"),
            "protocol": manifest.get("protocol"),
            "audit": manifest.get("audit"),
            "small_validation_videos": small_count,
        }.items()
        if value is not None
    }


def _fallen_audit_summary(path: Path) -> dict[str, Any] | None:
    audit = _read_json(path)
    if audit is None:
        return None
    return {
        key: audit.get(key)
        for key in (
            "dataset",
            "version",
            "declared_license",
            "rfdetr_contiguous_class_names",
            "splits",
            "images_total",
            "cross_split_exact_duplicates",
            "cross_split_near_duplicate_pair_count",
            "structure_and_labels_valid",
            "training_ready",
            "formal_original_split_evaluation_eligible",
            "group_isolation",
            "semantic_mapping",
        )
    }


def environment_status(root: str | Path = ".") -> dict[str, Any]:
    project_root = Path(root)
    cuda: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda = {
            "available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        cuda["detail"] = "torch package missing"
    official = {}
    for variant in ("nano", "small"):
        weights = project_root / f"weights/official/rf-detr-{variant}.pth"
        report_path = project_root / f"artifacts/real-smoke/{variant}-smoke-report.json"
        official[variant] = {
            **_file_status(weights),
            "smoke_report": _read_json(report_path),
        }
    fine_tuned = [
        _file_status(path)
        for pattern in ("artifacts/rfdetr-training/**/*.pth", "checkpoints/**/*.pth")
        for path in sorted(project_root.glob(pattern))
    ]
    gmd_archive = project_root / "data/raw/gmdcsa24/GMDCSA24-v2.0.zip"
    gmd_manifest = project_root / "data/processed/gmdcsa24/manifest.json"
    fall29_archive = project_root / "data/raw/figshare-fall29/VideoDataset.zip"
    fall29_manifest = project_root / "data/processed/figshare-fall29/manifest.json"
    fallen_root = project_root / "data/raw/fallen-person"
    fallen_audit = project_root / "data/processed/fallen-person/audit.json"
    validation_dir = project_root / "artifacts/validation"
    validation_reports = [
        str(path)
        for path in sorted(validation_dir.rglob("*.json"))
        if (_read_json(path) or {}).get("validation_kind")
        == "GROUPED_CLIP_LEVEL_INTERNAL_VALIDATION"
    ]
    fall29_cycle = validation_dir / "figshare-fall29-v1"
    threshold_lock, threshold_lock_path = _first_json(
        (
            fall29_cycle / "threshold-lock.json",
            validation_dir / "threshold-lock.json",
        )
    )
    threshold_confirmation, threshold_confirmation_path = _first_json(
        (
            fall29_cycle / "threshold-confirmation-migrated.json",
            fall29_cycle / "threshold-confirmation.json",
            validation_dir / "threshold-confirmation.json",
        )
    )
    active_cycle = (
        "figshare-fall29-v1"
        if threshold_lock_path is not None and threshold_lock_path.parent == fall29_cycle
        else "gmdcsa24-v2"
        if threshold_lock_path is not None
        else None
    )
    return {
        "packages": {
            name: _package_version(name)
            for name in ("rfdetr", "torch", "torchvision", "gradio", "supervision")
        },
        "cuda": cuda,
        "models": {"official": official, "fine_tuned_checkpoints": fine_tuned},
        "datasets": {
            "fallen_person": {
                "present": fallen_root.is_dir(),
                "path": str(fallen_root),
                "download_requires_roboflow_login_or_key": not fallen_root.is_dir(),
                "audit": _fallen_audit_summary(fallen_audit),
            },
            "gmdcsa24": {
                **_file_status(gmd_archive),
                "manifest": _manifest_summary(gmd_manifest),
            },
            "figshare_fall29": {
                **_file_status(fall29_archive),
                "manifest": _manifest_summary(fall29_manifest),
            },
        },
        "api_keys": {
            "OPENAI_API_KEY": {"present": bool(os.getenv("OPENAI_API_KEY"))},
            "DEEPSEEK_API_KEY": {"present": bool(os.getenv("DEEPSEEK_API_KEY"))},
            "ROBOFLOW_API_KEY": {"present": bool(os.getenv("ROBOFLOW_API_KEY"))},
            "validation_scope": "local presence/config only",
            "network_or_paid_call_performed": False,
        },
        "validation": {
            "active_cycle": active_cycle,
            "grouped_reports": validation_reports,
            "threshold_lock": threshold_lock,
            "threshold_lock_path": str(threshold_lock_path) if threshold_lock_path else None,
            "threshold_confirmation": threshold_confirmation,
            "threshold_confirmation_path": (
                str(threshold_confirmation_path) if threshold_confirmation_path else None
            ),
            "thresholds_frozen": is_pending_threshold_lock(threshold_lock),
            "thresholds_confirmed_on_group": is_threshold_confirmation(threshold_confirmation),
            "thresholds_confirmed_on_s3": is_threshold_confirmation(threshold_confirmation),
            "locked_test_policy": (
                "Declared test subjects require explicit unlock after grouped confirmation and "
                "never influence threshold selection"
            ),
        },
    }
