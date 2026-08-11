"""Audits for the external posture dataset before RF-DETR training."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fallguard.exceptions import ConfigurationError

EXPECTED_FALLEN_PERSON_CLASSES = {"standing", "fallen", "lying", "sitting"}
ROBOFLOW_SPLITS = {"train": "train", "validation": "valid", "test": "test"}


def _read_coco(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"COCO annotation file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid COCO JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"COCO root must be an object: {path}")
    for key in ("images", "annotations", "categories"):
        if not isinstance(value.get(key), list):
            raise ConfigurationError(f"COCO {key} must be a list: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference_hash(path: Path) -> int:
    with Image.open(path) as source:
        pixels = np.asarray(source.convert("L").resize((9, 8)), dtype=np.uint8).ravel()
    bits = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            bits = (bits << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return bits


def _categories(coco: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    categories = coco["categories"]
    normalized: list[dict[str, Any]] = []
    for raw in categories:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), int):
            raise ConfigurationError(f"invalid COCO category in {source}")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError(f"empty COCO category name in {source}")
        normalized.append({"id": raw["id"], "name": name.strip()})
    ids = [item["id"] for item in normalized]
    names = [item["name"].lower() for item in normalized]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ConfigurationError(f"duplicate COCO category id/name in {source}")
    normalized.sort(key=lambda item: item["id"])
    return normalized


def _audit_split(
    dataset_root: Path,
    logical_split: str,
    directory_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    split_root = dataset_root / directory_name
    annotation_path = split_root / "_annotations.coco.json"
    coco = _read_coco(annotation_path)
    categories = _categories(coco, annotation_path)
    category_ids = {item["id"] for item in categories}
    images_by_id: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for raw in coco["images"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), int):
            raise ConfigurationError(f"invalid image record in {annotation_path}")
        image_id = raw["id"]
        if image_id in images_by_id:
            raise ConfigurationError(f"duplicate image id {image_id} in {annotation_path}")
        filename = raw.get("file_name")
        width = raw.get("width")
        height = raw.get("height")
        if (
            not isinstance(filename, str)
            or not filename
            or not isinstance(width, int)
            or width <= 0
            or not isinstance(height, int)
            or height <= 0
        ):
            raise ConfigurationError(f"invalid image metadata for id {image_id}")
        image_path = (split_root / filename).resolve()
        if not image_path.is_relative_to(split_root.resolve()):
            raise ConfigurationError(f"image path escapes its split directory: {filename}")
        if not image_path.is_file():
            raise ConfigurationError(f"referenced image is missing: {image_path}")
        try:
            with Image.open(image_path) as opened_image:
                actual_size = opened_image.size
                opened_image.verify()
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"unreadable image: {image_path}: {exc}") from exc
        if actual_size != (width, height):
            raise ConfigurationError(
                f"image dimensions disagree with COCO metadata: {image_path}: "
                f"{actual_size} != {(width, height)}"
            )
        images_by_id[image_id] = raw
        records.append(
            {
                "split": logical_split,
                "image_id": image_id,
                "relative_path": str(image_path.relative_to(dataset_root)),
                "width": width,
                "height": height,
                "bytes": image_path.stat().st_size,
                "sha256": _sha256(image_path),
                "dhash64": f"{_difference_hash(image_path):016x}",
            }
        )
    annotation_ids: set[int] = set()
    class_counts: Counter[int] = Counter()
    for raw in coco["annotations"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), int):
            raise ConfigurationError(f"invalid annotation record in {annotation_path}")
        annotation_id = raw["id"]
        if annotation_id in annotation_ids:
            raise ConfigurationError(
                f"duplicate annotation id {annotation_id} in {annotation_path}"
            )
        annotation_ids.add(annotation_id)
        image_id = raw.get("image_id")
        category_id = raw.get("category_id")
        bbox = raw.get("bbox")
        if (
            not isinstance(image_id, int)
            or not isinstance(category_id, int)
            or image_id not in images_by_id
            or category_id not in category_ids
        ):
            raise ConfigurationError(f"annotation {annotation_id} has unknown image/category id")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not isinstance(value, (int, float)) for value in bbox)
            or any(not math.isfinite(float(value)) for value in bbox)
        ):
            raise ConfigurationError(f"annotation {annotation_id} has invalid bbox")
        x, y, box_width, box_height = (float(value) for value in bbox)
        image_metadata = images_by_id[image_id]
        if (
            x < 0
            or y < 0
            or box_width <= 0
            or box_height <= 0
            or x + box_width > image_metadata["width"] + 1e-6
            or y + box_height > image_metadata["height"] + 1e-6
        ):
            raise ConfigurationError(f"annotation {annotation_id} bbox lies outside its image")
        class_counts[category_id] += 1
    if not records or not annotation_ids:
        raise ConfigurationError(f"empty images/annotations in {annotation_path}")
    summary = {
        "directory": directory_name,
        "annotation_path": str(annotation_path.relative_to(dataset_root)),
        "annotation_sha256": _sha256(annotation_path),
        "images": len(records),
        "annotations": len(annotation_ids),
        "class_annotations": {
            next(item["name"] for item in categories if item["id"] == category_id): count
            for category_id, count in sorted(class_counts.items())
        },
    }
    return summary, categories, records


def audit_fallen_person(
    dataset_root: str | Path, *, near_duplicate_distance: int = 4
) -> dict[str, Any]:
    root = Path(dataset_root)
    if not root.is_dir():
        raise ConfigurationError(f"dataset root does not exist: {root}")
    split_summaries: dict[str, dict[str, Any]] = {}
    category_schema: list[dict[str, Any]] | None = None
    records: list[dict[str, Any]] = []
    for logical_split, directory_name in ROBOFLOW_SPLITS.items():
        summary, categories, split_records = _audit_split(root, logical_split, directory_name)
        if category_schema is None:
            category_schema = categories
        elif categories != category_schema:
            raise ConfigurationError(f"category schema differs in {logical_split} split")
        split_summaries[logical_split] = summary
        records.extend(split_records)
    assert category_schema is not None
    class_names = [item["name"] for item in category_schema]
    if {name.lower() for name in class_names} != EXPECTED_FALLEN_PERSON_CLASSES:
        raise ConfigurationError(
            f"unexpected Fallen Person classes: {class_names}; "
            f"expected {sorted(EXPECTED_FALLEN_PERSON_CLASSES)}"
        )

    exact_groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        exact_groups.setdefault(record["sha256"], []).append(record)
    cross_split_exact = [
        {
            "sha256": digest,
            "images": [item["relative_path"] for item in group],
            "splits": sorted({item["split"] for item in group}),
        }
        for digest, group in exact_groups.items()
        if len({item["split"] for item in group}) > 1
    ]

    by_split = {
        split: [record for record in records if record["split"] == split]
        for split in ROBOFLOW_SPLITS
    }
    near_pairs: list[dict[str, Any]] = []
    near_pair_count = 0
    for first_split, second_split in combinations(ROBOFLOW_SPLITS, 2):
        for first in by_split[first_split]:
            first_hash = int(first["dhash64"], 16)
            for second in by_split[second_split]:
                distance = (first_hash ^ int(second["dhash64"], 16)).bit_count()
                if distance <= near_duplicate_distance and first["sha256"] != second["sha256"]:
                    near_pair_count += 1
                    if len(near_pairs) < 100:
                        near_pairs.append(
                            {
                                "first": first["relative_path"],
                                "second": second["relative_path"],
                                "dhash_distance": distance,
                            }
                        )
    contiguous_class_names = {index: name for index, name in enumerate(class_names)}
    return {
        "dataset": "Fallen Person",
        "source": "https://universe.roboflow.com/ortaks/fallen-person-uhif8-5qvtq",
        "version": 1,
        "declared_license": "CC BY 4.0",
        "dataset_root": str(root),
        "categories_sorted_by_coco_id": category_schema,
        "rfdetr_contiguous_class_names": contiguous_class_names,
        "splits": split_summaries,
        "images_total": len(records),
        "cross_split_exact_duplicates": cross_split_exact,
        "cross_split_near_duplicate_pair_count": near_pair_count,
        "cross_split_near_duplicate_examples": near_pairs,
        "near_duplicate_method": f"64-bit dHash with Hamming distance <= {near_duplicate_distance}",
        "structure_and_labels_valid": True,
        "training_ready": True,
        "formal_original_split_evaluation_eligible": not cross_split_exact and near_pair_count == 0,
        "group_isolation": {
            "available": False,
            "reason": "export does not declare person_id, video_id, or source sequence groups",
        },
        "semantic_mapping": {
            "upright": ["standing", "sitting"],
            "fall": ["fallen"],
            "lying": ["lying"],
            "warning": "fallen is retained as its own posture label and is not renamed to falling",
        },
        "records": records,
    }


def posture_profile(audit: dict[str, Any]) -> dict[str, Any]:
    class_names = audit["rfdetr_contiguous_class_names"]
    return {
        "runtime": {"profile": "experiment"},
        "detector": {
            "mode": "posture_multiclass",
            "confidence_threshold": 0.25,
            "class_names": class_names,
            "person_class_names": list(class_names.values()),
            "class_aliases": {},
            "posture_groups": {
                key: audit["semantic_mapping"][key] for key in ("upright", "fall", "lying")
            },
        },
        "temporal": {
            "aspect_ratio_fall_min": None,
            "vertical_speed_frame_height_per_second_min": None,
            "suspect_duration_seconds": None,
            "lying_duration_seconds": None,
            "upright_aspect_ratio_max": None,
            "track_timeout_seconds": None,
        },
        "semantic": {
            "provider": "none",
            "model": None,
            "allow_fallback": False,
            "allow_mock": False,
            "fallback_providers": [],
        },
        "benchmark": {"formal": False},
    }


def validate_training_audit(
    audit_path: str | Path,
    dataset_root: str | Path,
    configured_class_names: dict[int, str],
) -> dict[str, Any]:
    path = Path(audit_path)
    if not path.is_file():
        raise ConfigurationError(f"dataset audit is missing: {path}")
    try:
        raw_report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid dataset audit JSON: {path}: {exc}") from exc
    if not isinstance(raw_report, dict):
        raise ConfigurationError(f"dataset audit root must be an object: {path}")
    report: dict[str, Any] = raw_report
    if not report.get("training_ready") or not report.get("structure_and_labels_valid"):
        raise ConfigurationError("dataset audit does not mark the export training-ready")
    root = Path(dataset_root)
    audited_root = Path(str(report.get("dataset_root", "")))
    if root.resolve() != audited_root.resolve():
        raise ConfigurationError(f"dataset root differs from audit: {root} != {audited_root}")
    raw_names = report.get("rfdetr_contiguous_class_names")
    if not isinstance(raw_names, dict):
        raise ConfigurationError("dataset audit has no RF-DETR class mapping")
    try:
        audited_names = {int(key): str(value) for key, value in raw_names.items()}
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("invalid RF-DETR class mapping in dataset audit") from exc
    if sorted(audited_names) != list(range(len(audited_names))):
        raise ConfigurationError("dataset audit class mapping is not contiguous from zero")
    if configured_class_names != audited_names:
        raise ConfigurationError(
            f"config class_names differ from audit: {configured_class_names} != {audited_names}"
        )
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ConfigurationError("dataset audit has no image manifest")
    for record in records:
        if not isinstance(record, dict):
            raise ConfigurationError("invalid image manifest record in dataset audit")
        relative_path = record.get("relative_path")
        expected_sha256 = record.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
            raise ConfigurationError("incomplete image manifest record in dataset audit")
        image_path = root / relative_path
        if not image_path.is_file() or _sha256(image_path) != expected_sha256:
            raise ConfigurationError(f"dataset image changed after audit: {image_path}")
    split_summaries = report.get("splits")
    if not isinstance(split_summaries, dict):
        raise ConfigurationError("dataset audit has no split summaries")
    for split, summary in split_summaries.items():
        if not isinstance(summary, dict):
            raise ConfigurationError(f"invalid split summary for {split}")
        relative_path = summary.get("annotation_path")
        expected_sha256 = summary.get("annotation_sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
            raise ConfigurationError(f"split {split} has no annotation hash")
        annotation_path = root / relative_path
        if not annotation_path.is_file() or _sha256(annotation_path) != expected_sha256:
            raise ConfigurationError(f"annotation file changed after audit: {annotation_path}")
    return report
