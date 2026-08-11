"""Lossless, provenance-recorded normalization for external COCO exports."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fallguard.data_audit import EXPECTED_FALLEN_PERSON_CLASSES, ROBOFLOW_SPLITS
from fallguard.exceptions import ConfigurationError

MAX_ROUNDING_CLIP_PIXELS = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _category_schema(coco: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for raw in coco["categories"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), int):
            raise ConfigurationError(f"invalid COCO category in {source}")
        category_id = raw["id"]
        name = raw.get("name")
        if category_id in seen_ids:
            raise ConfigurationError(f"duplicate COCO category id {category_id} in {source}")
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError(f"empty COCO category name in {source}")
        seen_ids.add(category_id)
        schema.append(
            {
                "id": category_id,
                "name": name.strip(),
                "supercategory": raw.get("supercategory"),
            }
        )
    schema.sort(key=lambda item: item["id"])
    return schema


def normalize_fallen_person(
    source_root: str | Path,
    output_root: str | Path,
    *,
    source_archive: str | Path | None = None,
) -> dict[str, Any]:
    """Create an independent four-class RF-DETR dataset without altering the export.

    Roboflow export v1 contains two categories named ``fallen``. The ID 0 entry is
    an unused hierarchy placeholder while ID 1 contains the actual annotations.
    We only collapse duplicate names when exactly one of those IDs is used across
    all splits; otherwise the normalization is ambiguous and fails closed.
    """

    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if not source.is_dir():
        raise ConfigurationError(f"source dataset root does not exist: {source}")
    if output == source or output.is_relative_to(source):
        raise ConfigurationError("normalized output must not be inside the raw source dataset")
    if output.exists():
        raise ConfigurationError(
            f"normalized output already exists; refusing to overwrite: {output}"
        )

    cocos: dict[str, dict[str, Any]] = {}
    annotation_paths: dict[str, Path] = {}
    split_schemas: dict[str, list[dict[str, Any]]] = {}
    usage: Counter[int] = Counter()
    for logical_split, directory_name in ROBOFLOW_SPLITS.items():
        annotation_path = source / directory_name / "_annotations.coco.json"
        coco = _read_coco(annotation_path)
        schema = _category_schema(coco, annotation_path)
        valid_ids = {category["id"] for category in schema}
        for annotation in coco["annotations"]:
            if not isinstance(annotation, dict) or not isinstance(
                annotation.get("category_id"), int
            ):
                raise ConfigurationError(f"invalid COCO annotation in {annotation_path}")
            category_id = annotation["category_id"]
            if category_id not in valid_ids:
                raise ConfigurationError(
                    f"annotation references unknown category id {category_id}: {annotation_path}"
                )
            usage[category_id] += 1
        cocos[logical_split] = coco
        annotation_paths[logical_split] = annotation_path
        split_schemas[logical_split] = schema

    reference_schema = split_schemas["train"]
    for split, schema in split_schemas.items():
        if schema != reference_schema:
            raise ConfigurationError(f"category schema differs in {split} split")

    by_name: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for category in reference_schema:
        by_name[category["name"].lower()].append(category)
    if set(by_name) != EXPECTED_FALLEN_PERSON_CLASSES:
        raise ConfigurationError(
            f"unexpected Fallen Person classes: {sorted(by_name)}; "
            f"expected {sorted(EXPECTED_FALLEN_PERSON_CLASSES)}"
        )

    keepers: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for normalized_name, categories in by_name.items():
        used = [category for category in categories if usage[category["id"]] > 0]
        if len(categories) > 1 and len(used) != 1:
            raise ConfigurationError(
                f"duplicate category name {normalized_name!r} is ambiguous: "
                f"used IDs {[item['id'] for item in used]}"
            )
        keeper = used[0] if len(categories) > 1 else categories[0]
        keepers.append(keeper)
        dropped.extend(category for category in categories if category["id"] != keeper["id"])

    keepers.sort(key=lambda item: item["id"])
    old_to_new: dict[int, int] = {}
    normalized_categories: list[dict[str, Any]] = []
    for new_id, keeper in enumerate(keepers):
        normalized_categories.append(
            {"id": new_id, "name": keeper["name"], "supercategory": "none"}
        )
        for same_name in by_name[keeper["name"].lower()]:
            old_to_new[same_name["id"]] = new_id

    archive_path = Path(source_archive).resolve() if source_archive is not None else None
    if archive_path is not None and not archive_path.is_file():
        raise ConfigurationError(f"source archive does not exist: {archive_path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    split_reports: dict[str, dict[str, Any]] = {}
    try:
        for logical_split, directory_name in ROBOFLOW_SPLITS.items():
            source_split = source / directory_name
            target_split = temporary / directory_name
            target_split.mkdir(parents=True)
            coco = cocos[logical_split]
            copied_paths: set[str] = set()
            for image in coco["images"]:
                if not isinstance(image, dict) or not isinstance(image.get("file_name"), str):
                    raise ConfigurationError(
                        f"invalid image record in {annotation_paths[logical_split]}"
                    )
                relative_name = image["file_name"]
                source_image = (source_split / relative_name).resolve()
                target_image = (target_split / relative_name).resolve()
                if not source_image.is_relative_to(source_split.resolve()):
                    raise ConfigurationError(f"image path escapes source split: {relative_name}")
                if not target_image.is_relative_to(target_split.resolve()):
                    raise ConfigurationError(
                        f"image path escapes normalized split: {relative_name}"
                    )
                if not source_image.is_file():
                    raise ConfigurationError(f"referenced image is missing: {source_image}")
                if relative_name not in copied_paths:
                    target_image.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_image, target_image)
                    copied_paths.add(relative_name)

            normalized_coco = copy.deepcopy(coco)
            normalized_coco["categories"] = normalized_categories
            images_by_id = {
                image["id"]: image
                for image in normalized_coco["images"]
                if isinstance(image, dict) and isinstance(image.get("id"), int)
            }
            bbox_adjustments: list[dict[str, Any]] = []
            maximum_bbox_clip = 0.0
            for annotation in normalized_coco["annotations"]:
                annotation["category_id"] = old_to_new[annotation["category_id"]]
                annotation_id = annotation.get("id")
                image = images_by_id.get(annotation.get("image_id"))
                bbox = annotation.get("bbox")
                if (
                    image is None
                    or not isinstance(image.get("width"), int)
                    or not isinstance(image.get("height"), int)
                    or not isinstance(bbox, list)
                    or len(bbox) != 4
                    or any(not isinstance(value, int | float) for value in bbox)
                    or any(not math.isfinite(float(value)) for value in bbox)
                ):
                    raise ConfigurationError(
                        f"annotation {annotation_id} has invalid image metadata or bbox"
                    )
                x, y, width, height = (float(value) for value in bbox)
                if width <= 0 or height <= 0:
                    raise ConfigurationError(f"annotation {annotation_id} has non-positive bbox")
                x2 = x + width
                y2 = y + height
                image_width = float(image["width"])
                image_height = float(image["height"])
                clipped_x1 = min(max(x, 0.0), image_width)
                clipped_y1 = min(max(y, 0.0), image_height)
                clipped_x2 = min(max(x2, 0.0), image_width)
                clipped_y2 = min(max(y2, 0.0), image_height)
                clip_amount = max(
                    abs(clipped_x1 - x),
                    abs(clipped_y1 - y),
                    abs(clipped_x2 - x2),
                    abs(clipped_y2 - y2),
                )
                if clip_amount > MAX_ROUNDING_CLIP_PIXELS + 1e-9:
                    raise ConfigurationError(
                        f"annotation {annotation_id} exceeds the allowed rounding clip: "
                        f"{clip_amount:.6f} pixels"
                    )
                clipped_width = clipped_x2 - clipped_x1
                clipped_height = clipped_y2 - clipped_y1
                if clipped_width <= 0 or clipped_height <= 0:
                    raise ConfigurationError(
                        f"annotation {annotation_id} becomes empty after boundary clipping"
                    )
                if clip_amount > 0:
                    original_bbox = list(annotation["bbox"])
                    annotation["bbox"] = [
                        clipped_x1,
                        clipped_y1,
                        clipped_width,
                        clipped_height,
                    ]
                    if "area" in annotation:
                        annotation["area"] = clipped_width * clipped_height
                    maximum_bbox_clip = max(maximum_bbox_clip, clip_amount)
                    if len(bbox_adjustments) < 20:
                        bbox_adjustments.append(
                            {
                                "annotation_id": annotation_id,
                                "original_bbox": original_bbox,
                                "normalized_bbox": annotation["bbox"],
                                "maximum_clip_pixels": clip_amount,
                            }
                        )
            target_annotation = target_split / "_annotations.coco.json"
            target_annotation.write_text(
                json.dumps(normalized_coco, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            split_reports[logical_split] = {
                "directory": directory_name,
                "images": len(coco["images"]),
                "annotations": len(coco["annotations"]),
                "source_annotation_sha256": _sha256(annotation_paths[logical_split]),
                "normalized_annotation_sha256": _sha256(target_annotation),
                "source_category_usage": {
                    str(category["id"]): sum(
                        int(annotation["category_id"] == category["id"])
                        for annotation in coco["annotations"]
                    )
                    for category in reference_schema
                },
                "bbox_rounding_adjustments": sum(
                    1
                    for original, normalized in zip(
                        coco["annotations"], normalized_coco["annotations"], strict=True
                    )
                    if original.get("bbox") != normalized.get("bbox")
                ),
                "maximum_bbox_clip_pixels": maximum_bbox_clip,
                "bbox_rounding_adjustment_examples": bbox_adjustments,
            }

        for name in ("README.dataset.txt", "README.roboflow.txt"):
            source_readme = source / name
            if source_readme.is_file():
                shutil.copy2(source_readme, temporary / name)

        report: dict[str, Any] = {
            "normalization_kind": "FALLEN_PERSON_RFDETR_FLAT_V1",
            "source_dataset_root": str(source),
            "normalized_dataset_root": str(output),
            "source_archive": (
                {
                    "path": str(archive_path),
                    "bytes": archive_path.stat().st_size,
                    "sha256": _sha256(archive_path),
                }
                if archive_path is not None
                else None
            ),
            "source_categories": reference_schema,
            "source_category_usage_all_splits": {
                str(category["id"]): usage[category["id"]] for category in reference_schema
            },
            "old_to_new_category_id": {
                str(old_id): new_id for old_id, new_id in sorted(old_to_new.items())
            },
            "dropped_unused_duplicate_categories": [
                {**category, "annotations": usage[category["id"]]} for category in dropped
            ],
            "normalized_categories": normalized_categories,
            "supercategory_policy": "flattened to 'none' to prevent hierarchy inference",
            "bbox_policy": (
                "clip only export-rounding overflow at image boundaries; "
                f"maximum allowed correction is {MAX_ROUNDING_CLIP_PIXELS} pixels"
            ),
            "image_materialization": "independent byte copies; raw source is not modified",
            "splits": split_reports,
        }
        (temporary / "NORMALIZATION.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return report
