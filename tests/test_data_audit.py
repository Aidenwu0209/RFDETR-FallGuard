from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from fallguard.data_audit import (
    audit_fallen_person,
    posture_profile,
    validate_training_audit,
)
from fallguard.exceptions import ConfigurationError

pytestmark = pytest.mark.unit

CATEGORIES = [
    {"id": 7, "name": "fallen"},
    {"id": 3, "name": "standing"},
    {"id": 11, "name": "lying"},
    {"id": 9, "name": "sitting"},
]


def write_split(
    root: Path,
    directory: str,
    *,
    wrong_width: bool = False,
    escaped_path: bool = False,
) -> None:
    split = root / directory
    split.mkdir(parents=True)
    images = []
    annotations = []
    for index, category in enumerate(CATEGORIES, start=1):
        filename = "../outside.png" if escaped_path and index == 1 else f"{directory}-{index}.png"
        image = Image.new("RGB", (32, 24), color=(index * 30, index * 20, index * 10))
        draw = ImageDraw.Draw(image)
        draw.rectangle((index, index, 10 + index, 15 + index), fill=(255, 255, 255))
        image_path = split / filename
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        images.append(
            {
                "id": index,
                "file_name": filename,
                "width": 31 if wrong_width and index == 1 else 32,
                "height": 24,
            }
        )
        annotations.append(
            {
                "id": index,
                "image_id": index,
                "category_id": category["id"],
                "bbox": [1, 1, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        )
    (split / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": CATEGORIES}),
        encoding="utf-8",
    )


def build_dataset(root: Path, *, wrong_width: bool = False) -> None:
    write_split(root, "train", wrong_width=wrong_width)
    write_split(root, "valid")
    write_split(root, "test")


def test_fallen_person_audit_uses_sorted_coco_ids_and_emits_profile(tmp_path) -> None:
    build_dataset(tmp_path)
    audit = audit_fallen_person(tmp_path)
    assert audit["images_total"] == 12
    assert audit["rfdetr_contiguous_class_names"] == {
        0: "standing",
        1: "fallen",
        2: "sitting",
        3: "lying",
    }
    assert audit["group_isolation"]["available"] is False
    profile = posture_profile(audit)
    assert profile["detector"]["posture_groups"]["fall"] == ["fallen"]
    assert "warning" not in profile["detector"]["posture_groups"]


def test_fallen_person_audit_rejects_metadata_dimension_mismatch(tmp_path) -> None:
    build_dataset(tmp_path, wrong_width=True)
    with pytest.raises(ConfigurationError, match="dimensions disagree"):
        audit_fallen_person(tmp_path)


def test_fallen_person_audit_rejects_paths_outside_split(tmp_path) -> None:
    write_split(tmp_path, "train", escaped_path=True)
    write_split(tmp_path, "valid")
    write_split(tmp_path, "test")
    with pytest.raises(ConfigurationError, match="escapes its split"):
        audit_fallen_person(tmp_path)


def test_training_gate_detects_post_audit_image_changes(tmp_path) -> None:
    build_dataset(tmp_path)
    audit = audit_fallen_person(tmp_path)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    class_names = audit["rfdetr_contiguous_class_names"]
    assert validate_training_audit(audit_path, tmp_path, class_names)["training_ready"] is True
    changed = tmp_path / audit["records"][0]["relative_path"]
    changed.write_bytes(changed.read_bytes() + b"changed")
    with pytest.raises(ConfigurationError, match="changed after audit"):
        validate_training_audit(audit_path, tmp_path, class_names)
