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
from fallguard.dataset_normalization import normalize_fallen_person
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


def test_fallen_person_audit_accepts_relative_dataset_root(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    build_dataset(dataset)
    monkeypatch.chdir(tmp_path)
    audit = audit_fallen_person("dataset")
    assert audit["dataset_root"] == str(dataset.resolve())


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


def test_normalization_drops_only_unused_duplicate_and_preserves_source(tmp_path) -> None:
    source = tmp_path / "raw"
    duplicate_categories = [
        {"id": 0, "name": "fallen", "supercategory": "none"},
        {"id": 1, "name": "fallen", "supercategory": "fallen"},
        {"id": 2, "name": "lying", "supercategory": "fallen"},
        {"id": 3, "name": "sitting", "supercategory": "fallen"},
        {"id": 4, "name": "standing", "supercategory": "fallen"},
    ]
    for directory in ("train", "valid", "test"):
        split = source / directory
        split.mkdir(parents=True)
        image = Image.new("RGB", (32, 24), color=(20, 40, 60))
        image.save(split / "sample.png")
        coco = {
            "images": [{"id": 1, "file_name": "sample.png", "width": 32, "height": 24}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [1, 1, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                }
            ],
            "categories": duplicate_categories,
        }
        (split / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")

    source_annotation = (source / "train/_annotations.coco.json").read_bytes()
    output = tmp_path / "processed" / "dataset"
    report = normalize_fallen_person(source, output)
    normalized = json.loads((output / "train/_annotations.coco.json").read_text())

    assert (source / "train/_annotations.coco.json").read_bytes() == source_annotation
    assert report["source_category_usage_all_splits"]["0"] == 0
    assert report["source_category_usage_all_splits"]["1"] == 3
    assert report["old_to_new_category_id"] == {"0": 0, "1": 0, "2": 1, "3": 2, "4": 3}
    assert [category["name"] for category in normalized["categories"]] == [
        "fallen",
        "lying",
        "sitting",
        "standing",
    ]
    assert normalized["annotations"][0]["category_id"] == 0
    assert audit_fallen_person(output)["training_ready"] is True


def test_normalization_rejects_ambiguous_used_duplicate_categories(tmp_path) -> None:
    source = tmp_path / "raw"
    duplicate_categories = [
        {"id": 0, "name": "fallen"},
        {"id": 1, "name": "fallen"},
        {"id": 2, "name": "lying"},
        {"id": 3, "name": "sitting"},
        {"id": 4, "name": "standing"},
    ]
    for split_index, directory in enumerate(("train", "valid", "test")):
        split = source / directory
        split.mkdir(parents=True)
        Image.new("RGB", (32, 24)).save(split / "sample.png")
        coco = {
            "images": [{"id": 1, "file_name": "sample.png", "width": 32, "height": 24}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": split_index % 2,
                    "bbox": [1, 1, 10, 10],
                }
            ],
            "categories": duplicate_categories,
        }
        (split / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="ambiguous"):
        normalize_fallen_person(source, tmp_path / "normalized")
