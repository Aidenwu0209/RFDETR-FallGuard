from __future__ import annotations

import hashlib

import pytest
from PIL import Image

from fallguard.exceptions import ConfigurationError
from fallguard.schemas import ImageRef, SemanticAssessment, SemanticTrainingSample
from fallguard.training.semantic import (
    QLoRAConfig,
    packed_messages,
    split_samples_by_group,
    validate_manifests,
)

pytestmark = pytest.mark.unit


def make_sample(tmp_path, sample_id: str, group: str) -> SemanticTrainingSample:
    image_path = tmp_path / f"{sample_id}.jpg"
    Image.new("RGB", (4, 4), "black").save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    target = SemanticAssessment(
        decision="fall",
        confidence=1,
        reason="fixture ground truth",
        attempt_to_stand=False,
        risk_level="high",
        provider="ground_truth",
        model="human-label",
        input_mode="images_and_text",
        latency_ms=0,
        schema_valid=True,
        provider_success=True,
        model_recommends_alert=True,
        ground_truth_verified=True,
    )
    return SemanticTrainingSample(
        sample_id=sample_id,
        source_id="fixture",
        session_id=group,
        event_id=f"event-{sample_id}",
        image_refs=[
            ImageRef(
                path=image_path,
                sha256=digest,
                width=4,
                height=4,
                kind="person_crop",
            )
        ],
        text_context="synthetic fixture only",
        target=target,
        split_group=group,
    )


def write_manifest(path, sample) -> None:
    path.write_text(sample.model_dump_json() + "\n", encoding="utf-8")


def config(tmp_path, train, validation, test) -> QLoRAConfig:
    return QLoRAConfig(
        schema_version=1,
        model_name_or_path=None,
        output_dir=tmp_path / "out",
        train_manifest=train,
        validation_manifest=validation,
        test_manifest=test,
        max_images=3,
        max_length=128,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj"],
        learning_rate=0.0002,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
    )


def test_synthetic_qlora_manifests_validate_without_training(tmp_path) -> None:
    paths = [tmp_path / name for name in ("train.jsonl", "validation.jsonl", "test.jsonl")]
    samples = [
        make_sample(tmp_path, "train", "person-a"),
        make_sample(tmp_path, "validation", "person-b"),
        make_sample(tmp_path, "test", "person-c"),
    ]
    for path, sample in zip(paths, samples, strict=True):
        write_manifest(path, sample)
    assert validate_manifests(config(tmp_path, *paths)) == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    messages = packed_messages(samples[0])
    assert messages[0]["role"] == "user" and messages[1]["role"] == "assistant"


def test_split_group_leakage_is_rejected(tmp_path) -> None:
    paths = [tmp_path / name for name in ("train.jsonl", "validation.jsonl", "test.jsonl")]
    for index, path in enumerate(paths):
        write_manifest(path, make_sample(tmp_path, str(index), "same-person"))
    with pytest.raises(ConfigurationError, match="leaks across"):
        validate_manifests(config(tmp_path, *paths))


def test_group_split_interface_never_leaks_a_group(tmp_path) -> None:
    samples = [make_sample(tmp_path, str(index), f"person-{index // 2}") for index in range(20)]
    splits = split_samples_by_group(samples, seed=7)
    group_sets = {
        split: {sample.split_group for sample in values} for split, values in splits.items()
    }
    assert group_sets["train"].isdisjoint(group_sets["validation"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["validation"].isdisjoint(group_sets["test"])
    assert sum(len(values) for values in splits.values()) == len(samples)
