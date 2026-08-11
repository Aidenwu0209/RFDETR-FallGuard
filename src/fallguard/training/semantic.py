"""QLoRA manifest validation, multimodal packing, and explicit execution entry."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fallguard.exceptions import ConfigurationError, DependencyUnavailableError
from fallguard.schemas import SemanticTrainingSample


class QLoRAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    model_name_or_path: Path | None
    output_dir: Path
    train_manifest: Path | None
    validation_manifest: Path | None
    test_manifest: Path | None
    seed: int = 42
    max_images: int = Field(gt=0)
    max_length: int = Field(gt=0)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    lora_r: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    lora_dropout: float = Field(ge=0.0, lt=1.0)
    target_modules: list[str] = Field(min_length=1)
    learning_rate: float = Field(gt=0.0)
    num_train_epochs: float = Field(gt=0.0)
    per_device_train_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)


def load_qlora_config(path: str | Path) -> QLoRAConfig:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigurationError("QLoRA configuration must be a YAML mapping")
    try:
        return QLoRAConfig.model_validate(value)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid QLoRA configuration: {exc}") from exc


def read_manifest(path: str | Path) -> list[SemanticTrainingSample]:
    records: list[SemanticTrainingSample] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(SemanticTrainingSample.model_validate_json(line))
            except ValidationError as exc:
                raise ConfigurationError(
                    f"invalid manifest line {line_number} in {path}: {exc}"
                ) from exc
    return records


def validate_manifests(config: QLoRAConfig) -> dict[str, int]:
    paths = {
        "train": config.train_manifest,
        "validation": config.validation_manifest,
        "test": config.test_manifest,
    }
    if any(path is None for path in paths.values()):
        raise ConfigurationError("train, validation, and test manifests are all required")
    split_samples = {name: read_manifest(path) for name, path in paths.items() if path is not None}
    ids: set[str] = set()
    groups: dict[str, str] = {}
    for split, samples in split_samples.items():
        for sample in samples:
            if sample.sample_id in ids:
                raise ConfigurationError(f"duplicate semantic sample_id: {sample.sample_id}")
            ids.add(sample.sample_id)
            previous_split = groups.setdefault(sample.split_group, split)
            if previous_split != split:
                raise ConfigurationError(
                    f"split_group {sample.split_group} leaks across {previous_split} and {split}"
                )
            if len(sample.image_refs) > config.max_images:
                raise ConfigurationError(
                    f"sample {sample.sample_id} exceeds configured max_images={config.max_images}"
                )
            for image in sample.image_refs:
                if not image.path.is_file():
                    raise ConfigurationError(
                        f"sample {sample.sample_id} image does not exist: {image.path}"
                    )
    return {name: len(samples) for name, samples in split_samples.items()}


def split_samples_by_group(
    samples: list[SemanticTrainingSample],
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[SemanticTrainingSample]]:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ConfigurationError("split ratios must be non-negative and sum to one")
    groups: dict[str, list[SemanticTrainingSample]] = {}
    for sample in samples:
        groups.setdefault(sample.split_group, []).append(sample)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    train_end = round(len(names) * train_ratio)
    validation_end = train_end + round(len(names) * validation_ratio)
    assignments = {
        "train": names[:train_end],
        "validation": names[train_end:validation_end],
        "test": names[validation_end:],
    }
    return {
        split: [sample for group in selected for sample in groups[group]]
        for split, selected in assignments.items()
    }


def packed_messages(sample: SemanticTrainingSample) -> list[dict[str, Any]]:
    user_content: list[dict[str, str]] = [
        {"type": "image", "image": str(image.path)} for image in sample.image_refs
    ]
    user_content.append({"type": "text", "text": sample.text_context})
    target = sample.target.model_dump(
        include={
            "decision",
            "confidence",
            "reason",
            "attempt_to_stand",
            "risk_level",
            "model_recommends_alert",
        }
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]


def execute_qlora(config: QLoRAConfig) -> None:
    """Execute only after an explicit CLI flag; callers normally use validation dry-run."""
    validate_manifests(config)
    if config.model_name_or_path is None:
        raise ConfigurationError("model_name_or_path is required for execution")
    if not config.model_name_or_path.exists():
        raise ConfigurationError("model_name_or_path must be an existing approved local path")
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise DependencyUnavailableError("install .[local-vlm] before QLoRA execution") from exc

    processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        config.model_name_or_path, local_files_only=True
    )
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    quantization = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=quantization,
        device_map="auto",
        local_files_only=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            task_type="CAUSAL_LM",
        ),
    )
    assert config.train_manifest is not None
    assert config.validation_manifest is not None
    train_samples = read_manifest(config.train_manifest)
    validation_samples = read_manifest(config.validation_manifest)
    collator = _MultimodalCollator(processor, config.max_length)
    arguments = TrainingArguments(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        remove_unused_columns=False,
        report_to=[],
        seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=_SemanticDataset(train_samples),
        eval_dataset=_SemanticDataset(validation_samples),
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(str(config.output_dir))
    processor.save_pretrained(str(config.output_dir))


class _SemanticDataset:
    def __init__(self, samples: list[SemanticTrainingSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SemanticTrainingSample:
        return self.samples[index]


class _MultimodalCollator:
    def __init__(self, processor: Any, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, samples: list[SemanticTrainingSample]) -> dict[str, Any]:
        from PIL import Image

        texts: list[str] = []
        images: list[list[Any]] = []
        opened: list[Any] = []
        try:
            for sample in samples:
                texts.append(
                    self.processor.apply_chat_template(
                        packed_messages(sample),
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
                sample_images = [Image.open(item.path).convert("RGB") for item in sample.image_refs]
                images.append(sample_images)
                opened.extend(sample_images)
            batch: dict[str, Any] = dict(
                self.processor(
                    text=texts,
                    images=images,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
            )
            batch["labels"] = batch["input_ids"].clone()
            return batch
        finally:
            for image in opened:
                image.close()
