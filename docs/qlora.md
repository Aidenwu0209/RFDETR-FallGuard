# QLoRA Infrastructure

This repository validates the infrastructure but does not perform formal QLoRA training during
engineering QA.

Each JSONL line is a strict `SemanticTrainingSample`: scoped event identity, one or more
`ImageRef` values, text context, a ground-truth-verified `SemanticAssessment`, and a
`split_group`. Validation rejects duplicate sample IDs, missing images, excess image counts, and
any group that crosses train/validation/test.

`split_samples_by_group()` provides deterministic seeded grouping before manifests are written,
so frames/events belonging to the same person, source clip, or collection unit cannot leak across
splits when that unit is used as `split_group`.

```bash
python scripts/train_semantic_adapter.py \
  --config configs/qlora.yaml \
  --allow-external-blockers
```

The default config intentionally has null manifests/model and reports `BLOCKED_EXTERNAL`.
`--execute` is a separate explicit action, requires an existing approved local model path and
the `local-vlm` extra, and never downloads the model implicitly. The execution path constructs
4-bit quantization, LoRA adapters, multimodal collation, Transformers Trainer, and adapter save.
