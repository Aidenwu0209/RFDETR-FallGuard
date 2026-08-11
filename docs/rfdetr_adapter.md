# RF-DETR Adapter

The adapter targets official `rfdetr==1.9.1` and supports Nano and Small. It exposes `load`,
`predict_image`, `predict_frame`, `predict_batch`, `train`, `evaluate`, and `close`, converting
official detections into the strict internal `Detection` schema.

Implicit weight download is disabled. To opt in to official pretrained weights, the caller must
set `allow_weight_download`; otherwise provide an existing `weights_path`.

`person_only` filters to configured person class names and never emits `falling` semantics.
`posture_multiclass` requires a fine-tuned checkpoint and external class metadata.

Training aliases are explicit:

```text
learning_rate -> lr
gradient_accumulation_steps -> grad_accum_steps
gpu_count -> devices
```

Official names such as `lr`, `grad_accum_steps`, `accelerator`, and `devices` pass through.
`gradient_checkpointing` raises `UnsupportedConfigurationError` because the inspected 1.9.1
`TrainConfig` does not expose it.
