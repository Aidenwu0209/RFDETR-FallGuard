# External Blockers

| Item | State | Evidence needed to unblock |
|---|---|---|
| Project license | BLOCKED_EXTERNAL | User must explicitly choose a license before a `LICENSE` file is added |
| Fallen Person COCO export | BLOCKED_EXTERNAL | Sign in to Roboflow in the prepared Chrome tab or securely configure `ROBOFLOW_API_KEY`; never paste credentials into logs/chat |
| Posture-multiclass validation | BLOCKED_EXTERNAL | Run the authorized short Nano/Small fine-tuning after the COCO export and audit its exact class order |
| Final temporal thresholds | BLOCKED_EXTERNAL | Run the grouped cascade with the posture checkpoints; freeze on S1-S2, validate once on S3, and keep S4 locked |
| Detection delay | BLOCKED_EXTERNAL | Human-confirmed fall-onset timestamps; GMDCSA-24 provides clip labels only |
| Local Qwen integration | BLOCKED_EXTERNAL | Explicit model selection/path, compatible dependencies, and enough GPU memory |
| Cloud semantic integration | BLOCKED_EXTERNAL | User-selected provider, API key, paid-test opt-in, and separate image privacy consent |
| Formal QLoRA training | BLOCKED_EXTERNAL | Curated labeled manifest, chosen local model, and explicit authorization for long GPU training |
| Final metrics | BLOCKED_EXTERNAL | Posture checkpoints, frozen thresholds, and non-mock grouped cascade results |

The pinned Supervision ByteTrack deprecation warning is a known compatibility risk, not a current
runtime blocker: the real adapter integration tests pass on `supervision==0.30.0`.
