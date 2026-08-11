# External Blockers

| Item | State | Evidence needed to unblock |
|---|---|---|
| GitHub commit/push | BLOCKED_EXTERNAL | Explicit user confirmation of publication scope; current origin is empty and no push has occurred |
| Project license | BLOCKED_EXTERNAL | User must explicitly choose a license before a `LICENSE` file is added |
| RF-DETR real inference | BLOCKED_EXTERNAL | Install optional GPU stack and obtain an approved official/fine-tuned weight; do not download weights by default |
| Posture-multiclass validation | BLOCKED_EXTERNAL | Dataset metadata plus trained checkpoint for standing/sitting/bending/lying/falling classes |
| Final temporal thresholds | BLOCKED_EXTERNAL | Leakage-safe validation split and recorded threshold-selection protocol |
| Dataset experiments | BLOCKED_EXTERNAL | Approved UP-Fall/Le2i/self-collected datasets and ground-truth manifests |
| Local Qwen integration | BLOCKED_EXTERNAL | Explicit model selection/path, compatible dependencies, and enough GPU memory |
| Cloud semantic integration | BLOCKED_EXTERNAL | User-selected provider, API key, paid-test opt-in, and separate image privacy consent |
| Formal QLoRA training | BLOCKED_EXTERNAL | Curated labeled manifest, chosen local model, and explicit authorization for long GPU training |
| Final metrics | BLOCKED_EXTERNAL | Real data, real weights, protocol thresholds, and non-mock benchmark run |

The pinned Supervision ByteTrack deprecation warning is a known compatibility risk, not a current
runtime blocker: the real adapter integration tests pass on `supervision==0.30.0`.
