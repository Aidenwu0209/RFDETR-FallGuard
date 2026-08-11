# External Blockers

| Item | State | Evidence needed to unblock |
|---|---|---|
| Project license | BLOCKED_EXTERNAL | User must explicitly choose a license before a `LICENSE` file is added |
| Final temporal thresholds | BLOCKED_EXTERNAL | Nano 0.75 passed the S1-S2 development gate but failed the one-time S3 confirmation gate at recall 0.5. Do not tune on S3 or unlock S4. Improve the posture checkpoint/pipeline using development data, then predeclare a new protocol with a fresh untouched person/group for confirmation |
| Detection delay | BLOCKED_EXTERNAL | Human-confirmed fall-onset timestamps; GMDCSA-24 provides clip labels only |
| Local Qwen integration | BLOCKED_EXTERNAL | Explicit model selection/path, compatible dependencies, and enough GPU memory |
| Cloud semantic integration | BLOCKED_EXTERNAL | User-selected provider, API key, paid-test opt-in, and separate image privacy consent |
| Formal QLoRA training | BLOCKED_EXTERNAL | Curated labeled manifest, chosen local model, and explicit authorization for long GPU training |
| Formal cascade metrics | BLOCKED_EXTERNAL | A confirmed threshold and an untouched locked test partition. The current S3 failure is reported as a negative result, not converted into a formal claim |

The pinned Supervision ByteTrack deprecation warning is a known compatibility risk, not a current
runtime blocker: the real adapter integration tests pass on `supervision==0.30.0`.

Completed inputs are intentionally absent from this blocker list: the Fallen Person archive is
verified and normalized, Nano/Small posture checkpoints were trained and reloaded, and GMDCSA-24
was partitioned by subject. API keys are optional for the local cascade and were not configured or
called during this validation.
