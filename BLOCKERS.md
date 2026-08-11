# External Blockers

| Item | State | Evidence needed to unblock |
|---|---|---|
| Project license | BLOCKED_EXTERNAL | User must explicitly choose a license before a `LICENSE` file is added |
| External real-world generalization | BLOCKED_EXTERNAL | A separately sourced, untouched elderly/real-scene cohort with reviewed labels. Fall29 confirms the threshold only inside its subject-isolated protocol; locked-test recall is 0.6847 and must not be improved by tuning on that test |
| Detection delay | BLOCKED_EXTERNAL | Human-confirmed fall-onset timestamps; GMDCSA-24 and Fall29 provide clip labels only |
| Local Qwen integration | BLOCKED_EXTERNAL | Explicit model selection/path, compatible dependencies, and enough GPU memory |
| Cloud semantic integration | BLOCKED_EXTERNAL | User-selected provider, API key, paid-test opt-in, and separate image privacy consent |
| Formal QLoRA training | BLOCKED_EXTERNAL | Curated labeled manifest, chosen local model, and explicit authorization for long GPU training |

The pinned Supervision ByteTrack deprecation warning is a known compatibility risk, not a current
runtime blocker: the real adapter integration tests pass on `supervision==0.30.0`.

Completed inputs are intentionally absent from this blocker list: the Fallen Person archive is
verified and normalized; Nano/Small posture checkpoints were trained and reloaded; GMDCSA-24 is
retained as a negative historical cycle; and Fall29 produced confirmed internal thresholds plus a
complete locked-test report. API keys are optional for the local cascade and were not configured
or called during this validation.
