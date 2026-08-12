# External Blockers

| Item | State | Evidence needed to unblock |
|---|---|---|
| Project license | BLOCKED_EXTERNAL | User must explicitly choose a license before a `LICENSE` file is added |
| External real-world generalization | BLOCKED_EXTERNAL | A separately sourced, untouched elderly/real-scene cohort with reviewed labels. Fall29 confirms the threshold only inside its subject-isolated protocol; locked-test recall is 0.6847 and must not be improved by tuning on that test |
| Detection delay | BLOCKED_EXTERNAL | Human-confirmed fall-onset timestamps; GMDCSA-24 and Fall29 provide clip labels only |
| Human-confirmed semantic event labels | BLOCKED_EXTERNAL | Review the generated Small/Nano before/during/after candidate bundles before formal semantic metrics or QLoRA |
| Cloud semantic integration | BLOCKED_EXTERNAL | User-selected provider, API key, paid-test opt-in, and separate image privacy consent |
| Formal QLoRA training | BLOCKED_EXTERNAL | Curated labeled manifest, chosen local model, and explicit authorization for long GPU training |

The pinned Supervision ByteTrack deprecation warning is a known compatibility risk, not a current
runtime blocker: the real adapter integration tests pass on `supervision==0.30.0`.

Completed inputs are intentionally absent from this blocker list: the Fallen Person archive is
verified and normalized; Nano/Small posture checkpoints were trained and reloaded; pinned official
Qwen3.5-4B completed real local three-image inference; GMDCSA-24 now has a non-leaky recovery
protocol; and Fall29 is retained as opened diagnostic evidence. API keys are optional for the
local cascade and were not used by the Qwen validation.
