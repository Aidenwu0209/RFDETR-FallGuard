# Implementation Status

Evidence date: 2026-08-11. Runtime evidence was collected on
`/home/aidenwu/Documents/RFDETR-FallGuard` on the Legion host. Only these states are used:
`VERIFIED_UNIT`, `VERIFIED_INTEGRATION`, `CODE_COMPLETE_UNVALIDATED`, `BLOCKED_EXTERNAL`, and
`NOT_IMPLEMENTED`.

| Milestone / surface | State | Evidence / next gate |
|---|---|---|
| M0 repository audit and baseline | VERIFIED_INTEGRATION | Public origin and both clones were empty; initial `pytest` failed because no project/environment existed; Python 3.12.3 and RTX 4060 driver visibility recorded |
| M1 Config, Schema, exceptions, logging, device, video, timing | VERIFIED_INTEGRATION | Strict-config/schema, secret redaction, video read/seek, session isolation, environment CLI, and deterministic vertical slice tests pass |
| M1 Mock vertical slice | VERIFIED_INTEGRATION | Exit 0; explicitly `MOCK`, 1 event, 1 alert, 3 transitions, before/during/after person crops; formal eligibility false |
| M2 RF-DETR adapter, conversion, image/frame/batch, training/evaluation mapping, CLIs | VERIFIED_UNIT | Fake official-output contract tests pass; real fine-tuned checkpoints use official `from_checkpoint`, enforce contiguous class IDs, and reject a Nano/Small architecture mismatch; unsupported `gradient_checkpointing` is rejected |
| M2 real RF-DETR inference | VERIFIED_INTEGRATION | Official Nano/Small weights passed official MD5 checks and separate RTX 4060 CUDA inference on the same public image; each returned four `person` detections at 0.5; reports explicitly remain single-image smoke evidence |
| M2 posture-multiclass training/evaluation | VERIFIED_INTEGRATION | The supplied Fallen Person archive was SHA-256 verified, normalized without modifying the raw copy, and used for authorized two-epoch Nano/Small CUDA fine-tuning. Both checkpoints reload independently. Engineering-only test mAP50:95 is 0.6940 Nano and 0.7071 Small; the source split has cross-split near duplicates and is not formal generalization evidence |
| M3 pinned ByteTrack adapter and TrackManager | VERIFIED_INTEGRATION | Real `supervision==0.30.0` tests pass for posture-class changes and one-frame disappearance/reappearance with stable ID; scope, history, expiry, and empty-frame advancement tested |
| M4 temporal state machine, events, keyframes | VERIFIED_INTEGRATION | A candidate now starts only from downward motion or the configured fall class; static lying/aspect ratio is supporting evidence only. Event records are promoted only after FALLING/LYING, so duration thresholds affect event presence. Seconds/coordinates/smoothing, timeout, atomic close/reopen, bounded buffer, random-access reread, and keyframes are tested |
| M5 Router, Mock Provider, privacy, AlertManager | VERIFIED_INTEGRATION | Mock end-to-end route passes; cloud image double-consent, fallback reason, unexpected-error propagation, structured schema, redaction, and application-owned alert tested |
| M5 OpenAI, DeepSeek, Local Qwen provider implementations | CODE_COMPLETE_UNVALIDATED | OpenAI 2.53.0 structured-parse SDK contract is present; no paid API call or Local Qwen model load was authorized |
| M5 QLoRA schema, grouped split, packing, validation, dry-run, execution entry | VERIFIED_UNIT | Synthetic manifests validate, cross-split leakage is rejected, dry-run reports missing real manifests as `BLOCKED_EXTERNAL`; no training ran |
| M6 detection/event/deployment evaluation and experiment recorder | VERIFIED_INTEGRATION | Official detection delegation, one-to-one event metrics, unavailable-metric reasons, mock formal rejection, warm-up reset, resource timing, and reproducibility snapshot tested |
| M6 GMDCSA-24 grouped validation preparation | VERIFIED_INTEGRATION | Zenodo v2.0 archive MD5 verified; 160 readable videos, 4 subjects, 79 ADL/81 Fall, no duplicate hashes or subject leakage; S1-S2 development, S3 validation, S4 locked test; 16-video subset generated |
| M6 real cascade engineering smoke | VERIFIED_INTEGRATION | Official Small `person_only` completed RF-DETR -> ByteTrack -> Temporal -> Event on one Fall and one ADL clip; both produced one unique candidate, demonstrating execution and also why person-only output is invalid for final thresholds |
| M6 historical GMDCSA threshold cycle | VERIFIED_INTEGRATION | Eight bounded S1-S2 candidates were evaluated with matching evidence. Nano confidence/tracker activation 0.75 passed development but failed the one-time S3 gate at recall 0.5. No GMDCSA threshold was confirmed; S4 remains unused and this negative result is immutable historical evidence |
| M6 fresh Figshare-Fall29 protocol | VERIFIED_INTEGRATION | Official CC BY 4.0 archive bytes/MD5 and extraction were verified. Two auxiliary timelapses and one exact duplicate were excluded only from the derived manifest, leaving 2,013 unique subject videos. Development, validation, and locked-test groups contain disjoint subjects; the declaration predates model results |
| M6 Fall29 threshold freeze and validation | VERIFIED_INTEGRATION | Twelve Nano/Small candidates used the same 48 development clips. Small confidence/tracker activation 0.40 achieved 24 TP, 0 FP, 0 FN, 24 TN and was frozen. Its one-time 20-clip validation achieved 9 TP, 0 FP, 1 FN, 10 TN (recall 0.90, F1 0.9474), exactly passing the predeclared gate without retuning |
| M6 full locked cascade test | VERIFIED_INTEGRATION | The confirmed Small profile evaluated all 374 locked clips from subjects 13/14/20/22/29: 139 TP, 19 FP, 64 FN, 152 TN; precision 0.8797, recall 0.6847, specificity 0.8889, F1 0.7701. Independent recomputation matched; IDs, checkpoint, parameters, manifest, commit, and implementation fingerprint were bound. This confirms an internal threshold, not external generalization |
| M7 Gradio, docs, CLI and full QA | VERIFIED_INTEGRATION | Five-tab UI rendered in a real browser with model/GPU reports, dataset protocol, training state, and local-only key status; non-blocking HTTP 200 probe passes |
| Continuous live camera streaming | NOT_IMPLEMENTED | UI intentionally supports uploaded/webcam-recorded finite clips and does not claim continuous real-time monitoring |
| GitHub publication | VERIFIED_INTEGRATION | The implementation and evidence-bound protocol are published on `main`; data, weights, artifacts, videos, environment files, and secrets remain ignored |

## Baseline QA evidence

```text
.venv/bin/ruff format .
  exit 0; 104 files unchanged on the latest implementation run
.venv/bin/ruff check .
  exit 0; all checks passed
.venv/bin/mypy src app scripts
  exit 0; success, 70 source files
.venv/bin/pytest -q
  exit 0; 85 passed, 2 deselected, 0 failed, 0 skipped, 1 warning
.venv/bin/python -m compileall -q src app scripts tests datasets
  exit 0
.venv/bin/python scripts/check_environment.py
  exit 0; local-only audit, network_or_paid_call_performed=false
.venv/bin/python scripts/smoke_gradio.py
  exit 0; HTTP 200 and server closed
21 required scripts with --help
  exit 0
```

The two deselected tests carry the `api` marker and additionally require
`RUN_PAID_API_INTEGRATION_TESTS=1`. The warning comes from the pinned real Supervision adapter:
its `ByteTrack` API is deprecated in 0.30.0 and announced for removal in 0.31.0; the exact pin is
therefore reproducible today but requires a deliberate future migration.

## Evidence boundaries

- No mock result was written into a formal benchmark. Engineering timing is labeled
  `MOCK_ENGINEERING_ONLY` and `formal_benchmark_eligible=false`.
- No paid API call, image upload, or privacy-data transfer was performed. The user explicitly
  authorized official model and dataset downloads plus short two-epoch detector fine-tuning.
- Official weights, dataset archives, manifests, reports, and future checkpoints remain outside Git.
- Fallen Person detector metrics are engineering-only because the export has 21,575 cross-split
  dHash near-duplicate pairs and no person/video group identifiers. They are not presented as
  scientific generalization evidence.
- The historical GMDCSA S3 result remains a valid negative result and was not reused for tuning.
- Fall29 confirms Small confidence/tracker activation 0.40 within its predeclared internal
  subject-isolated protocol. Locked-test recall fell to 0.6847, so the result does not establish
  real-world or elderly-population generalization and no detection-delay metric is reported.
- The ignored locked-test report SHA-256 is
  `e2cd46d08c9b7feecba72c5b42e3209bc9de0a4b53311d8bbb1f9354e2e94c04`; it binds manifest
  `29cfa569468a27e7dafc6b4d2c4a0175bf776c10a8d097ff4c474aa8affa5a42` and Small checkpoint
  `7c843e2570a077317c95e3458d50a402f36f6ac8ebcd8bffe7fa8da28f898e71`.
- The 1.9 GiB synthetic dataset and checkpoints were removed from their exact `/tmp` path after
  recording an ignored mechanics-only evidence report; no synthetic asset entered Git.
