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
| M2 posture-multiclass training/evaluation | BLOCKED_EXTERNAL | Data audit/hash gate and short-run CLI are ready. One-epoch synthetic Nano and Small GPU runs produced reloadable four-class checkpoints, proving mechanics only; Roboflow login/key is still required for real-data training |
| M3 pinned ByteTrack adapter and TrackManager | VERIFIED_INTEGRATION | Real `supervision==0.30.0` tests pass for posture-class changes and one-frame disappearance/reappearance with stable ID; scope, history, expiry, and empty-frame advancement tested |
| M4 temporal state machine, events, keyframes | VERIFIED_INTEGRATION | Seconds/coordinates/smoothing, auditable transitions, timeout, atomic event close/reopen, bounded buffer, random-access video reread, and before/during/after persistence tested |
| M5 Router, Mock Provider, privacy, AlertManager | VERIFIED_INTEGRATION | Mock end-to-end route passes; cloud image double-consent, fallback reason, unexpected-error propagation, structured schema, redaction, and application-owned alert tested |
| M5 OpenAI, DeepSeek, Local Qwen provider implementations | CODE_COMPLETE_UNVALIDATED | OpenAI 2.53.0 structured-parse SDK contract is present; no paid API call or Local Qwen model load was authorized |
| M5 QLoRA schema, grouped split, packing, validation, dry-run, execution entry | VERIFIED_UNIT | Synthetic manifests validate, cross-split leakage is rejected, dry-run reports missing real manifests as `BLOCKED_EXTERNAL`; no training ran |
| M6 detection/event/deployment evaluation and experiment recorder | VERIFIED_INTEGRATION | Official detection delegation, one-to-one event metrics, unavailable-metric reasons, mock formal rejection, warm-up reset, resource timing, and reproducibility snapshot tested |
| M6 GMDCSA-24 grouped validation preparation | VERIFIED_INTEGRATION | Zenodo v2.0 archive MD5 verified; 160 readable videos, 4 subjects, 79 ADL/81 Fall, no duplicate hashes or subject leakage; S1-S2 development, S3 validation, S4 locked test; 16-video subset generated |
| M6 real cascade engineering smoke | VERIFIED_INTEGRATION | Official Small `person_only` completed RF-DETR -> ByteTrack -> Temporal -> Event on one Fall and one ADL clip; both produced one unique candidate, demonstrating execution and also why person-only output is invalid for final thresholds |
| M6 formal real-data threshold result | BLOCKED_EXTERNAL | Grouped full-cascade runner is implemented, but a posture checkpoint is still required; clip labels cannot support detection-delay claims without human onset timestamps |
| M7 Gradio, docs, CLI and full QA | VERIFIED_INTEGRATION | Five-tab UI rendered in a real browser with model/GPU reports, dataset protocol, training state, and local-only key status; non-blocking HTTP 200 probe passes |
| Continuous live camera streaming | NOT_IMPLEMENTED | UI intentionally supports uploaded/webcam-recorded finite clips and does not claim continuous real-time monitoring |
| GitHub publication | VERIFIED_INTEGRATION | Commits `63f999e` and `5055b93` are on `main`; data, weights, artifacts, and secrets remain ignored |

## Baseline QA evidence

```text
.venv/bin/ruff format .
  exit 0; 93 files unchanged on the latest implementation run
.venv/bin/ruff check .
  exit 0; all checks passed
.venv/bin/mypy src app scripts
  exit 0; success, 61 source files after training extras were installed
.venv/bin/pytest -q
  exit 0; 59 passed, 2 deselected, 0 failed, 0 skipped, 1 warning
.venv/bin/python -m compileall -q src app scripts tests datasets
  exit 0
.venv/bin/python scripts/check_environment.py
  exit 0; local-only audit, network_or_paid_call_performed=false
.venv/bin/python scripts/smoke_gradio.py
  exit 0; HTTP 200 and server closed
15 required scripts with --help
  exit 0
```

The two deselected tests carry the `api` marker and additionally require
`RUN_PAID_API_INTEGRATION_TESTS=1`. The warning comes from the pinned real Supervision adapter:
its `ByteTrack` API is deprecated in 0.30.0 and announced for removal in 0.31.0; the exact pin is
therefore reproducible today but requires a deliberate future migration.

## Evidence boundaries

- No mock result was written into a formal benchmark. Engineering timing is labeled
  `MOCK_ENGINEERING_ONLY` and `formal_benchmark_eligible=false`.
- No paid API call, long training, image upload, or privacy-data transfer was performed. The user
  explicitly authorized official model and dataset downloads plus short detector fine-tuning.
- Official weights, dataset archives, manifests, reports, and future checkpoints remain outside Git.
- No AP, event metric, FPS, model quality, or VRAM result from synthetic/mock work is presented as
  scientific or deployment evidence. The synthetic Nano/Small run proves only that training,
  checkpoint serialization, official reload, and GPU inference execute.
- The generated validation artifacts live only under `/tmp`; an accidentally generated root-level
  mock artifact set was identified and removed before final QA.
