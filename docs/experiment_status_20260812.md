# RF-DETR FallGuard experiment status (2026-08-12)

## Current conclusion

The engineering and screening stage is complete on the RTX 5070 Ti host. Do not start
QLoRA yet. The zero-shot multimodal screen is currently strong enough to justify
collecting human-confirmed event labels and completing subject-isolated cross-validation
before adding trainable semantic parameters.

These results are not formal generalization results. All semantic metrics below use weak
clip labels, the sample contains only 12 deterministic screening clips, and locked Subject
4 has not been evaluated.

## Protocol boundary

- The previously opened Figshare Fall29 test split is diagnostic-only and must not be used
  for threshold selection or a final claim.
- GMDCSA-24 Subjects 1-3 form a three-fold, subject-isolated recovery protocol.
- GMDCSA-24 Subject 4 remains the untouched locked test and can be opened only after one
  Small/Nano profile is frozen from all three recovery folds.
- Candidate events and their before/during/after crops require human event-level review.

## Evidence obtained

### Opened-test failure attribution

Of 64 known false-negative clips, 50 reached a Temporal candidate but were not confirmed,
and 14 never entered the suspected state. The production detector produced fallen/lying
detections in 63 clips; the remaining clip also produced fallen/lying detections at the
0.05 diagnostic floor. This is post-hoc stage attribution, not a threshold-selection set.

### High-recall frontend screening

| Model | Partition | Clips | TP | FP | FN | TN | Recall | Specificity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | Subjects 1-2 development | 8 | 4 | 3 | 0 | 1 | 1.00 | 0.25 |
| Small | Subject 3 validation | 4 | 2 | 1 | 0 | 1 | 1.00 | 0.50 |
| Nano | Subjects 1-2 development | 8 | 4 | 3 | 0 | 1 | 1.00 | 0.25 |
| Nano | Subject 3 validation | 4 | 2 | 0 | 0 | 2 | 1.00 | 1.00 |

The high-recall frontend deliberately accepts false positives for semantic rejection.

### Local Qwen3.5-4B screening

- Official model: `Qwen/Qwen3.5-4B`
- Pinned revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Weight shard SHA-256 values:
  - `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61`
  - `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188`
- Input: three chronological person crops, with the detector rerun independently at each
  keyframe so a stale bounding box is not reused.
- Output: strict typed JSON; thinking disabled for deterministic serving behavior.
- Small: 16/16 provider-successful, schema-valid event reviews; mean latency about 13.68 s.
- Nano: 14/14 provider-successful, schema-valid event reviews; mean latency about 12.95 s.
- Both 12-clip full-cascade screens produced TP=6, FP=0, FN=0, TN=6 against weak clip
  labels. This 1.00 score is a screening observation only and must not be reported as final
  accuracy.

## QLoRA decision

Status: `DEFERRED_NOT_JUSTIFIED`.

QLoRA may start only after all of the following gates are met:

1. Human reviewers confirm event-level labels for the generated candidate manifests.
2. Small and Nano complete all-video, three-fold Subjects 1-3 evaluation with no
   subject/video overlap.
3. One threshold profile and one model variant are selected without inspecting Subject 4.
4. Zero-shot errors show a repeatable semantic failure pattern that fine-tuning can address.
5. Training moves to a Linux/WSL CUDA environment with a validated quantization stack;
   the current native-Windows runtime does not provide the required bitsandbytes path.

If zero-shot performance remains adequate after human review and full cross-validation,
retain zero-shot Qwen3.5-4B and omit QLoRA from the implementation rather than adding
unsupported training complexity.

## Next execution order

1. Review and label the Small and Nano candidate manifests.
2. Run Small and Nano on every Subjects 1-3 video for all three folds.
3. Select and freeze the production model and thresholds from only those folds.
4. Create the locked-test confirmation artifact.
5. Run Subject 4 exactly once and report bootstrap confidence intervals, per-stage counts,
   latency, schema validity, and failure cases.
