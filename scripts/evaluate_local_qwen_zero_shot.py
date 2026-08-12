#!/usr/bin/env python3
"""Run pinned Local Qwen on three-frame semantic candidate bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fallguard.schemas import FallEvent, ImageRef, SemanticReviewRequest
from fallguard.semantic.providers.local_qwen import LocalQwenProvider


def clip_metrics(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        clip = grouped.setdefault(
            str(row["video_id"]),
            {"expected": row["expected"], "predicted_fall": False},
        )
        if clip["expected"] != row["expected"]:
            raise ValueError("one video has conflicting weak clip labels")
        clip["predicted_fall"] = clip["predicted_fall"] or row["predicted"] == "fall"
    tp = sum(item["expected"] == "fall" and item["predicted_fall"] for item in grouped.values())
    fp = sum(item["expected"] == "not_fall" and item["predicted_fall"] for item in grouped.values())
    fn = sum(item["expected"] == "fall" and not item["predicted_fall"] for item in grouped.values())
    tn = sum(
        item["expected"] == "not_fall" and not item["predicted_fall"] for item in grouped.values()
    )
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "clips": len(grouped),
        "true_positive_clips": tp,
        "false_positive_clips": fp,
        "false_negative_clips": fn,
        "true_negative_clips": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    bundle = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    candidates = bundle.get("candidates", [])
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        parser.error("candidate manifest contains no selected candidates")

    provider = LocalQwenProvider(args.model_path, model_name="Qwen3.5-4B")
    provider.load()
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        image_refs = []
        for role in ("before", "during", "after"):
            raw = (
                candidate["images"][role].get("person_crop")
                or candidate["images"][role]["full_frame"]
            )
            image_refs.append(ImageRef.model_validate(raw))
        event = FallEvent(
            track_id=0,
            source_id=str(candidate["video_id"]),
            session_id="semantic-zero-shot",
            start_frame=int(candidate["images"]["during"]["frame_id"]),
            start_time=float(candidate["event_start_seconds"]),
            transition_reasons=["high-recall temporal frontend candidate"],
            metadata={
                "keyframe_order": "before,during,after",
                "clip_label_hidden_from_model": True,
            },
        )
        text_context = json.dumps(
            {
                "task": (
                    "Determine whether the same person undergoes an accidental fall across "
                    "three chronological person crops: before, during, after. Distinguish "
                    "ordinary sitting, bending, lying down intentionally, and camera artifacts."
                ),
                "image_order": ["before", "during", "after"],
                "temporal_candidate_start_seconds": candidate["event_start_seconds"],
                "semantic_output_language": "English",
            }
        )
        assessment = provider.review(
            SemanticReviewRequest(
                event=event,
                text_context=text_context,
                image_refs=image_refs,
                cloud_image_consent=False,
            )
        )
        row = {
            "candidate_id": candidate["candidate_id"],
            "video_id": candidate["video_id"],
            "subject_id": candidate["subject_id"],
            "expected": candidate["expected_semantic_decision_weak"],
            "predicted": assessment.decision,
            "assessment": assessment.model_dump(mode="json"),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(candidates)}",
                    "candidate_id": candidate["candidate_id"],
                    "predicted": assessment.decision,
                    "latency_ms": assessment.latency_ms,
                }
            ),
            flush=True,
        )
        report = {
            "evaluation_kind": "LOCAL_QWEN_ZERO_SHOT_WEAK_LABEL_SCREEN",
            "formal_ground_truth": False,
            "human_confirmation_required": True,
            "model": "Qwen3.5-4B",
            "model_revision": args.model_revision,
            "input_mode": "three_chronological_person_crops",
            "candidate_count": len(rows),
            "event_level_labels_available": False,
            "metrics_against_weak_clip_labels": clip_metrics(rows),
            "rows": rows,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
