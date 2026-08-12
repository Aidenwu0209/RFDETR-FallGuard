#!/usr/bin/env python3
"""Combine frontend and semantic reports into an all-clip cascade summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_metrics(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
    tp = sum(row["expected_fall"] and row["final_predicted_fall"] for row in rows)
    fp = sum(not row["expected_fall"] and row["final_predicted_fall"] for row in rows)
    fn = sum(row["expected_fall"] and not row["final_predicted_fall"] for row in rows)
    tn = sum(not row["expected_fall"] and not row["final_predicted_fall"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "clips": len(rows),
        "true_positive_clips": tp,
        "false_positive_clips": fp,
        "false_negative_clips": fn,
        "true_negative_clips": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def summarize(
    frontend_reports: list[tuple[Path, dict[str, Any]]],
    semantic_path: Path,
    semantic_report: dict[str, Any],
) -> dict[str, Any]:
    if not frontend_reports:
        raise ValueError("at least one frontend report is required")
    model_variants = {str(report.get("model_variant")) for _, report in frontend_reports}
    weights_hashes = {str(report.get("weights_sha256")) for _, report in frontend_reports}
    config_hashes = {str(report.get("config_sha256")) for _, report in frontend_reports}
    implementation_hashes = {
        str(report.get("pipeline_implementation_sha256")) for _, report in frontend_reports
    }
    if len(model_variants) != 1 or "None" in model_variants:
        raise ValueError("frontend reports must share one declared model variant")
    if len(weights_hashes) != 1 or "None" in weights_hashes:
        raise ValueError("frontend reports must share one declared weights hash")
    if len(config_hashes) != 1 or "None" in config_hashes:
        raise ValueError("frontend reports must share one declared config hash")
    if len(implementation_hashes) != 1 or "None" in implementation_hashes:
        raise ValueError("frontend reports must share one implementation hash")

    frontend_by_video: dict[str, dict[str, Any]] = {}
    for _, report in frontend_reports:
        for row in report.get("rows", []):
            video_id = str(row["video_id"])
            if video_id in frontend_by_video:
                raise ValueError(f"duplicate frontend video_id: {video_id}")
            frontend_by_video[video_id] = row
    if not frontend_by_video:
        raise ValueError("frontend reports contain no rows")

    semantic_by_video: dict[str, list[dict[str, Any]]] = {}
    candidate_ids: set[str] = set()
    for row in semantic_report.get("rows", []):
        candidate_id = str(row["candidate_id"])
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate semantic candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        video_id = str(row["video_id"])
        if video_id not in frontend_by_video:
            raise ValueError(f"semantic video absent from frontend reports: {video_id}")
        if not frontend_by_video[video_id]["predicted_fall"]:
            raise ValueError(f"semantic review exists for non-candidate clip: {video_id}")
        expected = "fall" if frontend_by_video[video_id]["expected_fall"] else "not_fall"
        if row["expected"] != expected:
            raise ValueError(f"weak label mismatch for {video_id}")
        semantic_by_video.setdefault(video_id, []).append(row)

    expected_events = sum(
        int(row["predicted_event_count"])
        for row in frontend_by_video.values()
        if row["predicted_fall"]
    )
    if len(candidate_ids) != expected_events:
        raise ValueError(
            "semantic candidate count does not match frontend event count: "
            f"{len(candidate_ids)} != {expected_events}"
        )
    rows: list[dict[str, Any]] = []
    for video_id in sorted(frontend_by_video):
        frontend = frontend_by_video[video_id]
        reviews = semantic_by_video.get(video_id, [])
        semantic_fall_events = sum(row["predicted"] == "fall" for row in reviews)
        rows.append(
            {
                "video_id": video_id,
                "subject_id": frontend["subject_id"],
                "expected_fall": frontend["expected_fall"],
                "frontend_candidate": frontend["predicted_fall"],
                "frontend_event_count": frontend["predicted_event_count"],
                "semantic_reviewed_event_count": len(reviews),
                "semantic_fall_event_count": semantic_fall_events,
                "final_predicted_fall": semantic_fall_events > 0,
            }
        )

    reviewed_events = len(candidate_ids)
    semantic_fall_events = sum(
        row["predicted"] == "fall"
        for rows_for_video in semantic_by_video.values()
        for row in rows_for_video
    )
    return {
        "evaluation_kind": "FULL_CASCADE_WEAK_LABEL_SCREEN",
        "formal_ground_truth": False,
        "formal_generalization_claim": False,
        "human_confirmation_required": True,
        "locked_test_subject4_evaluated": False,
        "model_variant": next(iter(model_variants)),
        "metrics_against_weak_clip_labels": binary_metrics(rows),
        "stage_counts": {
            "frontend_input_clips": len(rows),
            "frontend_candidate_clips": sum(row["frontend_candidate"] for row in rows),
            "frontend_candidate_events": expected_events,
            "semantic_reviewed_events": reviewed_events,
            "semantic_fall_events": semantic_fall_events,
            "semantic_rejected_events": reviewed_events - semantic_fall_events,
            "final_alert_clips": sum(row["final_predicted_fall"] for row in rows),
        },
        "semantic_model": semantic_report.get("model"),
        "semantic_model_revision": semantic_report.get("model_revision"),
        "provenance": {
            "weights_sha256": next(iter(weights_hashes)),
            "config_sha256": next(iter(config_hashes)),
            "pipeline_implementation_sha256": next(iter(implementation_hashes)),
            "frontend_reports": [
                {"path": str(path), "sha256": file_sha256(path)} for path, _ in frontend_reports
            ],
            "semantic_report": {
                "path": str(semantic_path),
                "sha256": file_sha256(semantic_path),
            },
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend-report", required=True, action="append", type=Path)
    parser.add_argument("--semantic-report", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    frontend_reports = [
        (path, json.loads(path.read_text(encoding="utf-8"))) for path in args.frontend_report
    ]
    semantic_report = json.loads(args.semantic_report.read_text(encoding="utf-8"))
    summary = summarize(frontend_reports, args.semantic_report, semantic_report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
