#!/usr/bin/env python3
"""Verify, extract, audit, and subject-split the public Figshare Fall29 dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import yaml

EXPECTED_ARCHIVE_BYTES = 2_529_520_868
EXPECTED_ARCHIVE_MD5 = "c784167d08f2fa94e3afd36cec758e1f"
EXPECTED_SUBJECTS = set(range(1, 30))
SUBJECT_PATTERN = re.compile(r"SBJ_(\d+)_LOC(\d+)")
DEVELOPMENT_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 11, 12, 15, 16, 17, 19, 23, 24, 25, 26]
VALIDATION_SUBJECTS = [8, 9, 10, 18, 21, 27, 28]
LOCKED_TEST_SUBJECTS = [13, 14, 20, 22, 29]
SUBJECT_PARTITIONS = {
    **{subject: "threshold_development" for subject in DEVELOPMENT_SUBJECTS},
    **{subject: "threshold_validation" for subject in VALIDATION_SUBJECTS},
    **{subject: "locked_test" for subject in LOCKED_TEST_SUBJECTS},
}


def digest(path: Path, algorithm: str = "md5") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def safe_extract(archive: Path, target: Path) -> None:
    target_resolved = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        members = []
        for member in bundle.infolist():
            resolved = (target / member.filename).resolve()
            if target_resolved not in resolved.parents and resolved != target_resolved:
                raise ValueError(f"unsafe archive member: {member.filename}")
            if member.filename.startswith("__MACOSX/"):
                continue
            members.append(member)
        bundle.extractall(target, members)


def video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid video metadata: {path}")
    return {
        "fps": fps,
        "frames": frames,
        "duration_seconds": frames / fps,
        "width": width,
        "height": height,
    }


def find_dataset_root(extracted: Path) -> Path:
    candidates = [
        path
        for path in extracted.rglob("VideoDataset")
        if path.is_dir() and (path / "ADL").is_dir() and (path / "Fall").is_dir()
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one VideoDataset root, found {len(candidates)}")
    return candidates[0]


def _subject_and_location(relative_path: Path) -> tuple[int, int] | None:
    for part in relative_path.parts:
        match = SUBJECT_PATTERN.fullmatch(part)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def build_manifest(root: Path, per_subject_label: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    auxiliary_videos: list[str] = []
    for label_dir, label in (("ADL", "adl"), ("Fall", "fall")):
        for video in sorted((root / label_dir).rglob("*.mp4")):
            relative = video.relative_to(root)
            parsed = _subject_and_location(relative)
            if parsed is None:
                auxiliary_videos.append(str(relative))
                continue
            subject_id, location_id = parsed
            partition = SUBJECT_PARTITIONS.get(subject_id)
            if partition is None:
                raise ValueError(f"unexpected subject {subject_id}: {relative}")
            records.append(
                {
                    "subject_id": subject_id,
                    "location_id": location_id,
                    "label": label,
                    "partition": partition,
                    "small_validation_subset": False,
                    "video_id": f"fall29/{relative.with_suffix('')}",
                    "relative_path": str(relative),
                    "bytes": video.stat().st_size,
                    "sha256": digest(video, "sha256"),
                    **video_metadata(video),
                }
            )
    if not records:
        raise ValueError("no subject-labeled MP4 files found")
    observed_subjects = {record["subject_id"] for record in records}
    if observed_subjects != EXPECTED_SUBJECTS:
        missing = sorted(EXPECTED_SUBJECTS - observed_subjects)
        extra = sorted(observed_subjects - EXPECTED_SUBJECTS)
        raise ValueError(f"subject inventory mismatch: missing={missing}, extra={extra}")
    for subject_id in sorted(EXPECTED_SUBJECTS):
        for label in ("adl", "fall"):
            group = sorted(
                (
                    record
                    for record in records
                    if record["subject_id"] == subject_id and record["label"] == label
                ),
                key=lambda record: str(record["relative_path"]),
            )
            if group and len(group) < per_subject_label:
                raise ValueError(f"subject {subject_id} label {label} has only {len(group)} videos")
            for record in group[:per_subject_label]:
                record["small_validation_subset"] = True
    subject_partitions: dict[int, set[str]] = {}
    for record in records:
        subject_partitions.setdefault(record["subject_id"], set()).add(record["partition"])
    if any(len(partitions) != 1 for partitions in subject_partitions.values()):
        raise ValueError("subject leakage detected across partitions")
    hashes = [record["sha256"] for record in records]
    duplicate_hashes = sorted(value for value, count in Counter(hashes).items() if count > 1)
    label_counts = Counter(str(record["label"]) for record in records)
    partition_counts = Counter(str(record["partition"]) for record in records)
    subject_label_availability = {
        str(subject_id): sorted(
            {str(record["label"]) for record in records if record["subject_id"] == subject_id}
        )
        for subject_id in sorted(EXPECTED_SUBJECTS)
    }
    return {
        "dataset": "Video-Based Fall Detection Dataset with 2017 Activities from 29 Subjects",
        "short_name": "Figshare-Fall29",
        "version": 2,
        "source": "https://figshare.com/articles/dataset/28596332",
        "doi": "10.6084/m9.figshare.28596332.v2",
        "repository_license": "CC BY 4.0",
        "protocol": {
            "partition_unit": "subject",
            "threshold_development": DEVELOPMENT_SUBJECTS,
            "threshold_validation": VALIDATION_SUBJECTS,
            "locked_test": LOCKED_TEST_SUBJECTS,
            "small_subset_selection": (
                f"first {per_subject_label} sorted available videos per subject/label"
            ),
            "detection_delay_available": False,
            "detection_delay_unavailable_reason": (
                "the public manifest provides clip labels but no reviewed fall-onset timestamps"
            ),
        },
        "audit": {
            "subject_videos": len(records),
            "auxiliary_videos_excluded": sorted(auxiliary_videos),
            "subjects": sorted(subject_partitions),
            "labels": dict(sorted(label_counts.items())),
            "partitions": dict(sorted(partition_counts.items())),
            "subject_label_availability": subject_label_availability,
            "duplicate_content_hashes": duplicate_hashes,
            "subject_leakage": False,
        },
        "records": sorted(records, key=lambda record: str(record["video_id"])),
    }


def validate_protocol_declaration(
    declaration: object, manifest: dict[str, Any], per_subject_label: int
) -> None:
    if not isinstance(declaration, dict):
        raise ValueError("protocol declaration must be a mapping")
    if declaration.get("status") != "predeclared_before_pipeline_evaluation":
        raise ValueError("protocol declaration is not marked as predeclared")
    dataset = declaration.get("dataset")
    if not isinstance(dataset, dict) or (
        dataset.get("archive_bytes") != EXPECTED_ARCHIVE_BYTES
        or dataset.get("archive_md5") != EXPECTED_ARCHIVE_MD5
        or dataset.get("doi") != manifest["doi"]
    ):
        raise ValueError("protocol declaration dataset identity differs from the manifest")
    partition = declaration.get("partition")
    if not isinstance(partition, dict):
        raise ValueError("protocol declaration has no subject partition")
    for key in ("threshold_development", "threshold_validation", "locked_test"):
        if partition.get(key) != manifest["protocol"][key]:
            raise ValueError(f"protocol declaration {key} differs from the manifest")
    if partition.get("small_subset_selection") != (
        f"first {per_subject_label} sorted available videos per subject and clip label"
    ):
        raise ValueError("protocol declaration small-subset rule differs from the manifest")
    candidates = declaration.get("candidates")
    if not isinstance(candidates, dict) or candidates.get("detector_and_tracker_confidence") != [
        0.4,
        0.5,
        0.6,
        0.7,
        0.75,
        0.8,
    ]:
        raise ValueError("protocol declaration candidate grid differs from the implementation")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-declaration",
        type=Path,
        default=Path("configs/validation/figshare_fall29_v1.yaml"),
    )
    parser.add_argument("--per-subject-label", type=int, default=2)
    args = parser.parse_args()
    if args.per_subject_label < 1:
        parser.error("--per-subject-label must be positive")
    if args.archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        parser.error(f"archive byte-size mismatch: {args.archive.stat().st_size}")
    actual_md5 = digest(args.archive)
    if actual_md5 != EXPECTED_ARCHIVE_MD5:
        parser.error(f"archive MD5 mismatch: {actual_md5}")
    if not args.extract_dir.exists():
        args.extract_dir.mkdir(parents=True)
        safe_extract(args.archive, args.extract_dir)
    try:
        dataset_root = find_dataset_root(args.extract_dir)
        manifest = build_manifest(dataset_root, args.per_subject_label)
        declaration = yaml.safe_load(args.protocol_declaration.read_text(encoding="utf-8"))
        validate_protocol_declaration(declaration, manifest, args.per_subject_label)
    except ValueError as exc:
        parser.error(str(exc))
    manifest["archive"] = {
        "path": str(args.archive),
        "bytes": args.archive.stat().st_size,
        "md5": actual_md5,
    }
    manifest["protocol_declaration"] = {
        "path": str(args.protocol_declaration),
        "sha256": digest(args.protocol_declaration, "sha256"),
        "protocol_id": declaration["protocol_id"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "manifest.csv", manifest["records"])
    write_csv(
        args.output_dir / "small_validation.csv",
        [record for record in manifest["records"] if record["small_validation_subset"]],
    )
    print(json.dumps({"manifest": str(manifest_path), **manifest["audit"]}, indent=2))


if __name__ == "__main__":
    main()
