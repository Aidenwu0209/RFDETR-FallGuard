#!/usr/bin/env python3
"""Extract, audit, and group-split GMDCSA-24 without subject/video leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

EXPECTED_ARCHIVE_MD5 = "49bf4eb15a84cc84cb0a4f9c6ddd59e6"
SUBJECT_PARTITIONS = {
    1: "threshold_development",
    2: "threshold_development",
    3: "threshold_validation",
    4: "locked_test",
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
        for member in bundle.infolist():
            resolved = (target / member.filename).resolve()
            if target_resolved not in resolved.parents and resolved != target_resolved:
                raise ValueError(f"unsafe archive member: {member.filename}")
        bundle.extractall(target)


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
    return {
        "fps": fps,
        "frames": frames,
        "duration_seconds": frames / fps if fps > 0 else None,
        "width": width,
        "height": height,
    }


def find_dataset_root(extracted: Path) -> Path:
    candidates = [path.parent for path in extracted.rglob("Subject 1") if path.is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"expected one GMDCSA-24 dataset root, found {len(candidates)}")
    return candidates[0]


def build_manifest(root: Path, per_subject_label: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for subject, partition in SUBJECT_PARTITIONS.items():
        for label in ("ADL", "Fall"):
            videos = sorted((root / f"Subject {subject}" / label).glob("*.mp4"))
            for index, video in enumerate(videos):
                records.append(
                    {
                        "subject_id": subject,
                        "label": label.lower(),
                        "partition": partition,
                        "small_validation_subset": index < per_subject_label,
                        "video_id": f"subject-{subject}/{label.lower()}/{video.stem}",
                        "relative_path": str(video.relative_to(root)),
                        "bytes": video.stat().st_size,
                        "sha256": digest(video, "sha256"),
                        **video_metadata(video),
                    }
                )
    if not records:
        raise ValueError("no MP4 files found")
    subject_partitions: dict[int, set[str]] = {}
    for record in records:
        subject_partitions.setdefault(record["subject_id"], set()).add(record["partition"])
    if any(len(partitions) != 1 for partitions in subject_partitions.values()):
        raise ValueError("subject leakage detected across partitions")
    hashes = [record["sha256"] for record in records]
    duplicate_hashes = sorted(value for value, count in Counter(hashes).items() if count > 1)
    return {
        "dataset": "GMDCSA-24",
        "version": "2.0",
        "source": "https://zenodo.org/records/12921216",
        "doi": "10.5281/zenodo.12921216",
        "repository_license": "MIT",
        "protocol": {
            "partition_unit": "subject",
            "threshold_development": [1, 2],
            "threshold_validation": [3],
            "locked_test": [4],
            "small_subset_selection": f"first {per_subject_label} sorted videos per subject/label",
            "warning": (
                "clip labels support clip-level event presence only; detection delay requires "
                "separate human-confirmed fall timestamps"
            ),
        },
        "audit": {
            "videos": len(records),
            "subjects": sorted(subject_partitions),
            "duplicate_content_hashes": duplicate_hashes,
            "subject_leakage": False,
        },
        "records": records,
    }


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
    parser.add_argument("--per-subject-label", type=int, default=2)
    args = parser.parse_args()
    if args.per_subject_label < 1:
        parser.error("--per-subject-label must be positive")
    actual_md5 = digest(args.archive)
    if actual_md5 != EXPECTED_ARCHIVE_MD5:
        parser.error(f"archive MD5 mismatch: {actual_md5}")
    if not args.extract_dir.exists():
        args.extract_dir.mkdir(parents=True)
        safe_extract(args.archive, args.extract_dir)
    dataset_root = find_dataset_root(args.extract_dir)
    manifest = build_manifest(dataset_root, args.per_subject_label)
    manifest["archive"] = {
        "path": str(args.archive),
        "bytes": args.archive.stat().st_size,
        "md5": actual_md5,
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
