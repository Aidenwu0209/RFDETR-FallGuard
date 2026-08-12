#!/usr/bin/env python3
"""Build three derived subject-isolated CV manifests while keeping Subject 4 locked."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

CV_SUBJECTS = (1, 2, 3)
LOCKED_TEST_SUBJECT = 4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_fold(source: dict[str, Any], validation_subject: int) -> dict[str, Any]:
    if validation_subject not in CV_SUBJECTS:
        raise ValueError("validation subject must be one of Subjects 1-3")
    source_subjects = {int(record["subject_id"]) for record in source.get("records", [])}
    if source_subjects != {*CV_SUBJECTS, LOCKED_TEST_SUBJECT}:
        raise ValueError(f"expected subjects 1-4 exactly, got {sorted(source_subjects)}")
    development_subjects = [subject for subject in CV_SUBJECTS if subject != validation_subject]
    partition_by_subject = {
        **{subject: "threshold_development" for subject in development_subjects},
        validation_subject: "threshold_validation",
        LOCKED_TEST_SUBJECT: "locked_test",
    }
    records = [
        {**record, "partition": partition_by_subject[int(record["subject_id"])]}
        for record in source["records"]
    ]
    subject_partitions: dict[int, set[str]] = {}
    for record in records:
        subject_partitions.setdefault(int(record["subject_id"]), set()).add(
            str(record["partition"])
        )
    if any(len(partitions) != 1 for partitions in subject_partitions.values()):
        raise ValueError("subject leakage detected in derived fold")
    return {
        **{
            key: value
            for key, value in source.items()
            if key not in {"protocol", "audit", "records"}
        },
        "protocol_id": f"gmdcsa24-recovery-cv-v2-fold-s{validation_subject}",
        "protocol": {
            "partition_unit": "subject",
            "threshold_development": development_subjects,
            "threshold_validation": [validation_subject],
            "locked_test": [LOCKED_TEST_SUBJECT],
            "cross_validation_subjects": list(CV_SUBJECTS),
            "warning": (
                "Subject 4 is locked; do not run it until one profile is frozen from all folds"
            ),
        },
        "audit": {
            **source.get("audit", {}),
            "subject_leakage": False,
            "partition_counts": dict(
                sorted(Counter(str(record["partition"]) for record in records).items())
            ),
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for validation_subject in CV_SUBJECTS:
        fold = build_fold(source, validation_subject)
        fold["source_manifest_sha256"] = file_sha256(args.manifest)
        output = args.output_dir / f"manifest-fold-s{validation_subject}.json"
        output.write_text(json.dumps(fold, indent=2) + "\n", encoding="utf-8")
        outputs.append(str(output))
    print(json.dumps({"outputs": outputs, "locked_test_subject": 4}, indent=2))


if __name__ == "__main__":
    main()
