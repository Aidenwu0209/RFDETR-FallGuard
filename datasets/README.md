# Dataset manifests

Raw datasets are never committed here. Future UP-Fall, Le2i, and self-collected sources must
be split by person, video, and scene before frame extraction to prevent adjacent-frame leakage.
Semantic QLoRA manifests use one JSON-serialized `SemanticTrainingSample` per line. The same
`split_group` may not appear in train, validation, and test manifests.

## Current authorized datasets

- Fallen Person (Roboflow Universe v1): 2,876 images, four published classes, CC BY 4.0. Export
  remains outside Git and requires a Roboflow login/API key. The exact exported class order must
  be audited before training; `fallen` must not be silently treated as a dynamic `falling` label.
- GMDCSA-24 v2.0: the official Zenodo archive is verified by MD5 and stored under ignored
  `data/raw/`. `prepare_gmdcsa24.py` creates ignored full and 16-video manifests. S1-S2 are for
  threshold development, S3 for one validation pass, and S4 remains locked test data.

GMDCSA-24 clip labels do not contain precise fall onset/end timestamps. They support clip-level
event presence, miss, and false-alarm metrics. Detection-delay evaluation needs a separate,
human-confirmed timestamp manifest and must never be inferred from filenames or descriptions.
