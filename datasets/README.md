# Dataset manifests

Raw datasets are never committed here. Future UP-Fall, Le2i, and self-collected sources must
be split by person, video, and scene before frame extraction to prevent adjacent-frame leakage.
Semantic QLoRA manifests use one JSON-serialized `SemanticTrainingSample` per line. The same
`split_group` may not appear in train, validation, and test manifests.
