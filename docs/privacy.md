# Privacy and Logging

- Full-frame retention defaults to false; person-crop retention defaults to true.
- Online raw frames live only in a bounded in-memory deque.
- Persisted schemas contain file paths and SHA-256 hashes, not NumPy arrays or Base64 payloads.
- Cloud image review requires configured consent plus per-request consent.
- Provider health checks do not call the network by default.
- Logs redact authorization values, key/token/secret/password assignments, OpenAI-style keys,
  and image data URLs.
- API keys are read from environment variables and never shown in Gradio.
- Uploaded video is offline processing. A webcam-recorded clip is not described as continuous
  real-time monitoring.

Artifacts and data directories are ignored by Git. Before sharing any experiment directory,
audit it separately for faces, source paths, environment metadata, and dataset licensing.
