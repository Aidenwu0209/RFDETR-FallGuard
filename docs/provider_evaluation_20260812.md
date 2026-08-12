# Semantic Provider Evaluation — 2026-08-12

## Scope and provenance

This is a paid, text-only engineering smoke matrix, not a formal model benchmark. It used eight
fixed synthetic fall-event contexts spanning clear falls, controlled activities, track loss,
contradictory posture, and first-observation lying. No images or personal data were sent. The
three `gpt-5.6-*` names were served by the configured third-party gateway, not the official
OpenAI endpoint. DeepSeek requests used the official endpoint. DeepSeek V4 Pro was stopped and
excluded after the requested scope was narrowed to V4 Flash only.

Raw final report (ignored by Git):
`artifacts/provider-evaluation-20260812/provider-matrix-final.json`

SHA-256: `7b29d6d5d3405221e2b7cd8dfa8792ffbe9d3c1b71be1ba28a2b533ab29b3f85`

## Final fixed-matrix results

| Requested model | Effort | Schema | Decision | Risk | Alert | Mean / median / max latency |
|---|---:|---:|---:|---:|---:|---:|
| Gateway Luna | max | 8/8 | 8/8 | 8/8 | 8/8 | 7.51 / 7.12 / 10.06 s |
| Gateway Terra | xhigh | 8/8 | 8/8 | 8/8 | 8/8 | 12.77 / 9.60 / 37.44 s |
| Gateway Sol | high | 8/8 | 8/8 | 8/8 | 8/8 | 8.58 / 8.37 / 13.62 s |
| DeepSeek V4 Flash | high | 8/8 | 8/8 | 8/8 | 8/8 | 4.98 / 4.81 / 9.55 s |

The gateway reported roughly 4.7–4.9k input tokens per very small request, so gateway-side prefix
or accounting overhead dominates this sample. Its price is unknown and cannot be inferred from
official OpenAI pricing. V4 Flash reported 2,948 input and 1,643 completion tokens for the final
eight calls. At the live official prices checked on 2026-08-12, that run is approximately
USD 0.00047–0.00087 depending on prompt-cache hits.

The original V4 Flash prompt produced valid JSON but only 4/8 Pydantic-valid objects because it
used words for numeric/nullable fields. After the prompt explicitly defined number, boolean, and
null types, both the hardening run and final run achieved 8/8 strict-schema success. Both records
are retained under `artifacts/provider-evaluation-20260812/` rather than hiding the initial
failure.

All four models tied on this small easy-to-medium matrix. The result therefore supports transport
and schema integration, not a claim that any model is more accurate on real falls. V4 Flash is
the best default for text-only shadow validation because it was fastest and its official cost is
known. Luna was the fastest gateway option. Terra's latency tail is unsuitable for a synchronous
alert path in this sample.

## Local-model audit

| Local model | Current evidence | FallGuard readiness |
|---|---|---|
| ThinkingCap-Qwen3.6-27B Q4_K_M GGUF | 16,810,713,056 bytes; SHA-256 `b0651e28555bde7d2459ce99f091319b1a547143463e8d49f2aa7f572675fe67`; live API returned HTTP 200 and the correct synthetic `fall` JSON | Technically live, but about 7.1 generated tokens/s and about 47.5 s for the tested answer; consumes about 15.9/16.3 GiB VRAM. Text-only GGUF is incompatible with the current Hugging Face VLM provider and is not a real-time fallback. |
| Dolphin-v2 Qwen2.5-VL weights | Complete two-shard model plus processor/tokenizer; current config resolves locally under Transformers 5.15.0; retained CUDA OCR smoke evidence reports 1/1 success and 75.3 s for one document page | This is a document-OCR model, not a posture/fall semantic checkpoint. It has not been validated on FallGuard events and must not be labeled ready. |
| FallGuard Local Qwen provider | Provider code and `AutoModelForMultimodalLM` are available | `semantic.local_model_path` is unset; no dedicated posture VLM checkpoint is installed; `peft` is absent, so QLoRA training is not ready. |

The live 27B server leaves about 131 MiB free VRAM, so it cannot coexist with RF-DETR or the
Dolphin VLM on this 16 GiB GPU. Stop it before detector/training runs or move it to a different
machine/GPU.

## Decision and next validation gate

Use DeepSeek V4 Flash `high` as the text-only semantic provider for shadow evaluation, with no
fallback in formal experiments. Keep `AlertManager` as the final alert authority; model
`model_recommends_alert` remains evidence only. Luna can be a bounded comparison arm, not a
production dependency, because it is a gateway alias with unknown pricing and trust boundary.

Before choosing a formal provider or alert thresholds, run the real
`posture_multiclass -> ByteTrack -> Temporal -> Event -> Semantic -> Alert` pipeline on a small
identity/video-isolated labeled validation set. Report event sensitivity, false alarms per hour,
time-to-alert, track fragmentation, semantic schema success, latency p50/p95, and an ablation
without semantics. The current production event text contains mainly lifecycle reasons and
keyframe roles, while this matrix used richer motion evidence; enrich and freeze the event-context
schema before treating these provider results as representative.

No API key was persisted in repository files or the remote process environment. Since both keys
were exposed in a screenshot/chat, rotate them after this test.
