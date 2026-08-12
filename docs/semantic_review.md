# Provider-Agnostic Semantic Review

## Contract

All providers return `SemanticAssessment` with:

```text
decision: fall | not_fall | uncertain
confidence: float | null
reason
attempt_to_stand: bool | null
risk_level: low | medium | high | unknown
provider
model
input_mode
latency_ms
schema_valid
provider_success
model_recommends_alert
input_tokens / output_tokens / reasoning_tokens when supplied
ground_truth_verified
```

`ground_truth_verified` defaults to false. A model cannot promote its own output to ground truth.
Only labeled training/evaluation data sets it true.

## Capabilities

| Provider | Images | Structured output | Default input | Notes |
|---|---:|---:|---|---|
| Mock | yes | yes | test fixture | Deterministic and never formal evidence |
| Local Qwen | yes | prompt + local validation | images and text | Existing local path only; no implicit download |
| OpenAI | model-dependent | official Responses structured parsing | images and text | Model name is configuration; two image-consent gates |
| DeepSeek | no | JSON Output plus Pydantic validation | text | OpenAI-compatible transport does not imply vision |

The selected model must actually support the requested capability. The router truncates images
to the configured/provider maximum, removes images for text-only providers, and records a
fallback reason. Formal experiments disable fallback.

`semantic.reasoning_effort` is passed explicitly when configured. OpenAI-compatible Responses
requests set `store=false`. DeepSeek V4 requests can enable thinking with
`semantic.deepseek_thinking`; its prompt spells out numeric, boolean, and null field types before
Pydantic validation. Provider aliases exposed by a third-party gateway are gateway models, not
proof that the same names exist on the official OpenAI endpoint.

## Health and paid tests

`health_check(live=False)` only inspects local packages, model paths, and key presence. It never
performs a billable request. Real API integration tests are marked `api` and additionally require:

```text
RUN_PAID_API_INTEGRATION_TESTS=1
```

## Privacy

Cloud images require both `semantic.allow_cloud_images=true` and per-request
`cloud_image_consent=true`. The UI checkbox supplies only the second gate. Logs do not contain
API keys, authorization headers, Base64 images, or full request bodies. DeepSeek receives only
text event context in this implementation.

## Alert ownership

`model_recommends_alert` is evidence. `AlertManager` owns `should_alert` and applies application
policy to temporal state and schema-valid semantic output. Tests verify that a provider's
recommendation cannot override a `not_fall` policy decision.
