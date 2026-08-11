# Third-Party API Verification

Checked on 2026-08-11 without model downloads or paid API calls:

| Integration | Target | Verification |
|---|---|---|
| RF-DETR | `rfdetr==1.9.1` | PyPI index plus wheel source/AST; Nano/Small, predict, train, evaluate and `TrainConfig` fields |
| ByteTrack | `supervision==0.30.0` | PyPI index plus wheel signature and real two-frame ID-continuity test |
| Gradio | `gradio==6.22.0` | Built app, started local server, HTTP 200 probe, closed server |
| OpenAI | `openai==2.53.0` | PyPI index, official Structured Outputs docs, local SDK `responses.parse` contract test |
| DeepSeek | HTTPX transport | Official `/chat/completions` JSON Output docs; no key or live call used |
| Qwen | `transformers==5.15.0`, `peft==0.20.0` | Official Qwen3.5-4B model card specifies `AutoModelForMultimodalLM`; model not downloaded |

Authoritative references:

- <https://github.com/roboflow/rf-detr>
- <https://api-docs.deepseek.com/guides/json_mode/>
- <https://platform.openai.com/docs/guides/structured-outputs>
- <https://huggingface.co/Qwen/Qwen3.5-4B>

The Supervision test emits an upstream deprecation warning: its `ByteTrack` class is scheduled
for removal in 0.31. Exact 0.30.0 pinning and the project adapter contain that change risk; a
future upgrade must replace/revalidate the backend before changing the pin.
