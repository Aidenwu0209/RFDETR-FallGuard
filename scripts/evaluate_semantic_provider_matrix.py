#!/usr/bin/env python3
"""Compare cloud semantic-review models on a fixed, text-only event matrix."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["fall", "not_fall", "uncertain"]
Risk = Literal["low", "medium", "high", "unknown"]
ReasoningEffort = Literal["low", "medium", "high", "xhigh", "max"]


class Judgment(BaseModel):
    """Strict common output contract used for every evaluated provider."""

    model_config = ConfigDict(extra="forbid")

    decision: Decision
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason: str = Field(min_length=1)
    attempt_to_stand: bool | None = None
    risk_level: Risk
    model_recommends_alert: bool


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    evidence: str
    expected_decision: Decision
    expected_attempt_to_stand: bool | None
    expected_risk: Risk
    expected_alert: bool


@dataclass(frozen=True)
class ModelSpec:
    label: str
    provider: Literal["openai_compatible_gateway", "deepseek"]
    model: str
    reasoning_effort: ReasoningEffort


POLICY = (
    "You review event evidence from a fall-detection pipeline. Use only the supplied evidence. "
    "Choose fall for a rapid downward posture transition followed by sustained lying, even if "
    "the person later attempts to stand. Choose not_fall for a clearly controlled activity of "
    "daily living without fall evidence. Choose uncertain when evidence is insufficient, lost, "
    "or contradictory. Recommend an alert only for fall. Use risk high for fall, low for clear "
    "not_fall, and unknown for uncertain. Return the requested JSON schema."
)

DEEPSEEK_JSON_CONTRACT = (
    "Return exactly one JSON object with only these keys and types: "
    '{"decision":"fall|not_fall|uncertain","confidence":0.0,'
    '"reason":"string","attempt_to_stand":false,'
    '"risk_level":"low|medium|high|unknown","model_recommends_alert":false}. '
    "confidence must be a JSON number from 0 through 1, or null; never use a word such as "
    "low, medium, or high. attempt_to_stand must be true, false, or null; use null when the "
    "evidence cannot establish it, and never use the string unknown."
)

CASES = (
    EvalCase(
        "rapid_fall_attempted_recovery",
        "Track 7 changed standing -> fallen in 0.43 s; normalized downward velocity 0.91 "
        "frame-height/s; width/height 1.54; remained lying 3.8 s; no tracking gap; then "
        "attempted to stand.",
        "fall",
        True,
        "high",
        True,
    ),
    EvalCase(
        "rapid_fall_no_recovery",
        "Track 12 changed standing -> fallen in 0.55 s; normalized downward velocity 0.72 "
        "frame-height/s; width/height rose from 0.48 to 1.38; remained lying 5.1 s; no attempt "
        "to stand was observed.",
        "fall",
        False,
        "high",
        True,
    ),
    EvalCase(
        "controlled_chair_sit",
        "Track 3 changed standing -> sitting over 2.6 s; normalized downward velocity 0.08 "
        "frame-height/s; width/height stayed 0.57; remained seated upright; no fallen or lying "
        "observation.",
        "not_fall",
        None,
        "low",
        False,
    ),
    EvalCase(
        "pick_up_object",
        "Track 5 changed standing -> bending -> standing over 1.8 s; peak normalized downward "
        "velocity 0.17 frame-height/s; width/height stayed below 0.72; full upright recovery in "
        "0.7 s; no lying observation.",
        "not_fall",
        None,
        "low",
        False,
    ),
    EvalCase(
        "controlled_bed_transfer",
        "Track 9 moved sitting -> lying over 4.2 s beside a bed; normalized downward velocity "
        "0.06 frame-height/s; no fallen class; motion was continuous and slow; remained lying.",
        "not_fall",
        None,
        "low",
        False,
    ),
    EvalCase(
        "track_gap_during_descent",
        "Track 14 was standing, then disappeared for 1.4 s which exceeds the 0.8 s track "
        "timeout. A new track appeared lying. Identity continuity and descent velocity are "
        "unavailable.",
        "uncertain",
        None,
        "unknown",
        False,
    ),
    EvalCase(
        "contradictory_posture",
        "Track 18 alternated fallen and standing on four consecutive frames with confidence "
        "0.41-0.46; vertical velocity was 0.03 frame-height/s; a camera cut occurred; duration "
        "after the cut is 0.3 s.",
        "uncertain",
        None,
        "unknown",
        False,
    ),
    EvalCase(
        "static_lying_first_observation",
        "Track 21 is already lying in the first available frame. No earlier frames, posture "
        "transition, or velocity are available. The person remains lying for 6 s.",
        "uncertain",
        None,
        "unknown",
        False,
    ),
)

DEFAULT_MODELS = (
    ModelSpec("luna-max", "openai_compatible_gateway", "gpt-5.6-luna", "max"),
    ModelSpec("terra-xhigh", "openai_compatible_gateway", "gpt-5.6-terra", "xhigh"),
    ModelSpec("sol-high", "openai_compatible_gateway", "gpt-5.6-sol", "high"),
    ModelSpec("deepseek-v4-flash-high", "deepseek", "deepseek-v4-flash", "high"),
)


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _required_secret(name: str, *, prompt: bool) -> str:
    value = os.getenv(name)
    if value:
        return value
    if prompt:
        return getpass.getpass(f"{name}: ")
    raise SystemExit(f"{name} is required; use the environment or --prompt-for-keys")


def _usage_value(value: object, name: str) -> int | None:
    candidate = getattr(value, name, None)
    return candidate if isinstance(candidate, int) else None


def _gateway_review(
    client: OpenAI, spec: ModelSpec, case: EvalCase, timeout_seconds: float
) -> tuple[Judgment, dict[str, int | None], float, str]:
    started = time.perf_counter()
    response = client.responses.parse(
        model=spec.model,
        reasoning={"effort": spec.reasoning_effort},
        input=[
            {"role": "developer", "content": POLICY},
            {"role": "user", "content": case.evidence},
        ],
        text_format=Judgment,
        max_output_tokens=1024,
        store=False,
        timeout=timeout_seconds,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    if response.output_parsed is None:
        raise ValueError("gateway returned no parsed structured output")
    usage = response.usage
    details = getattr(usage, "output_tokens_details", None)
    return (
        response.output_parsed,
        {
            "input_tokens": _usage_value(usage, "input_tokens"),
            "output_tokens": _usage_value(usage, "output_tokens"),
            "reasoning_tokens": _usage_value(details, "reasoning_tokens"),
        },
        latency_ms,
        str(response.model),
    )


def _deepseek_review(
    api_key: str,
    spec: ModelSpec,
    case: EvalCase,
    timeout_seconds: float,
) -> tuple[Judgment, dict[str, int | None], float, str]:
    started = time.perf_counter()
    response = httpx.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": spec.model,
            "messages": [
                {"role": "system", "content": POLICY},
                {
                    "role": "user",
                    "content": DEEPSEEK_JSON_CONTRACT + "\nEvent evidence:\n" + case.evidence,
                },
            ],
            "thinking": {"type": "enabled"},
            "reasoning_effort": spec.reasoning_effort,
            "response_format": {"type": "json_object"},
            "max_tokens": 4096,
            "stream": False,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    body = response.json()
    message = body["choices"][0]["message"]
    judgment = Judgment.model_validate_json(message["content"])
    latency_ms = (time.perf_counter() - started) * 1000
    usage = body.get("usage", {})
    reasoning_tokens = usage.get("reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
    return (
        judgment,
        {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": reasoning_tokens,
        },
        latency_ms,
        str(body.get("model", spec.model)),
    )


def score(case: EvalCase, judgment: Judgment) -> dict[str, bool]:
    return {
        "decision_correct": judgment.decision == case.expected_decision,
        "alert_correct": judgment.model_recommends_alert == case.expected_alert,
        "attempt_correct": (
            True
            if case.expected_attempt_to_stand is None
            else judgment.attempt_to_stand == case.expected_attempt_to_stand
        ),
        "risk_correct": judgment.risk_level == case.expected_risk,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] == "success"]
    latencies = [float(row["latency_ms"]) for row in successful]

    def accuracy(field: str) -> float | None:
        if not successful:
            return None
        return sum(bool(row["score"][field]) for row in successful) / len(successful)

    return {
        "calls": len(rows),
        "successful_calls": len(successful),
        "schema_success_rate": len(successful) / len(rows) if rows else None,
        "decision_accuracy": accuracy("decision_correct"),
        "alert_accuracy": accuracy("alert_correct"),
        "attempt_accuracy": accuracy("attempt_correct"),
        "risk_accuracy": accuracy("risk_correct"),
        "all_scored_fields_accuracy": (
            sum(all(row["score"].values()) for row in successful) / len(successful)
            if successful
            else None
        ),
        "latency_ms": (
            {
                "mean": round(statistics.fmean(latencies), 1),
                "median": round(statistics.median(latencies), 1),
                "minimum": round(min(latencies), 1),
                "maximum": round(max(latencies), 1),
            }
            if latencies
            else None
        ),
        "tokens": {
            key: sum(
                int(row["usage"][key] or 0)
                for row in successful
                if isinstance(row.get("usage"), dict)
            )
            for key in ("input_tokens", "output_tokens", "reasoning_tokens")
        },
    }


def _write_report(
    path: Path,
    *,
    models: tuple[ModelSpec, ...],
    rows: list[dict[str, Any]],
    gateway_base_url: str,
    completed: bool,
) -> dict[str, Any]:
    summaries = {
        spec.label: summarize([row for row in rows if row["model_label"] == spec.label])
        for spec in models
    }
    report = {
        "validation_kind": "PAID_TEXT_ONLY_SEMANTIC_PROVIDER_MATRIX_NOT_FORMAL_BENCHMARK",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed": completed,
        "privacy": {
            "images_sent": False,
            "personal_data_sent": False,
            "synthetic_fixed_event_contexts_only": True,
            "secret_values_rendered_or_stored": False,
        },
        "gateway": {
            "base_url_host": urlparse(gateway_base_url).hostname,
            "official_openai_endpoint": urlparse(gateway_base_url).hostname == "api.openai.com",
        },
        "deepseek_endpoint_host": "api.deepseek.com",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "policy_sha256": hashlib.sha256(POLICY.encode("utf-8")).hexdigest(),
        "cases_sha256": _sha256_json([asdict(case) for case in CASES]),
        "case_count": len(CASES),
        "models": [asdict(spec) for spec in models],
        "summaries": summaries,
        "rows": rows,
        "paid_api_calls_performed": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-paid-calls", action="store_true")
    parser.add_argument("--prompt-for-keys", action="store_true")
    parser.add_argument("--gateway-base-url", default="https://codexx.de5.net/v1")
    parser.add_argument("--timeout-seconds", type=float, default=240)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[spec.label for spec in DEFAULT_MODELS],
        help="optional model labels; by default all configured models are evaluated",
    )
    args = parser.parse_args()
    if not args.execute_paid_calls:
        parser.error("paid calls require --execute-paid-calls")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    selected_models = tuple(
        spec for spec in DEFAULT_MODELS if args.models is None or spec.label in args.models
    )
    needs_gateway = any(spec.provider == "openai_compatible_gateway" for spec in selected_models)
    needs_deepseek = any(spec.provider == "deepseek" for spec in selected_models)
    gateway_client = (
        OpenAI(
            api_key=_required_secret("OPENAI_API_KEY", prompt=args.prompt_for_keys),
            base_url=args.gateway_base_url,
            timeout=args.timeout_seconds,
        )
        if needs_gateway
        else None
    )
    deepseek_key = (
        _required_secret("DEEPSEEK_API_KEY", prompt=args.prompt_for_keys)
        if needs_deepseek
        else None
    )
    rows: list[dict[str, Any]] = []
    for spec in selected_models:
        for case in CASES:
            row: dict[str, Any] = {
                "model_label": spec.label,
                "provider": spec.provider,
                "requested_model": spec.model,
                "reasoning_effort": spec.reasoning_effort,
                "case_id": case.case_id,
                "expected": {
                    "decision": case.expected_decision,
                    "attempt_to_stand": case.expected_attempt_to_stand,
                    "risk_level": case.expected_risk,
                    "model_recommends_alert": case.expected_alert,
                },
            }
            try:
                if spec.provider == "openai_compatible_gateway":
                    assert gateway_client is not None
                    judgment, usage, latency_ms, actual_model = _gateway_review(
                        gateway_client, spec, case, args.timeout_seconds
                    )
                else:
                    assert deepseek_key is not None
                    judgment, usage, latency_ms, actual_model = _deepseek_review(
                        deepseek_key, spec, case, args.timeout_seconds
                    )
                row.update(
                    {
                        "status": "success",
                        "actual_model": actual_model,
                        "latency_ms": round(latency_ms, 1),
                        "usage": usage,
                        "judgment": judgment.model_dump(),
                        "score": score(case, judgment),
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    }
                )
            rows.append(row)
            _write_report(
                args.output_json,
                models=selected_models,
                rows=rows,
                gateway_base_url=args.gateway_base_url,
                completed=False,
            )
            print(
                json.dumps(
                    {
                        "model": spec.label,
                        "case": case.case_id,
                        "status": row["status"],
                        "decision": row.get("judgment", {}).get("decision"),
                        "latency_ms": row.get("latency_ms"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    report = _write_report(
        args.output_json,
        models=selected_models,
        rows=rows,
        gateway_base_url=args.gateway_base_url,
        completed=True,
    )
    print(json.dumps({"summaries": report["summaries"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
