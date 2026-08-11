"""Session-scoped Gradio prototype with explicit real/mock and privacy boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fallguard.config import load_config
from fallguard.factory import build_pipeline
from fallguard.mock_run import run_mock_vertical_slice
from fallguard.session import make_session_id
from fallguard.video import VideoReader


def build_app(config_path: str = "configs/profiles/development.yaml") -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio UI is optional; install .[ui]") from exc

    base_config = load_config(config_path)

    def process(
        video_path: str | None,
        mode: str,
        provider: str,
        cloud_image_consent: bool,
        state: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        session = dict(state or {"events": [], "runs": 0})
        session["runs"] = int(session.get("runs", 0)) + 1
        if mode == "MOCK engineering demo":
            config = base_config.model_copy(deep=True)
            config.semantic.provider = "mock"
            config.semantic.allow_mock = True
            result = run_mock_vertical_slice(config, Path("artifacts/gradio-mock"))
            output = {
                "mode": "MOCK",
                "formal_benchmark_eligible": False,
                "events": [item.model_dump(mode="json") for item in result.events],
                "alerts": [item.model_dump(mode="json") for item in result.alerts],
            }
            session["events"] = output["events"]
            return output, session
        if not video_path:
            return {"error": "Upload a video for real mode."}, session
        config = base_config.model_copy(deep=True)
        config.semantic.provider = provider
        config.semantic.allow_cloud_images = bool(cloud_image_consent)
        if config.detector.weights_path is None and not config.detector.allow_weight_download:
            return {
                "error": (
                    "Real mode is disabled until an approved RF-DETR weights_path is configured; "
                    "the UI will not download weights implicitly."
                )
            }, session
        if provider in {"openai", "deepseek"} and not config.semantic.model:
            return {"error": f"Configure semantic.model before selecting {provider}."}, session
        pipeline = build_pipeline(
            config,
            with_real_frontend=True,
            keyframe_output_dir=Path("artifacts/gradio") / f"session-{session['runs']}",
            cloud_image_consent=cloud_image_consent,
        )
        assert pipeline.detector is not None
        try:
            pipeline.detector.load()
            events = []
            alerts = []
            video = Path(video_path)
            with VideoReader(
                video,
                source_id=config.runtime.source_id,
                session_id=make_session_id(f"gradio-{session['runs']}-{video.stem}"),
            ) as reader:
                for packet in reader.frames():
                    result = pipeline.process_frame(packet.image_bgr, packet.metadata)
                    events.extend(result.events)
                    alerts.extend(result.alerts)
            output = {
                "mode": "REAL",
                "events": [item.model_dump(mode="json") for item in events],
                "alerts": [item.model_dump(mode="json") for item in alerts],
            }
            session["events"] = output["events"]
            return output, session
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}, session

    with gr.Blocks(title="RFDETR-FallGuard") as demo:
        gr.Markdown(
            "# RFDETR-FallGuard\n"
            "上传视频是离线处理; 本原型不把上传称为持续实时监控。"
            "云端关键帧发送需要显式选择和隐私同意。"
        )
        state = gr.State({"events": [], "runs": 0})
        video = gr.Video(label="Uploaded video / recorded clip", sources=["upload", "webcam"])
        mode = gr.Radio(
            choices=["MOCK engineering demo", "Real offline video"],
            value="MOCK engineering demo",
            label="Run mode",
        )
        provider = gr.Dropdown(
            choices=["mock", "local_qwen", "openai", "deepseek"],
            value=base_config.semantic.provider,
            label="Semantic provider",
        )
        consent = gr.Checkbox(
            value=False,
            label="I explicitly consent to sending selected person keyframes to the cloud provider",
        )
        run = gr.Button("Run selected mode")
        output = gr.JSON(label="Structured result")
        run.click(process, inputs=[video, mode, provider, consent, state], outputs=[output, state])
        gr.Markdown(
            "API keys are read only from environment variables and are never displayed. "
            "MOCK results are not experiment metrics."
        )
    return demo


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/profiles/development.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app(args.config).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
