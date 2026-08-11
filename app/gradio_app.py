"""Session-scoped Gradio prototype with explicit real/mock and privacy boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from fallguard.config import load_config
from fallguard.factory import build_pipeline
from fallguard.mock_run import run_mock_vertical_slice
from fallguard.session import make_session_id
from fallguard.status import environment_status
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
        model_variant: str,
        weights_path: str,
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
        config.semantic.provider = cast(
            Literal["none", "mock", "openai", "deepseek", "local_qwen"], provider
        )
        config.semantic.allow_cloud_images = bool(cloud_image_consent)
        configured_weights = (
            Path(weights_path.strip())
            if weights_path.strip()
            else Path(f"weights/official/rf-detr-{model_variant}.pth")
        )
        official_names = {"rf-detr-nano.pth": "nano", "rf-detr-small.pth": "small"}
        weight_variant = official_names.get(configured_weights.name)
        if weight_variant is not None and weight_variant != model_variant:
            return {
                "error": (
                    f"Selected {model_variant}, but {configured_weights.name} is the official "
                    f"{weight_variant} checkpoint. Choose matching values."
                )
            }, session
        config.detector = config.detector.model_copy(
            update={"model_variant": model_variant, "weights_path": configured_weights}
        )
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
        with gr.Tabs():
            with gr.Tab("Pipeline Demo"):
                video = gr.Video(
                    label="Uploaded video / recorded clip", sources=["upload", "webcam"]
                )
                mode = gr.Radio(
                    choices=["MOCK engineering demo", "Real offline video"],
                    value="MOCK engineering demo",
                    label="Run mode",
                )
                model_variant = gr.Radio(
                    choices=["nano", "small"],
                    value=base_config.detector.model_variant,
                    label="RF-DETR variant",
                )
                weights_path = gr.Textbox(
                    value="",
                    placeholder="blank = weights/official/rf-detr-{variant}.pth",
                    label="Local weights path (optional override)",
                )
                provider = gr.Dropdown(
                    choices=["mock", "local_qwen", "openai", "deepseek"],
                    value=base_config.semantic.provider,
                    label="Semantic provider",
                )
                consent = gr.Checkbox(
                    value=False,
                    label=(
                        "I explicitly consent to sending selected person keyframes to the "
                        "cloud provider"
                    ),
                )
                run = gr.Button("Run selected mode")
                output = gr.JSON(label="Structured result")
                run.click(
                    process,
                    inputs=[
                        video,
                        mode,
                        provider,
                        consent,
                        model_variant,
                        weights_path,
                        state,
                    ],
                    outputs=[output, state],
                )
            with gr.Tab("Models & GPU"):
                model_status = gr.JSON(value=lambda: environment_status()["models"])
                gpu_status = gr.JSON(value=lambda: environment_status()["cuda"])
                refresh_models = gr.Button("Refresh model/GPU status")
                refresh_models.click(
                    lambda: (environment_status()["models"], environment_status()["cuda"]),
                    outputs=[model_status, gpu_status],
                )
                gr.Markdown(
                    "Smoke reports are real single-image GPU checks, not formal accuracy "
                    "benchmarks."
                )
            with gr.Tab("Datasets & Protocol"):
                dataset_status = gr.JSON(value=lambda: environment_status()["datasets"])
                refresh_datasets = gr.Button("Refresh dataset status")
                refresh_datasets.click(
                    lambda: environment_status()["datasets"], outputs=dataset_status
                )
                gr.Markdown(
                    "GMDCSA-24 split is subject-isolated: S1-S2 threshold development, "
                    "S3 threshold validation, S4 locked test. Clip labels do not provide exact "
                    "fall timestamps; detection delay stays unavailable until human annotation."
                )
            with gr.Tab("Training & Thresholds"):
                gr.JSON(
                    value={
                        "short_finetune": {
                            "variants": ["nano", "small"],
                            "batch_size": 1,
                            "gradient_accumulation": 4,
                            "data_required": "Fallen Person COCO export",
                        },
                        "formal_threshold_state": "pending grouped validation",
                        "test_set_policy": "Subject 4 remains locked until thresholds freeze",
                    }
                )
                gr.Markdown(
                    "The UI reports training/validation readiness. Training is intentionally run "
                    "from the audited CLI so logs, checkpoint hashes, and exact parameters remain "
                    "reproducible."
                )
            with gr.Tab("API Configuration"):
                api_status = gr.JSON(value=lambda: environment_status()["api_keys"])
                refresh_api = gr.Button("Validate local API-key configuration")
                refresh_api.click(lambda: environment_status()["api_keys"], outputs=api_status)
                gr.Markdown(
                    "Only environment-variable presence is checked. Values are never displayed, "
                    "and this tab performs no network or paid API call."
                )
        gr.Markdown("MOCK results are not experiment metrics.")
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
