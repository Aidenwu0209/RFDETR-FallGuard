#!/usr/bin/env python3
"""Build, start, probe, and always close the Gradio prototype."""

from __future__ import annotations

import argparse
import socket

import httpx

from app.gradio_app import build_app


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/profiles/development.yaml")
    args = parser.parse_args()
    app = build_app(args.config)
    port = free_port()
    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=port,
            prevent_thread_lock=True,
            quiet=True,
        )
        response = httpx.get(f"http://127.0.0.1:{port}/", timeout=10)
        response.raise_for_status()
        print(f"GRADIO_SMOKE_OK status={response.status_code} port={port}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
