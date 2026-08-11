"""Reproducible experiment snapshots without secrets or mock/result ambiguity."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fallguard.config import AppConfig
from fallguard.device import inspect_device
from fallguard.exceptions import FormalBenchmarkRejectedError


def _command_output(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "returncode": None, "error": str(exc)}


class ExperimentRecorder:
    def __init__(self, root: str | Path = "experiments") -> None:
        self.root = Path(root)

    def create(
        self,
        *,
        name: str,
        config: AppConfig,
        mock_components: list[str],
        notes: str = "",
    ) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.root / f"{timestamp}-{name}"
        run_dir.mkdir(parents=True, exist_ok=False)
        formal_config_ready = True
        formal_readiness_reason = None
        try:
            config.assert_formal_ready()
        except FormalBenchmarkRejectedError as exc:
            formal_config_ready = False
            formal_readiness_reason = str(exc)
        self._write_json(run_dir / "config.json", config.model_dump(mode="json"))
        self._write_json(
            run_dir / "environment.json",
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "device": inspect_device(config.runtime.device).as_dict(),
                "mock_components": mock_components,
                "formal_config_ready": formal_config_ready,
                "formal_readiness_reason": formal_readiness_reason,
                "formal_result_eligible": not mock_components and formal_config_ready,
                "notes": notes,
            },
        )
        self._write_json(
            run_dir / "git.json",
            {
                "head": _command_output(["git", "rev-parse", "HEAD"]),
                "status": _command_output(["git", "status", "--short", "--branch"]),
                "diff_stat": _command_output(["git", "diff", "--stat"]),
            },
        )
        freeze = _command_output([sys.executable, "-m", "pip", "freeze"])
        (run_dir / "environment-freeze.txt").write_text(
            str(freeze.get("stdout", "")),
            encoding="utf-8",
        )
        return run_dir

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
