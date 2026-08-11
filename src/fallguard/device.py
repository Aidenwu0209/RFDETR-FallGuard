"""Device discovery without assuming that an NVIDIA driver implies usable PyTorch CUDA."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    resolved: str
    torch_available: bool
    cuda_available: bool
    torch_version: str | None
    cuda_runtime_version: str | None
    gpu_name: str | None
    gpu_memory_total_mib: int | None
    driver_visible: bool
    diagnostic: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _driver_probe() -> tuple[bool, str | None, int | None]:
    if shutil.which("nvidia-smi") is None:
        return False, None, None
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
        name, memory = [part.strip() for part in result.stdout.splitlines()[0].split(",", 1)]
        return True, name, int(memory)
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return True, None, None


def inspect_device(requested: str = "auto") -> DeviceInfo:
    driver_visible, driver_name, driver_memory = _driver_probe()
    try:
        import torch
    except ImportError:
        resolved = "cpu" if requested == "auto" else requested
        return DeviceInfo(
            requested=requested,
            resolved=resolved,
            torch_available=False,
            cuda_available=False,
            torch_version=None,
            cuda_runtime_version=None,
            gpu_name=driver_name,
            gpu_memory_total_mib=driver_memory,
            driver_visible=driver_visible,
            diagnostic=(
                "PyTorch is not installed; NVIDIA driver visibility is not CUDA runtime proof"
            ),
        )

    cuda_available = bool(torch.cuda.is_available())
    resolved = (
        "cuda"
        if requested == "auto" and cuda_available
        else ("cpu" if requested == "auto" else requested)
    )
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else driver_name
    diagnostic = (
        "PyTorch CUDA is available" if cuda_available else "PyTorch cannot currently use CUDA"
    )
    return DeviceInfo(
        requested=requested,
        resolved=resolved,
        torch_available=True,
        cuda_available=cuda_available,
        torch_version=str(torch.__version__),
        cuda_runtime_version=str(torch.version.cuda) if torch.version.cuda else None,
        gpu_name=gpu_name,
        gpu_memory_total_mib=driver_memory,
        driver_visible=driver_visible,
        diagnostic=diagnostic,
    )
