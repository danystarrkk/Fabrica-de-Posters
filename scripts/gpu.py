"""Detección de GPU NVIDIA y decisión de backend (CUDA vs CPU).

ComfyUI funciona:
  - Con GPU NVIDIA dedicada: instala PyTorch con índice CUDA y se lanza
    con `python3 main.py` (ComfyUI detecta CUDA solo).
  - Sin GPU: instala PyTorch CPU y se lanza con el flag `--cpu`.

Este módulo solo hace detección por inspección (nvidia-smi / presencia del
driver) y expone la decisión como datos. No ejecuta instalación.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    """Resultado de la detección de GPU."""

    has_cuda: bool
    reason: str = ""

    @property
    def backend(self) -> str:
        """'cuda' o 'cpu' — el backend que debe usarse en la instalación."""
        return "cuda" if self.has_cuda else "cpu"


def _nvidia_smi_works() -> bool:
    """True si nvidia-smi está presente y reporta al menos una GPU."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode == 0 and "GPU" in proc.stdout
    except (subprocess.TimeoutExpired, OSError):
        return False


def detect_gpu() -> GpuInfo:
    """Detecta GPU NVIDIA dedicada con nvidia-smi como fuente primaria."""
    if _nvidia_smi_works():
        return GpuInfo(has_cuda=True, reason="nvidia-smi reporta GPU")

    # Fallback: el driver puede exponer /proc/driver/nvidia aunque no esté nvidia-smi.
    try:
        from pathlib import Path

        if Path("/proc/driver/nvidia").exists():
            return GpuInfo(has_cuda=True, reason="presencia de /proc/driver/nvidia")
    except OSError:
        pass

    return GpuInfo(has_cuda=False, reason="sin GPU NVIDIA detectada")


if __name__ == "__main__":
    info = detect_gpu()
    print(f"CUDA disponible : {info.has_cuda}")
    print(f"Backend         : {info.backend}")
    print(f"Razón           : {info.reason}")