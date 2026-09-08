"""Registro de los modelos fijos del proyecto y sus fuentes oficiales.

Los 4 modelos que el workflow `Generador de Posters.json` referencia de forma
fija y que deben quedar descargados y listos tras la instalación:

  Diffusion Model (UNET): flux2-dev.safetensors
  VAE                  : ae.safetensors
  Upscale Model        : 4xNomos8kDAT.pth
  CLIP (Text Encoder)  : mistral_3_small_flux2_fp8mixed.safetensors

Todos se descargan de Hugging Face (Flux.2 es gated, por eso el token).
Cada entrada declara:
  - repo_id:      repositorio de Hugging Face.
  - filename:     archivo dentro del repo (o ruta relativa en el repo para 4xNomos).
  - install_dir:  subcarpeta de ComfyUI donde debe quedar.
  - required:     True si es obligatorio para el flujo A2..A6.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Raíz de instalación de ComfyUI (estructura estándar).
COMFYUI_ROOT = Path.home() / "ComfyUI"


@dataclass(frozen=True)
class ModelSpec:
    """Definición de un modelo fijo a descargar."""

    key: str
    repo_id: str
    filename: str
    install_dir: str  # relativo a COMFYUI_ROOT
    required: bool = True

    @property
    def target_dir(self) -> Path:
        return COMFYUI_ROOT / self.install_dir

    @property
    def target_path(self) -> Path:
        return self.target_dir / self.filename


# Los IDs de repositorio oficiales. Los de Flux (unet/vae/clip) son gated.
MODELS: dict[str, ModelSpec] = {
    "unet": ModelSpec(
        key="unet",
        repo_id="black-forest-labs/FLUX.2-dev",
        filename="flux2-dev.safetensors",
        install_dir="models/unet",
        required=True,
    ),
    "vae": ModelSpec(
        key="vae",
        repo_id="black-forest-labs/FLUX.2-dev",
        filename="ae.safetensors",
        install_dir="models/vae",
        required=True,
    ),
    "clip": ModelSpec(
        key="clip",
        repo_id="yushan777/Flux.2-Dev",
        filename="mistral_3_small_flux2_fp8mixed.safetensors",
        install_dir="models/clip",
        required=True,
    ),
    "upscale": ModelSpec(
        key="upscale",
        repo_id="Maxivi/SDXLModels",
        filename="4xNomos8kDAT.pth",
        install_dir="models/upscale_models",
        required=True,
    ),
}


def get_all_models() -> list[ModelSpec]:
    return list(MODELS.values())


def get_required_models() -> list[ModelSpec]:
    return [m for m in MODELS.values() if m.required]


if __name__ == "__main__":
    for spec in get_all_models():
        print(f"[{spec.key}] {spec.repo_id}/{spec.filename} -> {spec.target_dir}")