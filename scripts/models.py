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
    expected_size_bytes: int  # tamaño aproximado esperado
    required: bool = True

    @property
    def target_dir(self) -> Path:
        return COMFYUI_ROOT / self.install_dir

    @property
    def target_path(self) -> Path:
        return self.target_dir / self.filename

    def is_valid_size(self, actual_size: int) -> bool:
        """Verifica si el tamaño actual está dentro del umbral de tolerancia.
        
        Tolerancia:
        - Archivos en GB: -2 GB del tamaño esperado
        - Archivos en MB: -5 MB del tamaño esperado
        """
        if self.expected_size_bytes >= 1024**3:  # >= 1 GB
            threshold = self.expected_size_bytes - 2 * 1024**3  # -2 GB
        else:
            threshold = self.expected_size_bytes - 5 * 1024**2  # -5 MB
        return actual_size >= threshold

    def should_download(self) -> bool:
        """True si el archivo no existe o es demasiado pequeño (incompleto)."""
        if not self.target_path.exists():
            return True
        try:
            actual_size = self.target_path.stat().st_size
            return not self.is_valid_size(actual_size)
        except OSError:
            return True


# Los IDs de repositorio oficiales. Los de Flux (unet/vae/clip) son gated.
# Tamaños aproximados (basados en releases oficiales):
# - flux2-dev.safetensors: ~23.8 GB
# - ae.safetensors: ~335 MB  
# - mistral_3_small_flux2_fp8mixed.safetensors: ~5.5 GB
# - 4xNomos8kDAT.pth: ~167 MB
MODELS: dict[str, ModelSpec] = {
    "unet": ModelSpec(
        key="unet",
        repo_id="black-forest-labs/FLUX.2-dev",
        filename="flux2-dev.safetensors",
        install_dir="models/unet",
        expected_size_bytes=24 * 1024**3,  # ~24 GB -> threshold 22 GB
        required=True,
    ),
    "vae": ModelSpec(
        key="vae",
        repo_id="black-forest-labs/FLUX.2-dev",
        filename="ae.safetensors",
        install_dir="models/vae",
        expected_size_bytes=335 * 1024**2,  # ~335 MB -> threshold 330 MB
        required=True,
    ),
    "clip": ModelSpec(
        key="clip",
        repo_id="yushan777/Flux.2-Dev",
        filename="mistral_3_small_flux2_fp8mixed.safetensors",
        install_dir="models/clip",
        expected_size_bytes=5_500 * 1024**2,  # ~5.5 GB -> threshold 3.5 GB
        required=True,
    ),
    "upscale": ModelSpec(
        key="upscale",
        repo_id="Maxivi/SDXLModels",
        filename="4xNomos8kDAT.pth",
        install_dir="models/upscale_models",
        expected_size_bytes=167 * 1024**2,  # ~167 MB -> threshold 162 MB
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