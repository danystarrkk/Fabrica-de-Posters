"""Configuración de Hugging Face: instalación del cliente a nivel de sistema y persistencia segura del token.

Flux.2-dev es un modelo "gated" (requiere aceptar licencia/permisos), por lo
que la única descarga oficial es a través del cliente de Hugging Face
autenticado con un token (`HF_TOKEN`).

Estrategia:
  - Instalar `huggingface_hub` a nivel de sistema (pip install --user o global)
  - Usar el comando `hf` (CLI moderno) para login y descargas
  - El token se persiste en ~/.cache/huggingface/token (0600)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


class HuggingFaceSetupError(RuntimeError):
    """Se lanza cuando no se puede dejar el cliente HF listo o el token guardado."""


def get_token_from_env() -> str | None:
    """Lee el token de las variables de entorno conocidas, sin exponerlo."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Ejecuta comando y devuelve resultado."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise HuggingFaceSetupError(f"Comando falló: {' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc


def _pip_cmd() -> list[str]:
    """Comando pip del sistema (usa el pip del python actual)."""
    return [sys.executable, "-m", "pip"]


def install_huggingface_cli() -> None:
    """Instala `huggingface_hub` a nivel de sistema (--user para no requerir sudo)."""
    # Primero intenta --user (no requiere root)
    proc = _run([*_pip_cmd(), "install", "--upgrade", "--user", "huggingface_hub"], check=False)
    if proc.returncode != 0:
        # Fallback: instalación global (puede requerir sudo en algunos sistemas)
        proc = _run([*_pip_cmd(), "install", "--upgrade", "huggingface_hub"])
    # Verificar que el comando `hf` quedó disponible en PATH
    if not shutil.which("hf"):
        # A veces --user instala en ~/.local/bin que no está en PATH
        local_bin = Path.home() / ".local" / "bin"
        if (local_bin / "hf").exists():
            os.environ["PATH"] = f"{local_bin}{os.pathsep}{os.environ['PATH']}"
        else:
            raise HuggingFaceSetupError(
                "huggingface_hub se instaló pero el comando 'hf' no está en PATH. "
                "Asegúrate de que ~/.local/bin esté en tu PATH."
            )


def _hf_cmd() -> list[str]:
    """Devuelve el comando `hf` disponible en PATH."""
    if shutil.which("hf"):
        return ["hf"]
    # Fallback a ~/.local/bin/hf
    local_hf = Path.home() / ".local" / "bin" / "hf"
    if local_hf.exists():
        return [str(local_hf)]
    raise HuggingFaceSetupError("Comando 'hf' no encontrado en PATH.")


def persist_token(token: str) -> Path:
    """Guarda el token con `hf auth login` (no interactivo, sin log)."""
    cmd = _hf_cmd()
    proc = subprocess.run(
        [*cmd, "auth", "login", "--token", token, "--add-to-git-credential"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise HuggingFaceSetupError(
            f"No se pudo registrar el token: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return resolve_token_path()


def resolve_token_path() -> Path:
    """Devuelve la ruta del token que huggingface_hub persiste."""
    cache_token = Path.home() / ".cache" / "huggingface" / "token"
    legacy_token = Path.home() / ".huggingface" / "token"
    return cache_token if cache_token.exists() else legacy_token


def ensure_secure_permissions(token_path: Path | None = None) -> None:
    """Refuerza permisos 0600 sobre el token si existe (mejor práctica)."""
    token_path = token_path or resolve_token_path()
    if token_path.exists():
        token_path.chmod(0o600)


def setup(token: str | None = None) -> Path:
    """Pipeline preventivo completo: instala el cliente y persiste el token.

    Devuelve la ruta del token persistido. Lanza HuggingFaceSetupError si algo
    falla (sin exponer el token en el mensaje de error).
    """
    token = token or get_token_from_env()
    if not token:
        raise HuggingFaceSetupError(
            "No se encontró el token de Hugging Face. Define la variable de "
            "entorno HF_TOKEN o pásalo con --token."
        )

    install_huggingface_cli()
    token_path = persist_token(token)
    ensure_secure_permissions(token_path)
    return token_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Instala el cliente HF (hf) y persiste el token de forma segura."
    )
    parser.add_argument("--token", type=str, default=None, help="Token de Hugging Face (o usa HF_TOKEN).")
    args = parser.parse_args()

    try:
        token_path = setup(args.token)
        print(f"[OK] Token persistido de forma segura en: {token_path}")
    except HuggingFaceSetupError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)