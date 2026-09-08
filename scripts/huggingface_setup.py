"""Configuración de Hugging Face: instalación del cliente y persistencia segura del token.

Flux.2-dev es un modelo "gated" (requiere aceptar licencia/permisos), por lo
que la única descarga oficial es a través del cliente de Hugging Face
autenticado con un token (`HF_TOKEN`).

Requisito preventivo (lo llamamos "paso preventivo"): antes de descargar los
modelos, hay que:
  1. Instalar el cliente `huggingface_hub` (provee `hf` / `huggingface-cli`).
  2. Registrar el token de forma segura y persistente.

El token se acepta de dos fuentes (en orden):
  - Variable de entorno `HF_TOKEN`
  - Argumento `--token`

Se persiste mediante `huggingface_hub` en la ruta que la librería reconoce
(~/.cache/huggingface/token o ~/.huggingface/token), con permisos 0600, para
que las futuras descargas no vuelvan a pedirlo.
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


def _pip_cmd(python: str | None) -> list[str]:
    """Devuelve el comando pip como lista de partes para subprocess."""
    if python:
        python_path = Path(python).resolve()
        if not python_path.exists():
            raise HuggingFaceSetupError(f"Python del venv no encontrado: {python_path}")
        pip_path = python_path.parent / "pip"
        if pip_path.exists():
            return [str(pip_path)]
        # Fallback: python -m pip
        return [str(python_path), "-m", "pip"]
    # Sin python específico: usar el pip del sistema actual
    return [sys.executable, "-m", "pip"]


def _resolve_venv_bin(python: str | None, bin_name: str) -> Path | None:
    """Resuelve la ruta a un binario dentro del venv, si python apunta al venv."""
    if not python:
        return None
    try:
        venv_bin = Path(python).resolve().parent / bin_name
        return venv_bin if venv_bin.exists() else None
    except (OSError, ValueError):
        return None


def _find_huggingface_cli(python: str | None) -> list[str]:
    """Devuelve el comando para invocar huggingface-cli de forma robusta.

    Orden de preferencia:
      1. Binario directo en el venv (huggingface-cli) - LO QUE PIP INSTALA
      2. python -m huggingface_hub (versiones nuevas con __main__)
      3. python -m huggingface_hub.cli (versiones intermedias)
    """
    # 1. PRIORIDAD MÁXIMA: binario que pip instala en el venv
    cli_path = _resolve_venv_bin(python, "huggingface-cli")
    if cli_path:
        return [str(cli_path)]

    # Si tenemos python del venv, intentamos invocar el módulo de varias formas
    if python:
        # 2. Versiones nuevas: huggingface_hub tiene __main__
        probe = subprocess.run(
            [python, "-m", "huggingface_hub", "--help"],
            capture_output=True,
        )
        if probe.returncode == 0:
            return [python, "-m", "huggingface_hub"]

        # 3. Versiones intermedias: huggingface_hub.cli como módulo ejecutable
        probe = subprocess.run(
            [python, "-m", "huggingface_hub.cli", "--help"],
            capture_output=True,
        )
        if probe.returncode == 0:
            return [python, "-m", "huggingface_hub.cli"]

    # 4. Último recurso: confiar en PATH (solo si no hay venv)
    if shutil.which("huggingface-cli"):
        return ["huggingface-cli"]

    raise HuggingFaceSetupError(
        "No se encontró huggingface-cli. Asegúrate de instalar huggingface_hub en el venv."
    )


def install_huggingface_cli(python: str | None = None) -> None:
    """Instala `huggingface_hub` con pip dentro del entorno indicado y verifica el binario."""
    pip_cmd = _pip_cmd(python)
    
    # Forzar reinstalación para asegurar que los entry points se creen
    proc = subprocess.run(
        [*pip_cmd, "install", "--force-reinstall", "--no-deps", "huggingface_hub"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # Fallback: instalación normal si falla --force-reinstall
        proc = subprocess.run(
            [*pip_cmd, "install", "--upgrade", "huggingface_hub"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise HuggingFaceSetupError(
                f"Fallo instalando huggingface_hub: {proc.stderr.strip()}"
            )
    
    # VERIFICAR que el binario quedó creado en el venv
    if python:
        cli_path = _resolve_venv_bin(python, "huggingface-cli")
        if not cli_path:
            # Intentar regenerar entry points
            subprocess.run(
                [*pip_cmd, "install", "--force-reinstall", "huggingface_hub"],
                capture_output=True,
            )
            cli_path = _resolve_venv_bin(python, "huggingface-cli")
            if not cli_path:
                raise HuggingFaceSetupError(
                    "huggingface_hub se instaló pero NO se creó el binario huggingface-cli en el venv. "
                    "Verifica que pip esté instalando en el venv correcto."
                )


def persist_token(token: str, python: str | None = None) -> Path:
    """Guarda el token con `huggingface-cli login` (no pasa por shell, sin log).

    Devuelve la ruta del archivo de token persistido.
    Usa el flag `--add-to-git-credential` para que el token quede disponible
    de forma transparente para `hf download` sin reautenticación.
    """
    cli_cmd = _find_huggingface_cli(python)
    proc = subprocess.run(
        [*cli_cmd, "login", "--token", token, "--add-to-git-credential"],
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


def setup(token: str | None = None, python: str | None = None) -> Path:
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

    install_huggingface_cli(python)
    token_path = persist_token(token, python)
    ensure_secure_permissions(token_path)
    return token_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Instala el cliente HF y persiste el token de forma segura."
    )
    parser.add_argument("--token", type=str, default=None, help="Token de Hugging Face (o usa HF_TOKEN).")
    parser.add_argument("--python", type=str, default=None, help="Python del venv (ruta al binario).")
    args = parser.parse_args()

    try:
        token_path = setup(args.token, args.python)
        print(f"[OK] Token persistido de forma segura en: {token_path}")
    except HuggingFaceSetupError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)