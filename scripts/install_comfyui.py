"""Instalador de ComfyUI — orquesta los 7 pasos para dejar el sistema listo.

Pasos:
  1. Clonar el repositorio de ComfyUI en COMFYUI_ROOT (~/ComfyUI).
  2. Asegurar el soporte de venv (instalar python3-venv / python3-virtualenv
     con el gestor detectado: pacman / apt / dnf).
  3. Crear y activar el entorno virtual.
  4. Instalar las dependencias de ComfyUI (requirements.txt) dentro del venv,
     eligiendo el índice de PyTorch CUDA o CPU según la GPU detectada.
  5. Clonar los custom nodes (ComfyUI-Manager y UltimateSDUpscale) e instalar
     sus dependencias.
  6. (Preventivo) Instalar el cliente de Hugging Face y persistir el token.
  7. Descargar los 4 modelos fijos a sus carpetas correspondientes.

Cada paso es idempotente en la medida de lo posible (no reclona si ya existe,
no reinstala venv si ya existe, no redescarga modelos ya presentes).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Permite ejecutar el script desde cualquier cwd (imports relativos al módulo).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from system_info import SystemInfo  # noqa: E402
import package_manager as pm  # noqa: E402
from gpu import detect_gpu  # noqa: E402
from models import COMFYUI_ROOT, get_required_models  # noqa: E402
import huggingface_setup as hf  # noqa: E402

COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"

CUSTOM_NODES = [
    {
        "repo": "https://github.com/ltdrdata/ComfyUI-Manager.git",
        "dir": "ComfyUI-Manager",
    },
    {
        "repo": "https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git",
        "dir": "ComfyUI_UltimateSDUpscale",
    },
]


class ComfyUIInstaller:
    """Encapsula el proceso completo de instalación de ComfyUI."""

    def __init__(self, root: Path = COMFYUI_ROOT, token: str | None = None):
        self.root = root
        self.token = token or hf.get_token_from_env()
        self.system = SystemInfo.probe()
        self.venv_python = self.root / ".venv" / "bin" / "python"

    # ------------------------------------------------------------------ util
    def _run(self, cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
        print(f"  $ {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"Comando falló ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
            )
        return proc

    def _venv_pip(self) -> list[str]:
        return [str(self.venv_python), "-m", "pip"]

    # --------------------------------------------------------------- paso 1
    def clone_repo(self) -> None:
        print("\n[1/7] Clonando ComfyUI...")
        if (self.root / ".git").exists():
            print(f"  ComfyUI ya clonado en {self.root}. Saltando.")
            return
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self._run(["git", "clone", COMFYUI_REPO, str(self.root)])

    # --------------------------------------------------------------- paso 2
    def ensure_venv_support(self) -> None:
        print("\n[2/7] Verificando soporte de virtualenv...")
        if self._venv_works():
            print("  python3 -m venv ya funciona. Saltando.")
            return

        if not self.system.manager:
            raise RuntimeError(
                f"No se detectó un gestor de paquetes soportado (familia: {self.system.family}). "
                "Instala el soporte de venv manualmente."
            )

        pkg = self.system.manager.venv_package
        print(f"  Instalando '{pkg}' con {self.system.manager.name}...")
        result = pm.install(self.system.manager, [pkg])
        if not result.ok:
            raise RuntimeError(f"Fallo instalando {pkg}: {result.stderr.strip()}")

    def _venv_works(self) -> bool:
        try:
            return subprocess.run(
                [sys.executable, "-m", "venv", "--help"],
                capture_output=True,
            ).returncode == 0
        except OSError:
            return False

    # --------------------------------------------------------------- paso 3
    def create_venv(self) -> None:
        print("\n[3/7] Creando entorno virtual...")
        if self.venv_python.exists():
            print("  El venv ya existe. Saltando.")
            return
        self._run([sys.executable, "-m", "venv", str(self.root / ".venv")])
        # Asegurar pip actualizado dentro del venv
        self._run([*self._venv_pip(), "install", "--upgrade", "pip"])

    # --------------------------------------------------------------- paso 4
    def install_comfy_deps(self) -> None:
        print("\n[4/7] Instalando dependencias de ComfyUI...")
        gpu = detect_gpu()
        req_path = self.root / "requirements.txt"
        if not req_path.exists():
            raise RuntimeError("No se encontró requirements.txt en ComfyUI.")

        self._install_torch(gpu)
        self._run([*self._venv_pip(), "install", "-r", str(req_path)])

    def _install_torch(self, gpu) -> None:
        if gpu.has_cuda:
            print("  GPU NVIDIA detectada -> PyTorch con índice CUDA (cu124).")
            self._run(
                [
                    *self._venv_pip(),
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu124",
                ]
            )
        else:
            print("  Sin GPU -> PyTorch CPU.")
            self._run(
                [
                    *self._venv_pip(),
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                ]
            )

    # --------------------------------------------------------------- paso 5
    def clone_custom_nodes(self) -> None:
        print("\n[5/7] Instalando custom nodes...")
        custom_dir = self.root / "custom_nodes"
        custom_dir.mkdir(parents=True, exist_ok=True)

        for node in CUSTOM_NODES:
            dest = custom_dir / node["dir"]
            if (dest / ".git").exists():
                print(f"  {node['dir']} ya clonado. Saltando.")
            else:
                print(f"  Clonando {node['dir']}...")
                self._run(["git", "clone", node["repo"], str(dest)])

            req = dest / "requirements.txt"
            if req.exists():
                print(f"  Instalando dependencias de {node['dir']}...")
                self._run([*self._venv_pip(), "install", "-r", str(req)])

    # --------------------------------------------------------------- paso 6
    def setup_huggingface(self) -> None:
        print("\n[6/7] Configurando cliente de Hugging Face (token)...")
        if not self.token:
            raise RuntimeError(
                "No se encontró HF_TOKEN. Define la variable o pásalo con --token. "
                "Flux.2-dev es gated y requiere autenticación."
            )
        token_path = hf.setup(self.token, str(self.venv_python))
        print(f"  Token persistido en {token_path} (0600).")

    # --------------------------------------------------------------- paso 7
    def download_models(self) -> None:
        print("\n[7/7] Descargando los 4 modelos fijos...")
        from huggingface_setup import _find_huggingface_cli

        cli_cmd = _find_huggingface_cli(str(self.venv_python))
        for spec in get_required_models():
            if spec.target_path.exists():
                print(f"  [{spec.key}] ya presente. Saltando.")
                continue
            spec.target_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Descargando {spec.key}: {spec.repo_id}/{spec.filename}")
            self._run(
                [
                    *cli_cmd,
                    "download",
                    spec.repo_id,
                    spec.filename,
                    "--local-dir",
                    str(spec.target_dir),
                ]
            )

    # --------------------------------------------------------- orchestration
    def run(self) -> None:
        print(f"Instalando ComfyUI en: {self.root}")
        self.clone_repo()
        self.ensure_venv_support()
        self.create_venv()
        self.install_comfy_deps()
        self.clone_custom_nodes()
        self.setup_huggingface()
        self.download_models()
        print("\n[ÉXITO] ComfyUI instalado y configurado. Necesita reiniciarse.")
        print("  Para lanzarlo: "
              + ("python3 main.py" if detect_gpu().has_cuda
                 else "python3 main.py --cpu"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instala y configura ComfyUI con sus modelos fijos."
    )
    parser.add_argument("--root", type=Path, default=COMFYUI_ROOT,
                        help="Ruta de instalación de ComfyUI (default ~/ComfyUI).")
    parser.add_argument("--token", type=str, default=None,
                        help="Token de Hugging Face (o usa HF_TOKEN).")
    args = parser.parse_args()

    try:
        installer = ComfyUIInstaller(root=args.root, token=args.token)
        installer.run()
    except (RuntimeError, hf.HuggingFaceSetupError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()