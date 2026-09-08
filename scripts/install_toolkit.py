"""Instalador de AI-Toolkit (kohya-ss/sd-scripts) — orquesta la instalación completa.

Pasos:
  1. Clonar el repositorio de sd-scripts en TOOLKIT_ROOT (~/ai-toolkit).
  2. Asegurar el soporte de venv (instalar python3-venv / python3-virtualenv
     con el gestor detectado: pacman / apt / dnf).
  3. Crear y activar el entorno virtual.
  4. Instalar las dependencias de sd-scripts (requirements.txt) dentro del venv,
     eligiendo el índice de PyTorch CUDA o CPU según la GPU detectada.
  5. Instalar dependencias adicionales: xformers, bitsandbytes, etc.
  6. (Opcional) Configurar Hugging Face token para modelos gated.
  7. Verificar instalación con un test rápido.

Estructura paralela a install_comfyui.py para consistencia.
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
import huggingface_setup as hf  # noqa: E402

# Raíz de instalación de AI-Toolkit (sd-scripts)
TOOLKIT_ROOT = Path.home() / "ai-toolkit"
SD_SCRIPTS_REPO = "https://github.com/kohya-ss/sd-scripts.git"


class ToolkitInstaller:
    """Encapsula el proceso completo de instalación de AI-Toolkit (sd-scripts)."""

    def __init__(self, root: Path = TOOLKIT_ROOT, token: str | None = None):
        self.root = root
        self.token = token or hf.get_token_from_env()
        self.system = SystemInfo.probe()
        self.venv_python = self.root / ".venv" / "bin" / "python"

    # ------------------------------------------------------------------ util
    def _run(self, cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
        """Ejecuta comando capturando salida (para pasos donde no queremos ver ruido)."""
        print(f"  $ {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"Comando falló ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
            )
        return proc

    def _run_streaming(self, cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
        """Ejecuta comando SIN capturar salida (stdout/stderr van directo a terminal)."""
        print(f"  $ {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"Comando falló ({proc.returncode}): {' '.join(cmd)}"
            )
        return proc

    def _venv_pip(self) -> list[str]:
        return [str(self.venv_python), "-m", "pip"]

    # --------------------------------------------------------------- paso 1
    def clone_repo(self) -> None:
        print("\n[1/7] Clonando AI-Toolkit (sd-scripts)...")
        if (self.root / ".git").exists():
            print(f"  sd-scripts ya clonado en {self.root}. Saltando.")
            return
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self._run(["git", "clone", SD_SCRIPTS_REPO, str(self.root)])

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
    def install_sd_scripts_deps(self) -> None:
        print("\n[4/7] Instalando dependencias de sd-scripts...")
        gpu = detect_gpu()
        req_path = self.root / "requirements.txt"
        if not req_path.exists():
            raise RuntimeError("No se encontró requirements.txt en sd-scripts.")

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
    def install_extra_deps(self) -> None:
        print("\n[5/7] Instalando dependencias extra (xformers, bitsandbytes, accelerate)...")
        gpu = detect_gpu()
        
        # xformers - solo si hay GPU CUDA
        if gpu.has_cuda:
            print("  Instalando xformers (CUDA)...")
            self._run(
                [
                    *self._venv_pip(),
                    "install",
                    "xformers",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu124",
                ]
            )
        else:
            print("  Saltando xformers (requiere GPU CUDA).")
        
        # bitsandbytes (para 8-bit optimization)
        print("  Instalando bitsandbytes...")
        self._run([*self._venv_pip(), "install", "bitsandbytes"])
        
        # accelerate, peft, transformers (ya pueden venir en requirements.txt)
        print("  Instalando accelerate, peft, transformers, huggingface_hub...")
        self._run([*self._venv_pip(), "install", "accelerate", "peft", "transformers", "huggingface_hub"])

    # --------------------------------------------------------------- paso 6
    def setup_huggingface(self) -> None:
        print("\n[6/7] Configurando cliente de Hugging Face (token)...")
        if not self.token:
            print("  Sin token HF_TOKEN. Saltando (modelos gated no descargables).")
            return
        token_path = hf.setup(self.token)
        print(f"  Token persistido en {token_path} (0600).")

    # --------------------------------------------------------------- paso 7
    def verify_installation(self) -> None:
        print("\n[7/7] Verificando instalación...")
        # Test rápido: importar módulos clave
        test_script = """
import torch
import accelerate
import peft
import transformers
import bitsandbytes
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
print(f'accelerate: {accelerate.__version__}')
print(f'peft: {peft.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'bitsandbytes: {bitsandbytes.__version__}')
print('✓ Todos los módulos importados correctamente')
"""
        self._run_streaming([str(self.venv_python), "-c", test_script])

    # --------------------------------------------------------- orchestration
    def run(self) -> None:
        print(f"Instalando AI-Toolkit (sd-scripts) en: {self.root}")
        self.clone_repo()
        self.ensure_venv_support()
        self.create_venv()
        self.install_sd_scripts_deps()
        self.install_extra_deps()
        self.setup_huggingface()
        self.verify_installation()
        print("\n[ÉXITO] AI-Toolkit (sd-scripts) instalado y verificado.")
        print(f"  Para usarlo: source {self.root}/.venv/bin/activate")
        print(f"  Scripts disponibles en: {self.root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instala y configura AI-Toolkit (kohya-ss/sd-scripts)."
    )
    parser.add_argument("--root", type=Path, default=TOOLKIT_ROOT,
                        help="Ruta de instalación (default ~/ai-toolkit).")
    parser.add_argument("--token", type=str, default=None,
                        help="Token de Hugging Face (o usa HF_TOKEN).")
    args = parser.parse_args()

    try:
        installer = ToolkitInstaller(root=args.root, token=args.token)
        installer.run()
    except (RuntimeError, hf.HuggingFaceSetupError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()