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
        """Ejecuta comando capturando salida (para pasos donde no queremos ver ruido)."""
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
        token_path = hf.setup(self.token)
        print(f"  Token persistido en {token_path} (0600).")

    # --------------------------------------------------------------- paso 7
    def download_models(self) -> None:
        print("\n[7/7] Descargando los 4 modelos fijos...")
        from huggingface_setup import _hf_cmd

        hf_cmd = _hf_cmd()
        for spec in get_required_models():
            # Verificar estado del archivo
            needs_download, reason = self._check_model_status(spec)
            
            if not needs_download:
                size_mb = spec.target_path.stat().st_size / (1024**2)
                print(f"  [{spec.key}] ya presente ({size_mb:.0f} MB). Saltando.")
                continue

            # Mostrar info y preguntar al usuario
            expected_gb = spec.expected_size_bytes / (1024**3)
            if spec.target_path.exists():
                current_mb = spec.target_path.stat().st_size / (1024**2)
                print(f"\n  [{spec.key}] {spec.filename}")
                print(f"    Estado: {reason}")
                print(f"    Tamaño actual: {current_mb:.0f} MB | Esperado: ~{expected_gb:.1f} GB")
            else:
                print(f"\n  [{spec.key}] {spec.filename}")
                print(f"    Estado: {reason}")
                print(f"    Tamaño esperado: ~{expected_gb:.1f} GB")

            # Prompt interactivo
            while True:
                choice = input(f"    ¿Descargar este modelo? [s/N]: ").strip().lower()
                if choice in ('s', 'si', 'sí', 'y', 'yes'):
                    break
                elif choice in ('n', 'no', ''):
                    print(f"    Saltando {spec.key}.")
                    break
                else:
                    print(f"    Respuesta no válida. Use 's' para sí o 'n' para no.")

            if choice in ('n', 'no', ''):
                continue  # Pasar al siguiente modelo

            # Preparar directorio y limpiar locks
            spec.target_dir.mkdir(parents=True, exist_ok=True)
            self._clean_hf_locks(spec.target_dir)

            # Si existe archivo incompleto, eliminarlo
            if spec.target_path.exists():
                try:
                    spec.target_path.unlink()
                except OSError as e:
                    print(f"  [WARN] No se pudo eliminar archivo previo: {e}")

            # Descargar con streaming
            cmd = [
                *hf_cmd,
                "download",
                spec.repo_id,
                "--include",
                spec.filename,
                "--local-dir",
                str(spec.target_dir),
            ]
            print(f"  $ {' '.join(cmd)}")
            
            try:
                self._run_download_streaming(cmd)
            except RuntimeError as e:
                print(f"  [ERROR] Falló la descarga de {spec.key}: {e}")
                continue

            # Verificar tamaño tras descarga
            if spec.target_path.exists():
                actual_size = spec.target_path.stat().st_size
                if spec.is_valid_size(actual_size):
                    size_mb = actual_size / (1024**2)
                    print(f"  [{spec.key}] verificado OK ({size_mb:.0f} MB)")
                else:
                    size_mb = actual_size / (1024**2)
                    print(f"  [WARN] {spec.key}: tamaño inesperado ({size_mb:.0f} MB), esperado ~{spec.expected_size_bytes/(1024**2):.0f} MB")

    def _check_model_status(self, spec) -> tuple[bool, str]:
        """Verifica si un modelo necesita descarga.
        
        Returns:
            (needs_download: bool, reason: str)
        """
        if not spec.target_path.exists():
            return True, "archivo no existe"
        
        actual_size = spec.target_path.stat().st_size
        if not spec.is_valid_size(actual_size):
            if actual_size < spec.expected_size_bytes * 0.1:
                return True, "archivo muy pequeño (posible descarga corrupta)"
            else:
                return True, "archivo incompleto"
        
        return False, "completo y verificado"

    def _run_download_streaming(self, cmd: list[str]) -> None:
        """Ejecuta descarga con streaming de salida preservando barras de progreso (\r).
        
        Lee stdout en bruto (bytes) y escribe directo a sys.stdout.buffer para que
        los carriage returns (\r) de la barra de progreso funcionen correctamente.
        """
        import sys
        
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 bufsize=0)  # sin buffer para tiempo real
        
        try:
            # Leer byte a byte y escribir directo a stdout.buffer
            while True:
                chunk = proc.stdout.read(1024)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            
            returncode = proc.wait()
            if returncode != 0:
                raise RuntimeError(f"Descarga falló (exit code {returncode})")
        except KeyboardInterrupt:
            sys.stdout.write("\n  [INTERRUMPIDO] Descarga cancelada por usuario.\n")
            sys.stdout.flush()
            proc.send_signal(subprocess.signal.SIGINT)
            proc.wait()
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)

    def _clean_hf_locks(self, target_dir: Path) -> None:
        """Elimina archivos .lock stale en el cache de HF dentro de target_dir."""
        cache_dir = target_dir / ".cache" / "huggingface" / "download"
        if cache_dir.exists():
            for lock_file in cache_dir.rglob("*.lock"):
                try:
                    lock_file.unlink()
                    print(f"  [cleanup] Lock stale eliminado: {lock_file.name}")
                except OSError:
                    pass

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