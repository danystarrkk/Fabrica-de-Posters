"""Panel de Control AI — Orquestador principal.

Este módulo NO contiene lógica de negocio: solo presenta el menú y delega cada
opción a los scripts ubicados en `scripts/`. La lógica real vive en esos
módulos, importados aquí o ejecutados como proceso independiente.

Estructura (programación orientada a objetos):
  - Menu          -> abstrae la presentación y el bucle de selección.
  - Orchestrator  -> registra las acciones del panel de control.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable

# Ruta absoluta a la carpeta `scripts/`, para importarla o invocar sus módulos.
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


class Menu:
    """Menú interactivo reutilizable.

    `options` es un dict ordenado: {"tecla": ("Etiqueta", accion)}.
    `accion` puede ser None (vuelve al menú anterior) o un callable.
    """

    def __init__(self, title: str, options: dict[str, tuple[str, Callable | None]]):
        self.title = title
        self.options = options

    def display(self) -> None:
        print(f"\n=== {self.title} ===")
        for key, (label, _) in self.options.items():
            print(f"  {key}. {label}")
        print("===================")

    def run(self) -> None:
        while True:
            self.display()
            choice = input("Selecciona una opción: ").strip()

            if choice in self.options:
                _, action = self.options[choice]
                if action:
                    action()
                else:
                    break
            else:
                print("\n[!] Opción inválida. Intenta de nuevo.")


class Orchestrator:
    """Panel de control que delega cada acción a un script de `scripts/`."""

    def __init__(self, cli_args: argparse.Namespace | None = None) -> None:
        # Asegura que `scripts/` sea importable desde main.py.
        sys.path.insert(0, str(SCRIPTS_DIR))

        self.cli_args = cli_args or argparse.Namespace()

        self.main_menu = Menu(
            "Panel de Control AI",
            {
                "1": ("Instalar y configurar ComfyUI", self.install_comfy),
                "2": ("Instalar y configurar AI-Toolkit", self.install_toolkit),
                "3": ("Desplegar ComfyUI", self.deploy_comfy),
                "4": ("Desplegar AI-Toolkit", self.deploy_toolkit),
                "5": ("Salir", self.exit_app),
            },
        )

    # ------------------------------------------------------------- acciones
    def _run_script(self, module: str, *args: str) -> None:
        """Ejecuta un script de `scripts/` en un subproceso con el mismo intérprete."""
        script = SCRIPTS_DIR / module
        if not script.exists():
            print(f"\n[!] Script no encontrado: {script}")
            return
        subprocess.run([sys.executable, str(script), *args], check=False)

    def _comfy_root(self) -> Path:
        return Path(getattr(self.cli_args, "root", None) or Path.home() / "ComfyUI")

    def _token(self) -> str | None:
        # Prioridad: CLI arg > env var (se resuelve en install_comfyui)
        return getattr(self.cli_args, "token", None)

    def install_comfy(self) -> None:
        print("\n[+] Instalando y configurando ComfyUI...")
        args = ["--root", str(self._comfy_root())]
        token = self._token()
        if token:
            args += ["--token", token]
        self._run_script("install_comfyui.py", *args)

    def install_toolkit(self) -> None:
        print("\n[+] Instalando y configurando AI-Toolkit... (pendiente)")
        # self._run_script("install_toolkit.py")
        print("  Aún sin implementación.")

    def deploy_comfy(self) -> None:
        print("\n[+] Desplegando ComfyUI... (pendiente)")
        # self._run_script("comfy_orchestrator.py", "--prompts", ...)
        print("  Aún sin implementación.")

    def deploy_toolkit(self) -> None:
        print("\n[+] Desplegando AI-Toolkit... (pendiente)")
        print("  Aún sin implementación.")

    def exit_app(self) -> None:
        print("\nSaliendo del sistema...")
        sys.exit(0)

    # --------------------------------------------------------------- arranque
    def start(self) -> None:
        try:
            self.main_menu.run()
        except KeyboardInterrupt:
            self.exit_app()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Panel de Control AI — Fábrica de Posters",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Ruta de instalación de ComfyUI (default ~/ComfyUI).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token de Hugging Face (alternativa a HF_TOKEN).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    Orchestrator(parse_args()).start()