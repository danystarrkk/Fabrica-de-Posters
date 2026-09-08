"""Wrapper seguro para instalar paquetes con el gestor del sistema.

Separa la responsabilidad de "detectar" (system_info.py) de la de "ejecutar"
(este módulo). Nada más importa este módulo para instalar paquetes del SO.

Los comandos se construyen de forma explícita por gestor (sin interpolación de
strings del usuario) para evitar roturas y para que quede claro qué se ejecuta.

Se usa `list[str]` en vez de strings concatenados porque subprocess no pasa por
una shell: así evitamos problemas de quoting e inyección.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

from system_info import PackageManager


# Tiempos/estados que compartimos con el instalador principal.
class PackageInstallError(RuntimeError):
    """Se lanza cuando el gestor de paquetes falla la instalación."""


@dataclass
class InstallResult:
    """Resultado de una operación de instalación de paquetes."""

    manager: str
    packages: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _run(cmd: list[str], sudo: bool) -> subprocess.CompletedProcess:
    if sudo and not _running_as_root():
        cmd = ["sudo", *cmd]
    return subprocess.run(cmd, capture_output=True, text=True)


def _running_as_root() -> bool:
    try:
        return __import__("os").geteuid() == 0
    except AttributeError:  # en Windows no existe geteuid; no es nuestro caso
        return False


def _command_for(manager: PackageManager, packages: list[str]) -> list[str]:
    """Construye el comando exacto para cada gestor, sin shell."""
    name = manager.name
    if name == "pacman":
        return ["pacman", "-S", "--noconfirm", *packages]
    if name == "apt":
        return ["apt-get", "install", "-y", *packages]
    if name in ("dnf", "yum"):
        return [name, "install", "-y", *packages]
    raise PackageInstallError(f"Gestor de paquetes no soportado: {name}")


def update_indexes(manager: PackageManager, sudo: bool = True) -> InstallResult | None:
    """Actualiza los índices cuando el gestor lo requiere (apt)."""
    name = manager.name
    if name == "apt":
        proc = _run(["apt-get", "update"], sudo)
        return InstallResult(name, ("update",), proc.returncode, proc.stdout, proc.stderr)
    # pacman y dnf refrescan índices de forma inherente; nada que hacer.
    return None


def install(manager: PackageManager, packages: list[str], sudo: bool = True) -> InstallResult:
    """Instala una lista de paquetes y devuelve el resultado sin lanzar.

    Si `update_indexes` falla (apt) se lanza PackageInstallError para no
    continuar en un estado inconsistente.
    """
    if not packages:
        return InstallResult(manager.name, (), 0, "", "")

    update_result = update_indexes(manager, sudo)
    if update_result and not update_result.ok:
        raise PackageInstallError(
            f"Fallo actualizando índices de {manager.name}: {update_result.stderr.strip()}"
        )

    proc = _run(_command_for(manager, packages), sudo)
    return InstallResult(manager.name, tuple(packages), proc.returncode, proc.stdout, proc.stderr)


if __name__ == "__main__":
    # Smoke test: detecta el gestor y muestra qué comandos ejecutaría (dry-run).
    from system_info import SystemInfo

    info = SystemInfo.probe()
    if not info.manager:
        print("No se detectó gestor de paquetes soportado.")
        sys.exit(1)

    print(f"Gestor: {info.manager.name}")
    print(f"Comando para (python3-venv): "
          f"{_command_for(info.manager, [info.manager.venv_package])}")