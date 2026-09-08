"""Detección multiplataforma del sistema y del gestor de paquetes.

Este módulo NO ejecuta comandos con side-effects: solo inspecciona el sistema
para responder dos preguntas:
  1. ¿Qué familia de distribución corre? (arch / debian / rhel ...)
  2. ¿Qué gestor de paquetes hay que usar? (pacman / apt / dnf ...)

Se soportan las tres familias principales requeridas por el proyecto:
  - Arch Linux       -> pacman
  - Debian / Ubuntu  -> apt
  - Fedora / RHEL    -> dnf  (con fallback a yum para RHEL antiguos)

TODO se expone como datos (dataclasses), de modo que el resto de los scripts
tome decisiones en base a estos resultados sin acoplarse a strings sueltos.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PackageManager:
    """Un gestor de paquetes del sistema y los datos mínimos para usarlo."""

    name: str              # "pacman" | "apt" | "dnf" | "yum" | ...
    family: str            # "arch" | "debian" | "rhel"
    venv_package: str      # nombre del paquete que provee `python3 -m venv`

    def is_available(self) -> bool:
        return shutil.which(self.name) is not None

    def install(self, *packages: str, sudo: bool = True) -> subprocess.CompletedProcess:
        """Instala paquetes. Devuelve el proceso para que el llamador decida.

        Los comandos soportados son los mínimos no interactivos:
          - pacman: pacman -S --noconfirm <pkgs>
          - apt:    apt-get install -y <pkgs>   (requiere apt-get update previo)
          - dnf:    dnf install -y <pkgs>
        """
        raise NotImplementedError  # cubierto por el módulo package_manager


# Mapa familia -> gestores candidatos, en orden de preferencia.
_FAMILY_MANAGERS: dict[str, list[str]] = {
    "arch": ["pacman"],
    "debian": ["apt"],
    "rhel": ["dnf", "yum"],
}

# Nombre del paquete que provee `python3 -m venv` en cada gestor.
_VENV_PACKAGE: dict[str, str] = {
    "pacman": "python-virtualenv",
    "apt": "python3-venv",
    "dnf": "python3-virtualenv",
    "yum": "python3-virtualenv",
}


def _read_os_release() -> dict[str, str]:
    """Lee /etc/os-release y devuelve sus claves en minúscula, sin comillas."""
    data: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.exists():
        # Fallback: /usr/lib/os-release como provee systemd
        path = Path("/usr/lib/os-release")
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        data[key.lower()] = value.strip().strip('"').strip("'")
    return data


def detect_family() -> str:
    """Devuelve la familia de la distribución: arch | debian | rhel | unknown."""
    os_release = _read_os_release()
    ids: list[str] = []

    # id_like suele ser más confiable (ej. Ubuntu -> "debian").
    id_like = os_release.get("id_like", "")
    if id_like:
        ids.extend(id_like.lower().split())

    dist_id = os_release.get("id", "").lower()
    if dist_id:
        ids.append(dist_id)

    for candidate in ids:
        if candidate == "arch":
            return "arch"
        if candidate in ("debian", "ubuntu", "linuxmint", "pop", "kali"):
            return "debian"
        if candidate in ("fedora", "rhel", "centos", "rocky", "almalinux", "ol"):
            return "rhel"

    return "unknown"


def detect_manager(family: str | None = None) -> PackageManager | None:
    """Detecta el gestor de paquetes disponible para la familia indicada.

    Si no se pasa `family`, se detecta. Devuelve None si no hay gestor conocido.
    """
    family = family or detect_family()
    for name in _FAMILY_MANAGERS.get(family, []):
        if shutil.which(name):
            return PackageManager(
                name=name,
                family=family,
                venv_package=_VENV_PACKAGE.get(name, "python3-venv"),
            )
    return None


@dataclass
class SystemInfo:
    """Resumen del sistema que consumen el resto de los scripts."""

    family: str
    manager: PackageManager | None
    os_release: dict[str, str] = field(default_factory=dict)

    @classmethod
    def probe(cls) -> "SystemInfo":
        family = detect_family()
        return cls(
            family=family,
            manager=detect_manager(family),
            os_release=_read_os_release(),
        )

    def supports_unknown_family(self) -> bool:
        return self.manager is None


if __name__ == "__main__":
    info = SystemInfo.probe()
    print(f"Familia detectada : {info.family}")
    print(f"Gestor detectado  : {info.manager.name if info.manager else 'Ninguno'}")
    if info.manager:
        print(f"Paquete venv      : {info.manager.venv_package}")