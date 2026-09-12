import json
from pathlib import Path

workflow_path = Path("/mnt/data/Generador de Posters.json")
workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

script_path = Path("/mnt/data/generate_posters_comfy.py")

script = f'''#!/usr/bin/env python3
"""
Generador masivo de posters para ComfyUI.

Uso:
    python3 generate_posters_comfy.py prompts.json
    python3 generate_posters_comfy.py prompts.json --url http://127.0.0.1:8188

El workflow de ComfyUI está hardcodeado en WORKFLOW.
Para cada prompt del JSON se generan cinco salidas:
    poster<ID>_A2
    poster<ID>_A3
    poster<ID>_A4
    poster<ID>_A5
    poster<ID>_A6

El prompt se coloca en el nodo CLIPTextEncode positivo (nodo 8).
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import requests


# ============================================================
# CONFIGURACIÓN
# ============================================================

DEFAULT_COMFY_URL = "http://127.0.0.1:8188"

# Workflow exportado desde ComfyUI.
# Se modifica únicamente:
#   - nodo 8: prompt
#   - nodos 17, 20, 24, 26, 29: filename_prefix
WORKFLOW = {json.dumps(workflow, ensure_ascii=False, indent=4)}


# ============================================================
# FUNCIONES
# ============================================================

def load_prompts(path: Path):
    """Carga y valida prompts.json."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {{path}}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        prompts = data.get("prompts")
    elif isinstance(data, list):
        prompts = data
    else:
        prompts = None

    if not isinstance(prompts, list):
        raise ValueError(
            "El JSON debe contener una lista 'prompts'."
        )

    result = []

    for index, item in enumerate(prompts, start=1):
        if isinstance(item, dict):
            prompt_id = item.get("id", index)
            text = item.get("prompt")
        elif isinstance(item, str):
            prompt_id = index
            text = item
        else:
            raise ValueError(
                f"Elemento #{{index}} del JSON no es válido."
            )

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"El prompt #{{index}} está vacío o no es texto."
            )

        result.append((int(prompt_id), text.strip()))

    return result


def build_workflow(prompt_text: str, poster_id: int):
    """
    Crea una copia independiente del workflow para un poster.
    No modifica el workflow base.
    """
    workflow = json.loads(json.dumps(WORKFLOW))

    # Prompt positivo.
    workflow["8"]["inputs"]["text"] = prompt_text

    # Cinco salidas del workflow.
    workflow["17"]["inputs"]["filename_prefix"] = f"poster{{poster_id}}_A2"
    workflow["20"]["inputs"]["filename_prefix"] = f"poster{{poster_id}}_A3"
    workflow["24"]["inputs"]["filename_prefix"] = f"poster{{poster_id}}_A4"
    workflow["26"]["inputs"]["filename_prefix"] = f"poster{{poster_id}}_A5"
    workflow["29"]["inputs"]["filename_prefix"] = f"poster{{poster_id}}_A6"

    return workflow


def queue_prompt(comfy_url: str, workflow: dict):
    """Envía un workflow a /prompt de ComfyUI."""
    client_id = str(uuid.uuid4())

    payload = {{
        "prompt": workflow,
        "client_id": client_id,
    }}

    response = requests.post(
        f"{{comfy_url.rstrip('/')}}/prompt",
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    if "prompt_id" not in data:
        raise RuntimeError(
            f"ComfyUI no devolvió prompt_id: {{data}}"
        )

    return data["prompt_id"]


def main():
    parser = argparse.ArgumentParser(
        description="Genera posters desde prompts.json usando ComfyUI."
    )

    parser.add_argument(
        "json_file",
        type=Path,
        help="Archivo JSON generado por el generador de prompts.",
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_COMFY_URL,
        help=f"URL de ComfyUI. Default: {{DEFAULT_COMFY_URL}}",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Espera entre envíos a ComfyUI en segundos. Default: 0.5",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="ID desde el que comenzar.",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="ID máximo a procesar.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Comprobar ComfyUI
    # --------------------------------------------------------

    print("=" * 70)
    print("       T3CHGRNG POSTER GENERATOR → COMFYUI")
    print("=" * 70)

    print(f"ComfyUI: {{args.url}}")
    print(f"JSON:    {{args.json_file}}")

    try:
        response = requests.get(
            f"{{args.url.rstrip('/')}}/system_stats",
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print("\\nERROR: No se pudo conectar con ComfyUI.")
        print(f"{{e}}")
        print("\\nAsegúrate de que ComfyUI esté ejecutándose.")
        sys.exit(1)

    # --------------------------------------------------------
    # Cargar prompts
    # --------------------------------------------------------

    try:
        prompts = load_prompts(args.json_file)
    except Exception as e:
        print(f"\\nERROR leyendo JSON: {{e}}")
        sys.exit(1)

    print(f"Prompts encontrados: {{len(prompts)}}")

    # Filtro opcional.
    if args.start is not None:
        prompts = [
            item for item in prompts
            if item[0] >= args.start
        ]

    if args.end is not None:
        prompts = [
            item for item in prompts
            if item[0] <= args.end
        ]

    if not prompts:
        print("No hay prompts dentro del rango indicado.")
        sys.exit(0)

    print(f"Prompts a procesar: {{len(prompts)}}")
    print("Salidas por prompt: A2, A3, A4, A5, A6")
    print(f"Total de imágenes solicitadas: {{len(prompts) * 5}}")

    # --------------------------------------------------------
    # Enviar workflows
    # --------------------------------------------------------

    submitted = 0

    for position, (poster_id, prompt_text) in enumerate(prompts, start=1):
        print("\\n" + "-" * 70)
        print(
            f"POSTER {{poster_id}} "
            f"({{position}}/{{len(prompts)}})"
        )
        print("-" * 70)

        print(f"Prompt: {{prompt_text[:180]}}")
        if len(prompt_text) > 180:
            print("        ...")

        workflow = build_workflow(
            prompt_text=prompt_text,
            poster_id=poster_id,
        )

        try:
            prompt_id = queue_prompt(
                args.url,
                workflow,
            )
        except Exception as e:
            print(f"ERROR enviando poster {{poster_id}}: {{e}}")
            print(
                f"Se detiene aquí para evitar perder la secuencia. "
                f"{{submitted}} posters fueron enviados."
            )
            sys.exit(1)

        submitted += 1

        print(f"✓ Enviado a ComfyUI")
        print(f"  prompt_id: {{prompt_id}}")
        print(
            f"  salidas: poster{{poster_id}}_A2 ... "
            f"poster{{poster_id}}_A6"
        )

        if args.delay > 0 and position < len(prompts):
            time.sleep(args.delay)

    print("\\n" + "=" * 70)
    print("ENVÍO COMPLETADO")
    print("=" * 70)
    print(f"Posters enviados: {{submitted}}")
    print(f"Imágenes solicitadas: {{submitted * 5}}")
    print("\\nComfyUI continuará procesando la cola normalmente.")


if __name__ == "__main__":
    main()
'''

script_path.write_text(script, encoding="utf-8")
script_path.chmod(0o755)

# Validar que el script generado es sintácticamente correcto.
import py_compile
py_compile.compile(str(script_path), doraise=True)

print(f"Script creado: {script_path}")
print(f"Tamaño: {script_path.stat().st_size:,} bytes")
print("Sintaxis Python: OK")
print(f"Nodos del workflow hardcodeados: {len(workflow)}")
print("Salidas configuradas: A2, A3, A4, A5, A6")

