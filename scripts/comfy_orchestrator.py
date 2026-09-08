import json
import os
import time
import shutil
import requests
import argparse

# Configuración de rutas y API local
COMFY_API_URL = "http://127.0.0.1:8188"
COMFY_OUTPUT_DIR = os.path.expanduser("~/ComfyUI/output")
HOME_DIR = os.path.expanduser("~")

# Mapeo de LoRAs por Nicho (Generación Base vs. Inyección de Textura al Escalar)
LORA_MAP = {
    "t3chgrng": {
        "base": "t3chgrng_base_v1.safetensors",
        "upscale": "t3chgrng_texture_injector_v1.safetensors",  # El del Step 2500
    },
    "anmcoutr": {
        "base": "anmcoutr_base_v1.safetensors",
        "upscale": "anmcoutr_texture_v1.safetensors",
    },
    "drkcdmia": {
        "base": "drkcdmia_base_v1.safetensors",
        "upscale": "drkcdmia_texture_v1.safetensors",
    },
    "jpndwabi": {
        "base": "jpndwabi_base_v1.safetensors",
        "upscale": "jpndwabi_texture_v1.safetensors",
    },
}

# MAPEO DE NODOS (Actualiza estos IDs abriendo tu workflow_api.json en Neovim)
NODE_IDS = {
    "clip_text": "6",  # ID del nodo CLIP Text Encode (Prompt Positivo)
    "load_lora_base": "10",  # ID del Load LoRA conectado al modelo base (Etapa 1)
    "load_lora_upscale": "35",  # ID del Load LoRA conectado a los Ultimate SD Upscale (Etapa 2)
    "save_a2": "20",  # ID del nodo Save Image (Rama A2)
    "save_a3": "21",  # ID del nodo Save Image (Rama A3)
    "save_a4": "22",  # ID del nodo Save Image (Rama A4)
    "save_a5": "23",  # ID del nodo Save Image (Rama A5)
    "save_a6": "24",  # ID del nodo Save Image (Rama A6)
}


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def trigger_comfy_generation(workflow_data):
    """Envía el flujo a ComfyUI y devuelve el ID de la tarea."""
    response = requests.post(f"{COMFY_API_URL}/prompt", json={"prompt": workflow_data})
    if response.status_code == 200:
        return response.json()["prompt_id"]
    raise Exception(f"Fallo al enviar a ComfyUI: {response.text}")


def wait_for_completion(prompt_id):
    """Hace polling al historial de ComfyUI hasta que termine la generación."""
    print(f"Esperando a que ComfyUI procese la tarea {prompt_id}...")
    while True:
        history_res = requests.get(f"{COMFY_API_URL}/history/{prompt_id}")
        if history_res.status_code == 200 and prompt_id in history_res.json():
            print("¡Generación completada en ComfyUI!")
            return history_res.json()[prompt_id]["outputs"]
        time.sleep(5)


def organize_outputs(niche_id, safe_title):
    """Mueve los archivos desde ComfyUI/output a la estructura ~/nicho_id/titulo/"""
    target_dir = os.path.join(HOME_DIR, niche_id, safe_title)
    os.makedirs(target_dir, exist_ok=True)

    files_moved = 0
    for filename in os.listdir(COMFY_OUTPUT_DIR):
        if filename.startswith(safe_title) and filename.endswith(".png"):
            source_path = os.path.join(COMFY_OUTPUT_DIR, filename)
            dest_path = os.path.join(target_dir, filename)
            shutil.move(source_path, dest_path)
            files_moved += 1

    print(f"Se movieron {files_moved} archivos a: {target_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Orquestador de ComfyUI para pipeline de impresión."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Ruta al archivo JSON de prompts (ej. t3chgrng_prompts.json)",
    )
    args = parser.parse_args()

    try:
        workflow = load_json("workflow_api.json")
        prompts_db = load_json(args.prompts)
    except Exception as e:
        print(f"Error cargando archivos JSON: {e}")
        return

    for item in prompts_db:
        niche = item["niche_id"]
        raw_title = item["title"]
        prompt_text = item["prompt"]

        safe_title = raw_title.replace(" ", "_").replace("/", "").replace("\\", "")
        print(f"\n--- Procesando: {safe_title} ({niche}) ---")

        # Configurar los LoRAs (Base y Upscale) de forma independiente
        loras = LORA_MAP.get(
            niche,
            {
                "base": "default_base.safetensors",
                "upscale": "default_upscale.safetensors",
            },
        )

        workflow[NODE_IDS["load_lora_base"]]["inputs"]["lora_name"] = loras["base"]
        workflow[NODE_IDS["load_lora_upscale"]]["inputs"]["lora_name"] = loras[
            "upscale"
        ]

        # Inyectar prompt maestro
        workflow[NODE_IDS["clip_text"]]["inputs"]["text"] = prompt_text

        # Configurar los prefijos de guardado para la cascada descendente
        workflow[NODE_IDS["save_a2"]]["inputs"]["filename_prefix"] = f"{safe_title}_A2"
        workflow[NODE_IDS["save_a3"]]["inputs"]["filename_prefix"] = f"{safe_title}_A3"
        workflow[NODE_IDS["save_a4"]]["inputs"]["filename_prefix"] = f"{safe_title}_A4"
        workflow[NODE_IDS["save_a5"]]["inputs"]["filename_prefix"] = f"{safe_title}_A5"
        workflow[NODE_IDS["save_a6"]]["inputs"]["filename_prefix"] = f"{safe_title}_A6"

        try:
            prompt_id = trigger_comfy_generation(workflow)
            wait_for_completion(prompt_id)
            organize_outputs(niche, safe_title)
        except Exception as e:
            print(f"Error en el ciclo de generación: {e}")


if __name__ == "__main__":
    main()
