#!/usr/bin/env python3
"""Robust production runner for ComfyUI."""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests

DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_STATE = "production_state.json"
DEFAULT_LOG = "production_log.txt"
DEFAULT_OUTPUT_DIR = "output"
POLL_INTERVAL = 5.0
CONNECTION_RETRY_DELAY = 10.0
REQUEST_TIMEOUT = 60
SAVE_NODES = ("17", "20", "24", "26", "29")
SAVE_LABELS = {"17": "A2", "20": "A3", "24": "A4", "26": "A5", "29": "A6"}
WORKFLOW = json.loads(r"""{
    "1": {
        "inputs": {
            "unet_name": "flux2-dev.safetensors",
            "weight_dtype": "fp8_e4m3fn"
        },
        "class_type": "UNETLoader",
        "_meta": {
            "title": "Load Diffusion Model"
        }
    },
    "4": {
        "inputs": {
            "vae_name": "ae.safetensors"
        },
        "class_type": "VAELoader",
        "_meta": {
            "title": "Load VAE"
        }
    },
    "5": {
        "inputs": {
            "lora_name": "t3chgrng_lora_000002000.safetensors",
            "strength_model": 0.85,
            "model": [
                "1",
                0
            ]
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {
            "title": "Load LoRA"
        }
    },
    "6": {
        "inputs": {
            "width": 1024,
            "height": 1448,
            "batch_size": 1
        },
        "class_type": "EmptyLatentImage",
        "_meta": {
            "title": "Empty Latent Image"
        }
    },
    "7": {
        "inputs": {
            "seed": 751804602429006,
            "steps": 30,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "sgm_uniform",
            "denoise": 1,
            "model": [
                "5",
                0
            ],
            "positive": [
                "9",
                0
            ],
            "negative": [
                "12",
                0
            ],
            "latent_image": [
                "6",
                0
            ]
        },
        "class_type": "KSampler",
        "_meta": {
            "title": "KSampler"
        }
    },
    "8": {
        "inputs": {
            "text": "t3chgrng style, a meticulously detailed digital painting depicting a cyberpunk figure immersed in a technological haze, characterized by a soft, diffused ambient light that casts gentle shadows while creating a luminous, warm glow from the small rectangular screen obscuring the character's eyes and the flickering blue interface of a vintage CRT television resting on the tiled floor. A chaotic, sprawling network of vibrant red and blue cables dominates the composition, snaking frantically from the character's headgear, looping through the sleeves of a navy blue bomber jacket, wrapping around intricate mechanical limbs, and plugging into both a handheld gaming device and the floor-bound monitor in a tangled web of connectivity. The artistic texture is defined by a clean, cel-shaded aesthetic with bold outlines and a subtle, grainy overlay reminiscent of high-fidelity comic art, utilizing a color palette of deep navy, muted teal, and striking crimson. Environmental chaos is concentrated in the dense layering of the character's cybernetics, where the prosthetic legs are a complex assembly of exposed gears, pistons, and wiring, seamlessly integrating into red and white high-top sneakers. The figure, sporting shock-teal hair and a colorful graphic t-shirt, sits perched on a simple wooden stool, creating a stark contrast between the organic simplicity of the furniture and the overwhelming technological clutter of the hardware and circuitry that binds them to their digital world.",
            "clip": [
                "31",
                0
            ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
            "title": "CLIP Text Encode (Prompt)"
        }
    },
    "9": {
        "inputs": {
            "guidance": 1.4,
            "conditioning": [
                "8",
                0
            ]
        },
        "class_type": "FluxGuidance",
        "_meta": {
            "title": "FluxGuidance"
        }
    },
    "10": {
        "inputs": {
            "samples": [
                "7",
                0
            ],
            "vae": [
                "4",
                0
            ]
        },
        "class_type": "VAEDecode",
        "_meta": {
            "title": "VAE Decode"
        }
    },
    "12": {
        "inputs": {
            "text": "",
            "clip": [
                "31",
                0
            ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
            "title": "CLIP Text Encode (Prompt)"
        }
    },
    "13": {
        "inputs": {
            "lora_name": "t3chgrng_lora_000002250.safetensors",
            "strength_model": 0.15,
            "model": [
                "1",
                0
            ]
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {
            "title": "Load LoRA"
        }
    },
    "14": {
        "inputs": {
            "upscale_by": 1.65,
            "seed": 213792688416422,
            "steps": 8,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 0.05,
            "mode_type": "None",
            "tile_width": 1536,
            "tile_height": 1536,
            "mask_blur": 24,
            "tile_padding": 96,
            "seam_fix_mode": "None",
            "seam_fix_denoise": 0.05,
            "seam_fix_width": 64,
            "seam_fix_mask_blur": 8,
            "seam_fix_padding": 16,
            "force_uniform_tiles": true,
            "tiled_decode": true,
            "batch_size": 8,
            "image": [
                "16",
                0
            ],
            "model": [
                "13",
                0
            ],
            "positive": [
                "33",
                0
            ],
            "negative": [
                "12",
                0
            ],
            "vae": [
                "4",
                0
            ],
            "upscale_model": [
                "15",
                0
            ]
        },
        "class_type": "UltimateSDUpscale",
        "_meta": {
            "title": "Ultimate SD Upscale"
        }
    },
    "15": {
        "inputs": {
            "model_name": "4xNomos8kDAT.pth"
        },
        "class_type": "UpscaleModelLoader",
        "_meta": {
            "title": "Load Upscale Model"
        }
    },
    "16": {
        "inputs": {
            "upscale_model": [
                "15",
                0
            ],
            "image": [
                "10",
                0
            ]
        },
        "class_type": "ImageUpscaleWithModel",
        "_meta": {
            "title": "Upscale Image (using Model)"
        }
    },
    "17": {
        "inputs": {
            "filename_prefix": "poster5_A2",
            "images": [
                "14",
                0
            ]
        },
        "class_type": "SaveImage",
        "_meta": {
            "title": "Save Image"
        }
    },
    "18": {
        "inputs": {
            "upscale_method": "lanczos",
            "width": 4093,
            "height": 5787,
            "crop": "disabled",
            "image": [
                "17",
                0
            ]
        },
        "class_type": "ImageScale",
        "_meta": {
            "title": "Upscale Image"
        }
    },
    "19": {
        "inputs": {
            "upscale_by": 1,
            "seed": 945951685987144,
            "steps": 6,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 0.04,
            "mode_type": "None",
            "tile_width": 1536,
            "tile_height": 1536,
            "mask_blur": 24,
            "tile_padding": 96,
            "seam_fix_mode": "None",
            "seam_fix_denoise": 0.05,
            "seam_fix_width": 64,
            "seam_fix_mask_blur": 8,
            "seam_fix_padding": 16,
            "force_uniform_tiles": true,
            "tiled_decode": true,
            "batch_size": 12,
            "image": [
                "18",
                0
            ],
            "model": [
                "13",
                0
            ],
            "positive": [
                "33",
                0
            ],
            "negative": [
                "12",
                0
            ],
            "vae": [
                "4",
                0
            ],
            "upscale_model": [
                "15",
                0
            ]
        },
        "class_type": "UltimateSDUpscale",
        "_meta": {
            "title": "Ultimate SD Upscale"
        }
    },
    "20": {
        "inputs": {
            "filename_prefix": "poster5_A3",
            "images": [
                "19",
                0
            ]
        },
        "class_type": "SaveImage",
        "_meta": {
            "title": "Save Image"
        }
    },
    "21": {
        "inputs": {
            "upscale_method": "lanczos",
            "width": 2894,
            "height": 4093,
            "crop": "disabled",
            "image": [
                "17",
                0
            ]
        },
        "class_type": "ImageScale",
        "_meta": {
            "title": "Upscale Image"
        }
    },
    "22": {
        "inputs": {
            "upscale_by": 1,
            "seed": 344393827267091,
            "steps": 6,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 0.03,
            "mode_type": "None",
            "tile_width": 1536,
            "tile_height": 1536,
            "mask_blur": 24,
            "tile_padding": 96,
            "seam_fix_mode": "None",
            "seam_fix_denoise": 0.05,
            "seam_fix_width": 64,
            "seam_fix_mask_blur": 8,
            "seam_fix_padding": 16,
            "force_uniform_tiles": true,
            "tiled_decode": true,
            "batch_size": 12,
            "image": [
                "21",
                0
            ],
            "model": [
                "13",
                0
            ],
            "positive": [
                "33",
                0
            ],
            "negative": [
                "12",
                0
            ],
            "vae": [
                "4",
                0
            ],
            "upscale_model": [
                "15",
                0
            ]
        },
        "class_type": "UltimateSDUpscale",
        "_meta": {
            "title": "Ultimate SD Upscale"
        }
    },
    "23": {
        "inputs": {
            "upscale_method": "area",
            "width": 2041,
            "height": 2894,
            "crop": "disabled",
            "image": [
                "17",
                0
            ]
        },
        "class_type": "ImageScale",
        "_meta": {
            "title": "Upscale Image"
        }
    },
    "24": {
        "inputs": {
            "filename_prefix": "poster5_A4",
            "images": [
                "22",
                0
            ]
        },
        "class_type": "SaveImage",
        "_meta": {
            "title": "Save Image"
        }
    },
    "25": {
        "inputs": {
            "upscale_by": 1,
            "seed": 792202144675391,
            "steps": 5,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 0.03,
            "mode_type": "None",
            "tile_width": 1024,
            "tile_height": 1024,
            "mask_blur": 24,
            "tile_padding": 64,
            "seam_fix_mode": "None",
            "seam_fix_denoise": 0.05,
            "seam_fix_width": 64,
            "seam_fix_mask_blur": 8,
            "seam_fix_padding": 16,
            "force_uniform_tiles": true,
            "tiled_decode": true,
            "batch_size": 12,
            "image": [
                "23",
                0
            ],
            "model": [
                "13",
                0
            ],
            "positive": [
                "33",
                0
            ],
            "negative": [
                "12",
                0
            ],
            "vae": [
                "4",
                0
            ],
            "upscale_model": [
                "15",
                0
            ]
        },
        "class_type": "UltimateSDUpscale",
        "_meta": {
            "title": "Ultimate SD Upscale"
        }
    },
    "26": {
        "inputs": {
            "filename_prefix": "poster5_A5",
            "images": [
                "25",
                0
            ]
        },
        "class_type": "SaveImage",
        "_meta": {
            "title": "Save Image"
        }
    },
    "27": {
        "inputs": {
            "upscale_method": "area",
            "width": 1446,
            "height": 2041,
            "crop": "disabled",
            "image": [
                "17",
                0
            ]
        },
        "class_type": "ImageScale",
        "_meta": {
            "title": "Upscale Image"
        }
    },
    "28": {
        "inputs": {
            "upscale_by": 1,
            "seed": 414449628071470,
            "steps": 5,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 0.03,
            "mode_type": "None",
            "tile_width": 1024,
            "tile_height": 1024,
            "mask_blur": 24,
            "tile_padding": 64,
            "seam_fix_mode": "None",
            "seam_fix_denoise": 0.05,
            "seam_fix_width": 64,
            "seam_fix_mask_blur": 8,
            "seam_fix_padding": 16,
            "force_uniform_tiles": true,
            "tiled_decode": true,
            "batch_size": 12,
            "image": [
                "27",
                0
            ],
            "model": [
                "13",
                0
            ],
            "positive": [
                "33",
                0
            ],
            "negative": [
                "12",
                0
            ],
            "vae": [
                "4",
                0
            ],
            "upscale_model": [
                "15",
                0
            ]
        },
        "class_type": "UltimateSDUpscale",
        "_meta": {
            "title": "Ultimate SD Upscale"
        }
    },
    "29": {
        "inputs": {
            "filename_prefix": "poster5_A6",
            "images": [
                "28",
                0
            ]
        },
        "class_type": "SaveImage",
        "_meta": {
            "title": "Save Image"
        }
    },
    "31": {
        "inputs": {
            "clip_name": "mistral_3_small_flux2_fp8mixed.safetensors",
            "type": "flux2",
            "device": "default"
        },
        "class_type": "CLIPLoader",
        "_meta": {
            "title": "Load CLIP"
        }
    },
    "32": {
        "inputs": {
            "text": "Preserve and faithfully refine the existing image.\n\nTreat the input image as the primary source of truth. Preserve its composition, perspective, framing, proportions, geometry, object identity, spatial relationships, lighting direction, depth, atmosphere, color character and visual intent.\n\nUpscale and reconstruct missing or degraded details by following the existing shapes, structures, textures, patterns and material behavior already present in the image. Restore continuity in fine details and complete partially resolved elements in a physically coherent and visually consistent way.\n\nImprove local detail, surface definition, material separation, edge continuity and fine texture while maintaining the original appearance. Preserve natural variation, realistic imperfections, subtle texture and photographic or cinematic character.\n\nDo not redesign the image. Do not reinterpret the subject. Do not introduce new objects, structures, patterns or visual concepts that are not already supported by the image. Do not duplicate, merge, relocate, simplify or unnecessarily alter existing elements.\n\nWhen details are ambiguous or partially missing, infer them conservatively from their surrounding context and continue the existing visual structure rather than inventing new detail.\n\nMaintain natural tonal transitions, realistic highlights and shadows, restrained color, authentic material response and atmospheric depth. Preserve the emotional impact and visual character of the original image.\n\nThe goal is faithful reconstruction, restoration and controlled enhancement: more complete, coherent and refined detail while remaining unmistakably the same image.",
            "clip": [
                "31",
                0
            ]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {
            "title": "CLIP Text Encode (Prompt)"
        }
    },
    "33": {
        "inputs": {
            "guidance": 1.4,
            "conditioning": [
                "32",
                0
            ]
        },
        "class_type": "FluxGuidance",
        "_meta": {
            "title": "FluxGuidance"
        }
    }
}""")


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_log(log_path, message):
    line = f"[{now_iso()}] {message}\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line.rstrip())


def atomic_write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_prompts(path):
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = (
        data.get("prompts")
        if isinstance(data, dict)
        else data if isinstance(data, list) else None
    )
    if not isinstance(prompts, list):
        raise ValueError("El JSON debe contener una lista 'prompts'.")
    result = []
    for index, item in enumerate(prompts, 1):
        if isinstance(item, dict):
            poster_id = item.get("id", index)
            text = item.get("prompt")
        elif isinstance(item, str):
            poster_id = index
            text = item
        else:
            raise ValueError(f"Elemento #{index} del JSON no es válido.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"El prompt #{index} está vacío o no es texto.")
        result.append((int(poster_id), text.strip()))
    return result


def build_workflow(prompt_text, poster_id):
    wf = json.loads(json.dumps(WORKFLOW))
    wf["8"]["inputs"]["text"] = prompt_text
    for node, label in SAVE_LABELS.items():
        wf[node]["inputs"]["filename_prefix"] = f"poster{poster_id}_{label}"
    return wf


def check_comfy(url):
    r = requests.get(f"{url.rstrip('/')}/system_stats", timeout=10)
    r.raise_for_status()


def queue_prompt(url, wf):
    payload = {"prompt": wf, "client_id": str(uuid.uuid4())}
    r = requests.post(
        f"{url.rstrip('/')}/prompt", json=payload, timeout=REQUEST_TIMEOUT
    )
    r.raise_for_status()
    data = r.json()
    if "prompt_id" not in data:
        raise RuntimeError(f"ComfyUI no devolvió prompt_id: {data}")
    return data["prompt_id"]


def get_history(url, prompt_id):
    r = requests.get(f"{url.rstrip('/')}/history/{prompt_id}", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get(prompt_id)


def find_saved_outputs(output_dir, poster_id):
    if not output_dir.exists() or not output_dir.is_dir():
        return {}
    found = {}
    valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    for label in SAVE_LABELS.values():
        prefix = f"poster{poster_id}_{label}"
        matches = [
            str(x)
            for x in output_dir.rglob(f"{prefix}*")
            if x.is_file() and x.suffix.lower() in valid_ext
        ]
        found[label] = sorted(matches)
    return found


def saved_outputs_complete(output_dir, poster_id):
    found = find_saved_outputs(output_dir, poster_id)
    complete = all(found.get(label) for label in SAVE_LABELS.values())
    return complete, found


def inspect_history(history):
    if not history:
        return "unknown", {}, None
    status = history.get("status") or {}
    status_str = status.get("status_str")
    completed = status.get("completed")
    outputs = history.get("outputs") or {}
    found = {}
    for node in SAVE_NODES:
        images = (outputs.get(node) or {}).get("images") or []
        found[SAVE_LABELS[node]] = [
            x.get("filename")
            for x in images
            if isinstance(x, dict) and x.get("filename")
        ]
    all_ok = all(found.get(label) for label in SAVE_LABELS.values())
    if status_str == "error":
        return "error", found, status.get("messages") or "ComfyUI reportó error."
    if completed is True and all_ok:
        return "success", found, None
    if completed is True and not all_ok:
        return "error", found, "ComfyUI terminó pero faltan una o más salidas A2-A6."
    return "running", found, None


def wait_for_connection(url, log_path):
    while True:
        try:
            check_comfy(url)
            return
        except requests.RequestException as e:
            append_log(
                log_path,
                f"ComfyUI no disponible: {e}. Reintentando en {CONNECTION_RETRY_DELAY}s...",
            )
            time.sleep(CONNECTION_RETRY_DELAY)


def wait_for_result(url, prompt_id, poll, log_path):
    while True:
        try:
            h = get_history(url, prompt_id)
            status, outputs, error = inspect_history(h)
            if status in ("success", "error"):
                return status, outputs, error
            time.sleep(poll)
        except requests.RequestException as e:
            append_log(
                log_path,
                f"Sin conexión mientras se esperaba prompt_id={prompt_id}: {e}. Reintentando...",
            )
            time.sleep(CONNECTION_RETRY_DELAY)


def new_item(pid):
    return {
        "id": pid,
        "status": "pending",
        "prompt_id": None,
        "attempts": 0,
        "started_at": None,
        "completed_at": None,
        "outputs": {},
        "error": None,
    }


def new_state(json_file, prompts):
    return {
        "version": 1,
        "source_json": str(json_file.resolve()),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "items": {str(pid): new_item(pid) for pid, _ in prompts},
    }


def load_state(path, json_file, prompts, reset=False):
    if reset or not path.exists():
        state = new_state(json_file, prompts)
        atomic_write_json(path, state)
        return state
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state.get("items"), dict):
        raise ValueError("Estado inválido: falta 'items'.")
    for pid, _ in prompts:
        state["items"].setdefault(str(pid), new_item(pid))
    state["source_json"] = str(json_file.resolve())
    state["updated_at"] = now_iso()
    atomic_write_json(path, state)
    return state


def save_state(path, state):
    state["updated_at"] = now_iso()
    atomic_write_json(path, state)


def main():
    p = argparse.ArgumentParser(
        description="Producción robusta de posters para ComfyUI."
    )
    p.add_argument("json_file", type=Path)
    p.add_argument("--url", default=DEFAULT_COMFY_URL)
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--poll", type=float, default=POLL_INTERVAL)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--state", type=Path, default=Path(DEFAULT_STATE))
    p.add_argument("--log", type=Path, default=Path(DEFAULT_LOG))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directorio de salida de ComfyUI. Default: ./output",
    )
    p.add_argument("--reset-state", action="store_true")
    a = p.parse_args()
    prompts = load_prompts(a.json_file)
    if a.start is not None:
        prompts = [x for x in prompts if x[0] >= a.start]
    if a.end is not None:
        prompts = [x for x in prompts if x[0] <= a.end]
    if not prompts:
        print("No hay prompts dentro del rango indicado.")
        return 0
    state = load_state(a.state, a.json_file, prompts, a.reset_state)
    output_dir = a.output_dir.resolve()
    print(f"Output:  {output_dir}")
    append_log(
        a.log,
        f"INICIO/REANUDACIÓN: {len(prompts)} prompts en este rango. Producción secuencial: 1 poster a la vez.",
    )
    completed = sum(
        1
        for pid, _ in prompts
        if state["items"].get(str(pid), {}).get("status") == "completed"
    )
    for pos, (pid, text) in enumerate(prompts, 1):
        item = state["items"][str(pid)]
        print("\n" + "-" * 70 + f"\nPOSTER {pid} ({pos}/{len(prompts)})\n" + "-" * 70)
        if item.get("status") == "completed":
            print(f"✓ Poster {pid} ya completado; se omite.")
            continue

        # Recuperación adicional por archivos: sobrevive incluso si ComfyUI
        # perdió su historial tras un reinicio/apagón.
        files_complete, files_found = saved_outputs_complete(output_dir, pid)
        if files_complete:
            item.update(
                status="completed",
                outputs=files_found,
                completed_at=now_iso(),
                error=None,
            )
            save_state(a.state, state)
            completed += 1
            append_log(
                a.log,
                f"RECUPERADO POR ARCHIVOS poster={pid}: A2/A3/A4/A5/A6 ya existen; no se duplica.",
            )
            continue

        existing = item.get("prompt_id")
        if existing:
            wait_for_connection(a.url, a.log)
            try:
                status, outputs, error = inspect_history(get_history(a.url, existing))
                if status == "success":
                    file_complete, file_outputs = saved_outputs_complete(
                        output_dir, pid
                    )
                    final_outputs = file_outputs if file_complete else outputs
                    item.update(
                        status="completed",
                        outputs=final_outputs,
                        completed_at=now_iso(),
                        error=None,
                    )
                    save_state(a.state, state)
                    completed += 1
                    append_log(
                        a.log,
                        f"RECUPERADO poster={pid} prompt_id={existing}: ya estaba completo; no se duplica.",
                    )
                    continue
                if status == "running":
                    append_log(
                        a.log,
                        f"RECUPERANDO poster={pid} prompt_id={existing}: sigue en ejecución.",
                    )
                    status, outputs, error = wait_for_result(
                        a.url, existing, a.poll, a.log
                    )
                    if status == "success":
                        file_complete, file_outputs = saved_outputs_complete(
                            output_dir, pid
                        )
                        final_outputs = file_outputs if file_complete else outputs
                        item.update(
                            status="completed",
                            outputs=final_outputs,
                            completed_at=now_iso(),
                            error=None,
                        )
                        save_state(a.state, state)
                        completed += 1
                        append_log(
                            a.log,
                            f"COMPLETADO poster={pid} prompt_id={existing} tras recuperación.",
                        )
                        continue
                item.update(
                    status="pending",
                    prompt_id=None,
                    error=str(error) if error else "trabajo perdido o fallido",
                )
                save_state(a.state, state)
                append_log(
                    a.log,
                    f"RECUPERACIÓN poster={pid}: el trabajo anterior no quedó completado; se reintentará.",
                )
            except requests.RequestException as e:
                append_log(
                    a.log,
                    f"No se pudo consultar la recuperación de poster={pid}: {e}. Se volverá a comprobar y/o reintentar.",
                )
        success = False
        for attempt in range(1, a.retries + 1):
            wait_for_connection(a.url, a.log)
            item.update(
                status="running",
                attempts=int(item.get("attempts", 0)) + 1,
                started_at=now_iso(),
                error=None,
            )
            save_state(a.state, state)
            append_log(a.log, f"ENVIANDO poster={pid} intento={attempt}/{a.retries}.")
            try:
                prompt_id = queue_prompt(a.url, build_workflow(text, pid))
                item["prompt_id"] = prompt_id
                save_state(a.state, state)
                append_log(
                    a.log,
                    f"ACEPTADO poster={pid} prompt_id={prompt_id}; esperando finalización y verificación A2-A6.",
                )
                status, outputs, error = wait_for_result(
                    a.url, prompt_id, a.poll, a.log
                )
                if status == "success":
                    item.update(
                        status="completed",
                        outputs=outputs,
                        completed_at=now_iso(),
                        error=None,
                    )
                    save_state(a.state, state)
                    completed += 1
                    success = True
                    append_log(a.log, f"OK poster={pid}: A2/A3/A4/A5/A6 verificadas.")
                    break
                item.update(status="pending", error=str(error))
                save_state(a.state, state)
                append_log(a.log, f"ERROR poster={pid}: {error}")
            except (requests.RequestException, RuntimeError) as e:
                item.update(status="pending", error=str(e))
                save_state(a.state, state)
                append_log(a.log, f"FALLO poster={pid} intento={attempt}: {e}")
            if not success and attempt < a.retries:
                time.sleep(CONNECTION_RETRY_DELAY)
        if not success:
            append_log(
                a.log,
                f"DETENIDO en poster={pid}: se agotaron los reintentos. Estado guardado para reanudar.",
            )
            print(
                f"\nProducción detenida en poster {pid}. No se pierde el progreso anterior."
            )
            return 2
        if a.delay > 0:
            time.sleep(a.delay)
    append_log(
        a.log,
        f"PRODUCCIÓN FINALIZADA: {completed}/{len(prompts)} posters completados; {completed*5} imágenes verificadas.",
    )
    print("\n" + "=" * 70 + "\nPRODUCCIÓN COMPLETADA\n" + "=" * 70)
    print(f"Posters completados: {completed}/{len(prompts)}")
    print(f"Estado: {a.state}")
    print(f"Log:    {a.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
