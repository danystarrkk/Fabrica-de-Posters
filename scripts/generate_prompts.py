import argparse
import json
import os

import requests

API_KEY = os.getenv("NVIDIA_API_KEY")
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

NICHES_CONFIG = {
    "t3chgrng": {
        "context": "Physical subjects: industrial server cores, underground vaults, tangled fiber optic bundles, heavy machinery, thick iron beams, mechanical geometry, low-angle architectural shots, harsh directional lighting."
    },
    "anmcoutr": {
        "context": "Physical subjects: asymmetrical garments, layered technical fabrics, oversized jackets, dynamic fashion poses, urban concrete backgrounds, sharp tailoring, localized neon light sources."
    },
    "drkcdmia": {
        "context": "Physical subjects: towering mahogany bookshelves, stacked leather-bound volumes, gothic stone archways, heavy wooden desks, scattered parchment, localized light from dusty windows or candle clusters, deep spatial depth."
    },
    "jpndwabi": {
        "context": "Physical subjects: minimalist stone gardens, weathered wooden structures, irregular ceramic bowls, bamboo dividers, soft directional shadows, asymmetrical room layouts, natural wear."
    },
}


def generate_prompts_from_ai(niche_key, niche_data, count):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    prompt_instruction = (
        f"Generate exactly {count} distinct image prompt(s) and short title(s) for the category '{niche_key}'. "
        f"Subject structural guidelines: {niche_data['context']}. "
        "Output a valid JSON array of objects, where each object has 'title' and 'prompt' keys. Do not include markdown formatting inside the json text."
    )

    system_instruction = (
        "You are a technical prompt engineer for advanced generative image models. "
        "CRITICAL RULES FOR PROMPTS: "
        "1. DO NOT use style tags, aesthetic buzzwords, or mood adjectives (e.g., avoid 'cyberpunk', 'anime', 'gritty', 'wabi-sabi', 'masterpiece'). "
        "2. The aesthetic style is already applied externally by a LoRA neural network. "
        "3. Focus STRICTLY on describing concrete physical objects, structural geometry, camera angles, lighting placement, and composition. "
        "4. Write in natural, descriptive English. "
        "Output a strict JSON array containing objects with 'title' and 'prompt' keys. No extra commentary."
    )

    payload = {
        "model": "nvidia/nemotron-3-ultra-550b-a55b",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_instruction},
        ],
        "temperature": 0.7,
        "max_tokens": 1500,
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
        try:
            clean_content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content)
        except json.JSONDecodeError:
            print(
                f"Advertencia: No se pudo parsear el JSON directamente para {niche_key}."
            )
            return []
    else:
        raise Exception(f"Error en la API ({response.status_code}): {response.text}")


def main():
    parser = argparse.ArgumentParser(
        description="Generador automatizado de prompts estructurales por nicho."
    )
    parser.add_argument(
        "--niche",
        type=str,
        default="all",
        choices=list(NICHES_CONFIG.keys()) + ["all"],
        help="Nicho específico o 'all' para todos.",
    )
    parser.add_argument(
        "--count", type=int, default=1, help="Número de prompts a generar por nicho."
    )
    args = parser.parse_args()

    if not API_KEY:
        print(
            "Error crítico: La variable de entorno NVIDIA_API_KEY no está configurada en ZSH."
        )
        return

    niches_to_process = (
        list(NICHES_CONFIG.keys()) if args.niche == "all" else [args.niche]
    )

    # El bucle ahora guarda un archivo independiente por cada nicho procesado
    for niche in niches_to_process:
        print(f"Generando {args.count} prompt(s) para el nicho: {niche}...")
        niche_database = []

        try:
            ai_output = generate_prompts_from_ai(
                niche, NICHES_CONFIG[niche], args.count
            )
            if isinstance(ai_output, dict):
                ai_output = [ai_output]

            for item in ai_output:
                entry = {
                    "niche_id": niche,
                    "title": item.get("title", "Untitled"),
                    "prompt": item.get("prompt", ""),
                }
                niche_database.append(entry)

            # Guardado específico del nicho
            output_filename = f"{niche}_prompts.json"
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(niche_database, f, indent=4, ensure_ascii=False)
            print(f"¡Base de datos guardada exitosamente en {output_filename}!\n")

        except Exception as e:
            print(f"Fallo al procesar {niche}: {e}\n")


if __name__ == "__main__":
    main()
