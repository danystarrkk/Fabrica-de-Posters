import os
import sys

from ollama import Client

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
input_dir = sys.argv[1]
trigger_word = sys.argv[2]
modelo_vision = "gemma4:31b"
cliente_ollama = Client(host="http://localhost:11434")

# ==========================================
# PROMPT DE INSTRUCCIÓN
# ==========================================

prompt_instruccion = f"""
Describe this image in extreme detail as if explaining a digital painting.
ALWAYS start your final response with exactly these words: "{trigger_word} style, ".
Focus on describing the lighting, cables, screens, artistic texture, and environmental chaos.
Write everything in a single continuous paragraph. Do not include introductions, notes, or line breaks. The description MUST be entirely in English.
"""

# ==========================================
# LÓGICA DE PROCESAMIENTO
# ==========================================
valid_extensions = (".png", ".jpg", ".jpeg", ".webp")

try:
    image_files = [
        f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)
    ]
    image_files.sort()
except FileNotFoundError:
    print(f"[ERROR] La carpeta '{input_dir}' no existe. Verifica la ruta.")
    exit(1)

print(
    f"Iniciando etiquetado de {len(image_files)} imágenes con el modelo '{modelo_vision}'..."
)
print("Conectando a Ollama a través de localhost:11434...\n")

for filename in image_files:
    image_path = os.path.join(input_dir, filename)
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(input_dir, txt_filename)

    if os.path.exists(txt_path):
        print(f"[SKIP] {txt_filename} ya existe.")
        continue

    try:
        response = cliente_ollama.chat(
            model=modelo_vision,
            messages=[
                {"role": "user", "content": prompt_instruccion, "images": [image_path]}
            ],
            options={
                "temperature": 0.2,
                # Subimos el presupuesto de tokens para que quepa el bloque de pensamiento + la respuesta
                "num_predict": 2000,
            },
        )

        msg = response.get("message")
        raw_content = getattr(msg, "content", "") or ""

        # En caso de que el modelo haya volcado el resultado antes de cerrar thinking
        if not raw_content and getattr(msg, "thinking", None):
            # Si content está vacío pero terminó por longitud, el modelo no alcanzó a responder
            print(f"\n[ALERTA] {filename} agotó tokens en fase de razonamiento.")
            print(f"Razonamiento parcial: {msg.thinking[:120]}...")
            continue

        descripcion = raw_content.strip().replace("\n", " ")

        # Validaciones de integridad
        if not descripcion:
            print(f"\n[ALERTA - RESPUESTA VACÍA] para {filename}")
            continue

        # Si el modelo no inició con el trigger word, se lo forzamos al inicio para no perder el caption
        if not descripcion.startswith(f"{trigger_word} style"):
            descripcion = f"{trigger_word} style, {descripcion}"

        # Guardar en el archivo .txt
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(descripcion)

        print(f"[OK] {filename} -> {txt_filename}")

    except Exception as e:
        print(f"\n[ERROR CRÍTICO] Falló al procesar {filename}: {e}\n")

print("\n¡Etiquetado finalizado!")
