import os
import sys

from PIL import Image

# Configuración de carpetas
input_dir = sys.argv[1]  # <-- CAMBIA ESTO
output_dir = sys.argv[2]
name = sys.argv[3]

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Extensiones válidas
valid_extensions = (".png", ".jpg", ".jpeg", ".webp")
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
image_files.sort()

target_short_edge = 1024

print(f"Procesando {len(image_files)} imágenes...")

for idx, filename in enumerate(image_files, start=1):
    input_path = os.path.join(input_dir, filename)

    try:
        with Image.open(input_path) as img:
            # Convertir a RGB (elimina transparencias que rompen el entrenamiento)
            img = img.convert("RGB")

            width, height = img.size

            # Calcular nueva resolución manteniendo la proporción
            if width < height:
                new_width = target_short_edge
                new_height = int(target_short_edge * (height / width))
            else:
                new_height = target_short_edge
                new_width = int(target_short_edge * (width / height))

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Recortar al múltiplo de 64 más cercano desde el centro
            crop_width = (new_width // 64) * 64
            crop_height = (new_height // 64) * 64

            left = (new_width - crop_width) / 2
            top = (new_height - crop_height) / 2
            right = (new_width + crop_width) / 2
            bottom = (new_height + crop_height) / 2

            img = img.crop((left, top, right, bottom))

            # Renombrar y guardar
            new_filename = f"{name}_{idx:03d}.jpg"
            output_path = os.path.join(output_dir, new_filename)

            # Guardar con máxima calidad
            img.save(output_path, "JPEG", quality=100)
            print(f"[OK] {filename} -> {new_filename} ({crop_width}x{crop_height})")

    except Exception as e:
        print(f"[ERROR] Falló al procesar {filename}: {e}")

print("¡Procesamiento completo!")
