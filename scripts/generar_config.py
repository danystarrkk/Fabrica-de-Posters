import os
import sys

# Verificación de argumentos actualizados (ahora son 3)
if len(sys.argv) != 4:
    print("Uso incorrecto.")
    print(
        "Comando: python generar_config.py <trigger_word> <ruta_dataset> <ruta_salida>"
    )
    print(
        "Ejemplo: python generar_config.py t3chgrng ../dataset/t3chgrng ../modelos_terminados"
    )
    sys.exit(1)

trigger_word = sys.argv[1]
ruta_dataset = os.path.abspath(sys.argv[2])
ruta_salida = os.path.abspath(sys.argv[3])  # Nuevo parámetro
nombre_proyecto = f"{trigger_word}_lora"

yaml_content = f"""---
job: extension
config:
  name: "{nombre_proyecto}"
  process:
    - type: 'sd_trainer'
      training_folder: "{ruta_salida}"  # <--- Ahora apunta a tu carpeta personalizada
      device: cuda:0
      
      # 1. Arquitectura del LoRA
      network:
        type: "lora"
        linear: 128
        linear_alpha: 64
        
      # 2. Guardado por etapas
      save:
        dtype: float16 
        save_every: 250
        max_step_saves_to_keep: 10
        
      # 3. Tu Dataset
      datasets:
        - folder_path: "{ruta_dataset}"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [1024, 1024]
          
      # 4. Parámetros de Entrenamiento
      train:
        batch_size: 1
        steps: 2500
        gradient_accumulation_steps: 2
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "prodigy"
        lr: 1.0
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16
        
      # 5. El Modelo Base (FLUX.2 Dev)
      model:
        name_or_path: "black-forest-labs/FLUX.2-dev"
        is_flux: true
        quantize: true
meta:
  name: "{nombre_proyecto}"
  version: '1.0'
"""

with open("config.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"\n[ÉXITO] Archivo 'config.yaml' generado.")
print(f"Dataset de entrada: {ruta_dataset}")
print(f"Los LoRAs terminados se guardarán en: {ruta_salida}/{nombre_proyecto}")
print("Puedes iniciar el entrenamiento con: python run.py config.yaml\n")
