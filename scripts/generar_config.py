import os
import sys

if len(sys.argv) != 4:
    print("Uso incorrecto.")
    print(
        "Comando: python generar_config.py <trigger_word> <ruta_dataset> <ruta_salida>"
    )
    sys.exit(1)

trigger_word = sys.argv[1]
ruta_dataset = os.path.abspath(sys.argv[2])
ruta_salida = os.path.abspath(sys.argv[3])
nombre_proyecto = f"{trigger_word}_lora"

yaml_content = f"""---
job: "extension"
config:
  name: "{nombre_proyecto}"
  process:
    - type: "diffusion_trainer"
      training_folder: "{ruta_salida}"
      sqlite_db_path: "./aitk_db.db"
      device: "cuda:0"
      trigger_word: "{trigger_word}"
      performance_log_every: 10
      network:
        type: "lora"
        linear: 128
        linear_alpha: 64
        conv: 16
        conv_alpha: 16
        lokr_full_rank: true
        lokr_factor: -1
        network_kwargs:
          ignore_if_contains: []
      save:
        dtype: "float16"
        save_every: 250
        max_step_saves_to_keep: 10
        save_format: "diffusers"
        push_to_hub: false
      datasets:
        - folder_path: "{ruta_dataset}"
          mask_path: null
          mask_min_value: 0.1
          default_caption: ""
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          cache_latents_to_disk: true
          is_reg: false
          network_weight: 1
          resolution:
            - 1024
            - 1024
          controls: []
          shrink_video_to_frames: true
          num_frames: 1
          flip_x: false
          flip_y: false
          num_repeats: 1
          control_path_1: null
          control_path_2: null
          control_path_3: null
      train:
        batch_size: 1
        bypass_guidance_embedding: false
        steps: 2500
        gradient_accumulation: 2
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "prodigy"
        timestep_type: "weighted"
        content_or_style: "balanced"
        optimizer_params:
          weight_decay: 0.0001
        unload_text_encoder: true      # <--- ¡NUEVO! Borra el TE después de leer textos
        cache_text_embeddings: true    # <--- ¡NUEVO! Guarda los textos para no volver a cargar Mistral
        lr: 1.0
        ema_config:
          use_ema: true
          ema_decay: 0.99
        skip_first_sample: false
        force_first_sample: false
        disable_sampling: false
        dtype: "bf16"
        diff_output_preservation: false
        diff_output_preservation_multiplier: 1
        diff_output_preservation_class: "person"
        switch_boundary_every: 1
        loss_type: "mse"
      logging:
        log_every: 1
        use_ui_logger: true
      model:
        name_or_path: "black-forest-labs/FLUX.2-dev"
        quantize: false
        qtype: "qfloat8"
        quantize_te: false
        qtype_te: "qfloat8"
        arch: "flux2"
        low_vram: false
        model_kwargs:
          match_target_res: false
        compile: false
        layer_offloading: true                         # <--- ¡NUEVO! Activa offloading
        layer_offloading_text_encoder_percent: 1       # <--- ¡NUEVO! Envía Mistral (el Text Encoder) 100% a la CPU
        layer_offloading_transformer_percent: 0        # Flux2 se queda 100% en la GPU
      sample:
        sampler: "flowmatch"
        sample_every: 250
        sample_start_step: 0
        width: 1024
        height: 1024
        samples: []
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 4
        sample_steps: 30
        num_frames: 1
        fps: 1
meta:
  name: "{nombre_proyecto}"
  version: "1.0"
"""

with open("config.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"\\n[ÉXITO] Archivo 'config.yaml' con Caché y CPU Offloading generado.")
