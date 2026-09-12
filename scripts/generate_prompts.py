import re
from pathlib import Path

src = Path("/mnt/data/generate_prompts.py")
text = src.read_text(encoding="utf-8")

# 1) Cambiar la validación de ask_ollama para exigir exactamente 1 prompt.
text = re.sub(
    r"def ask_ollama\(model, prompt, expected_count\):",
    "def ask_ollama(model, prompt):",
    text,
)

text = re.sub(
    r"""    if len\(prompts\) != expected_count:\n        raise RuntimeError\(\n            f"Expected \{expected_count\} prompts, "\n            f"but Ollama returned \{len\(prompts\)\}."\n        \)\n""",
    """    if len(prompts) != 1:
        raise RuntimeError(
            f"Expected exactly 1 prompt, but Ollama returned {len(prompts)}."
        )
""",
    text,
)

# 2) La función de construcción ahora pide exactamente un prompt.
text = re.sub(
    r"def build_generation_prompt\(batch, amount\):",
    "def build_generation_prompt(batch):",
    text,
)

text = re.sub(
    r'''    prompt = f""" 
    \{SYSTEM_INSTRUCTION\}

    You must generate exactly \{amount\} original prompts\.
''',
    '''    prompt = f"""
    {SYSTEM_INSTRUCTION}

    You must generate exactly ONE original prompt.
''',
    text,
)

text = re.sub(
    r"""    Now generate exactly \{amount\} completely original prompts\.
""",
    """    Now generate exactly ONE completely original prompt.
""",
    text,
)

# 3) Reemplazar el bloque de generación por llamada individual.
old = """        print(f"Generating {amount} prompt(s)...")

        generation_prompt = build_generation_prompt(batch, amount)

        success = False

        # Reintentar si Gemma devuelve JSON incorrecto,
        # cantidad incorrecta, etc.
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):

            try:
                prompts = ask_ollama(args.model, generation_prompt, amount)

                success = True
                break

            except Exception as e:

                print(f"\\n  ERROR on attempt " f"{attempt}/{max_attempts}:")

                print(f"  {e}")

                if attempt < max_attempts:
                    print("\\n  Retrying in 3 seconds...")
                    time.sleep(3)

        if not success:
            print("\\nERROR: Could not generate this batch.")

            print(
                "Already generated prompts have been " "preserved in the output file."
            )

            sys.exit(1)

        # ----------------------------------------------------
        # Añadir resultados
        # ----------------------------------------------------

        all_prompts.extend(prompts)

        # ----------------------------------------------------
        # Guardar inmediatamente
        # ----------------------------------------------------

        save_json(args.output, all_prompts)

        print(f"\\n✓ Batch {batch_index} completed.")

        print(f"✓ Generated so far: " f"{len(all_prompts)}/{args.number}")

        print(f"✓ Saved: {args.output}")
"""

new = """        print(
            f"Generating {amount} prompt(s) individually "
            f"from the same {len(batch)} references..."
        )

        generation_prompt = build_generation_prompt(batch)

        # Cada llamada a Ollama genera EXACTAMENTE un prompt.
        # Las mismas 20 referencias se vuelven a enviar en cada llamada
        # perteneciente a este batch.
        for prompt_number in range(1, amount + 1):

            print(
                f"\\n  Prompt {prompt_number}/{amount} "
                f"(global {len(all_prompts) + 1}/{args.number})"
            )

            success = False
            max_attempts = 3

            for attempt in range(1, max_attempts + 1):

                try:
                    prompts = ask_ollama(args.model, generation_prompt)

                    # ask_ollama ya valida que exista exactamente un prompt.
                    all_prompts.extend(prompts)

                    success = True
                    break

                except Exception as e:

                    print(
                        f"\\n  ERROR on attempt "
                        f"{attempt}/{max_attempts}:"
                    )
                    print(f"  {e}")

                    if attempt < max_attempts:
                        print("\\n  Retrying in 3 seconds...")
                        time.sleep(3)

            if not success:
                print(
                    f"\\nERROR: Could not generate prompt "
                    f"{prompt_number} of batch {batch_index}."
                )

                print(
                    "Already generated prompts have been "
                    "preserved in the output file."
                )

                save_json(args.output, all_prompts)
                sys.exit(1)

            # Guardar después de CADA prompt, no solamente al terminar
            # el batch. Si el proceso se interrumpe, todo lo generado
            # hasta ese momento queda preservado.
            save_json(args.output, all_prompts)

            print(
                f"  ✓ Prompt generated and saved "
                f"({len(all_prompts)}/{args.number})"
            )

        print(f"\\n✓ Batch {batch_index} completed.")

        print(f"✓ Generated so far: {len(all_prompts)}/{args.number}")

        print(f"✓ Saved: {args.output}")
"""

if old not in text:
    raise RuntimeError("No se encontró el bloque de generación esperado.")

text = text.replace(old, new)

# 4) Añadir información explícita al encabezado de ejecución.
text = text.replace(
    'print(f"Files/batch:    {args.files_per_batch}")',
    'print(f"Files/batch:    {args.files_per_batch}")\n'
    '    print("Generation:     1 prompt per Ollama request")',
)

# 5) Ajustar el comentario de distribución para dejar claro que la
# distribución sigue siendo por batch, pero las llamadas son individuales.
text = text.replace("# DISTRIBUIR PROMPTS", "# DISTRIBUIR PROMPTS ENTRE BATCHES")

# 6) Verificaciones de seguridad.
if "ask_ollama(args.model, generation_prompt, amount)" in text:
    raise RuntimeError("Quedó una llamada antigua a ask_ollama.")

if "build_generation_prompt(batch, amount)" in text:
    raise RuntimeError("Quedó una llamada antigua a build_generation_prompt.")

out = Path("./")
out.write_text(text, encoding="utf-8")

# Validación de sintaxis.
compile(text, str(out), "exec")

print(f"Archivo creado: {out}")
print(f"Tamaño: {out.stat().st_size} bytes")
print("Sintaxis Python: OK")
print("\nLógica nueva:")
print("- Cada request a Ollama genera exactamente 1 prompt.")
print("- Cada request recibe nuevamente las 20 referencias del batch.")
print("- La distribución N entre los 4 batches se mantiene.")
print("- Se guarda prompts.json después de cada prompt.")
