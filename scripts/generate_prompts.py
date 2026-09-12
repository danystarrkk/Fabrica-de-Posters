#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import requests

# ============================================================
# CONFIGURACIÓN
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:31b"

# Número máximo de archivos de referencia por grupo.
# Con 80 archivos tendremos 4 grupos de 20.
FILES_PER_BATCH = 20

# Archivo de salida por defecto
DEFAULT_OUTPUT = "prompts.json"

# Timeout de Ollama.
REQUEST_TIMEOUT = 3600


# ============================================================
# PROMPT MAESTRO
# ============================================================

SYSTEM_INSTRUCTION = r"""
You are generating one original image-generation prompt.

You will receive a collection of reference prompts.

These reference prompts are your ONLY reference and source of inspiration.

Study ALL of the reference prompts carefully before generating.

Your task is to create ONE completely new, original, innovative,
and visually compelling prompt based on what you learn from the
reference prompts.

Do NOT copy or reproduce the content of the references.

Do NOT copy sentences or distinctive phrases.

Do NOT paraphrase any reference.

Do NOT reproduce the same scene, character, setting, composition,
or concept from a reference.

Do NOT simply combine elements from multiple references.

Instead, study the references as a whole and create something
genuinely new that feels naturally related to them.

Let the reference prompts determine everything about the resulting prompt:
- how it is written
- how much it describes
- how it is structured
- its vocabulary
- its level of detail
- its visual language
- its composition
- its subjects
- its environments
- its lighting
- its colors
- its materials
- its textures
- its atmosphere
- its mood
- its artistic language
- its visual storytelling
- and any other characteristic present in the references

Do not impose an external style definition.

Do not impose characteristics that are not supported by the references.

The references are the authority.

Create something new, striking, coherent, imaginative,
and visually interesting while remaining faithful to
what you learned from ALL of the reference prompts.

EVERY generated prompt MUST begin exactly with:

"t3chgrng style,"

Return ONLY valid JSON using exactly this structure:

{
  "prompts": [
    "t3chgrng style, ..."
  ]
}

The "prompts" array MUST contain exactly ONE string.

Do not add a title.

Do not add explanations.

Do not add markdown.

Do not add anything before or after the JSON object.
"""


# ============================================================
# ARGUMENTOS
# ============================================================


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate image prompts from TXT reference files using Ollama."
    )

    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing the TXT reference files.",
    )

    parser.add_argument(
        "number",
        type=int,
        help="Total number of prompts to generate.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output JSON file. Default: {DEFAULT_OUTPUT}",
    )

    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Ollama model. Default: {MODEL}",
    )

    parser.add_argument(
        "--files-per-batch",
        type=int,
        default=FILES_PER_BATCH,
        help=f"Reference TXT files per batch. Default: {FILES_PER_BATCH}",
    )

    return parser.parse_args()


# ============================================================
# LEER TXT
# ============================================================


def load_txt_files(directory: Path):
    if not directory.exists():
        print(f"ERROR: Directory does not exist: {directory}")
        sys.exit(1)

    if not directory.is_dir():
        print(f"ERROR: Not a directory: {directory}")
        sys.exit(1)

    files = sorted(
        [
            file
            for file in directory.iterdir()
            if file.is_file() and file.suffix.lower() == ".txt"
        ]
    )

    if not files:
        print(f"ERROR: No .txt files found in: {directory}")
        sys.exit(1)

    references = []

    print(f"\nFound {len(files)} TXT reference files.\n")

    for index, file in enumerate(files, start=1):
        try:
            text = file.read_text(encoding="utf-8").strip()

        except UnicodeDecodeError:
            print(f"WARNING: Could not read UTF-8 file: {file}")
            continue

        except Exception as e:
            print(f"WARNING: Could not read {file}: {e}")
            continue

        if not text:
            print(f"WARNING: Empty file skipped: {file}")
            continue

        references.append(
            {
                "name": file.name,
                "text": text,
            }
        )

        print(f"  [{index:03d}] {file.name}")

    if not references:
        print("ERROR: No readable TXT files found.")
        sys.exit(1)

    return references


# ============================================================
# DISTRIBUIR PROMPTS
# ============================================================


def distribute_prompts(total_prompts, number_of_batches):
    """
    Distributes N prompts as evenly as possible.

    Examples:

    4 prompts / 4 batches
    -> [1, 1, 1, 1]

    7 prompts / 4 batches
    -> [2, 2, 2, 1]

    10 prompts / 4 batches
    -> [3, 3, 2, 2]

    100 prompts / 4 batches
    -> [25, 25, 25, 25]
    """

    base = total_prompts // number_of_batches
    remainder = total_prompts % number_of_batches

    distribution = []

    for i in range(number_of_batches):
        amount = base + (1 if i < remainder else 0)
        distribution.append(amount)

    return distribution


# ============================================================
# CONSTRUIR REFERENCIAS
# ============================================================


def build_reference_text(batch):
    sections = []

    for index, reference in enumerate(batch, start=1):
        sections.append(f"""
==============================
REFERENCE PROMPT {index}
FILE: {reference["name"]}
==============================

{reference["text"]}

==============================
END REFERENCE {index}
==============================
""")

    return "\n".join(sections)


# ============================================================
# LLAMAR A OLLAMA
# ============================================================


def ask_ollama(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.9},
    }

    print(f"  Sending request to Ollama ({model})...")

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    if "response" not in data:
        raise RuntimeError("Ollama response does not contain 'response'.")

    raw_response = data["response"].strip()

    try:
        result = json.loads(raw_response)

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Ollama returned invalid JSON: {e}\n\n"
            f"Raw response:\n{raw_response[:2000]}"
        )

    if not isinstance(result, dict):
        raise RuntimeError("JSON root must be an object.")

    prompts = result.get("prompts")

    if not isinstance(prompts, list):
        raise RuntimeError("JSON does not contain a 'prompts' array.")

    if len(prompts) != 1:
        raise RuntimeError(
            f"Expected exactly 1 prompt, but Ollama returned {len(prompts)}."
        )

    cleaned_prompts = []

    for index, item in enumerate(prompts, start=1):

        if not isinstance(item, str):
            raise RuntimeError(f"Prompt #{index} is not a string.")

        item = item.strip()

        if not item:
            raise RuntimeError(f"Prompt #{index} is empty.")

        # Garantizar que todos comiencen correctamente.
        if not item.startswith("t3chgrng style,"):
            item = "t3chgrng style, " + item

        cleaned_prompts.append(item)

    return cleaned_prompts


# ============================================================
# CONSTRUIR PROMPT DE GENERACIÓN
# ============================================================


def build_generation_prompt(batch):
    references = build_reference_text(batch)

    prompt = f"""
{SYSTEM_INSTRUCTION}

Here are the reference prompts:

==============================
REFERENCE MATERIAL
==============================

{references}

==============================
END REFERENCE MATERIAL
==============================

Generate ONE completely new, original, innovative,
and visually compelling prompt based on your study
of ALL the reference prompts.

Use the references as the sole basis for deciding
how the generated prompt should be written and what
kind of visual result it should produce.

Return only the required JSON.
"""

    return prompt


# ============================================================
# GUARDAR JSON
# ============================================================


def save_json(output_path, prompts):
    data = {
        "prompts": [
            {
                "id": index,
                "prompt": prompt,
            }
            for index, prompt in enumerate(prompts, start=1)
        ]
    }

    # Escritura atómica:
    # primero guardamos un archivo temporal y después lo
    # reemplazamos. Así evitamos dejar un JSON corrupto.
    temp_path = output_path.with_suffix(".tmp")

    temp_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temp_path.replace(output_path)


# ============================================================
# MAIN
# ============================================================


def main():

    args = parse_arguments()

    if args.number <= 0:
        print("ERROR: Number of prompts must be greater than 0.")
        sys.exit(1)

    if args.files_per_batch <= 0:
        print("ERROR: files-per-batch must be greater than 0.")
        sys.exit(1)

    print("=" * 70)
    print("       T3CHGRNG PROMPT GENERATOR")
    print("=" * 70)

    print(f"\nModel:          {args.model}")
    print(f"References:     {args.directory}")
    print(f"Prompts wanted: {args.number}")
    print(f"Files/batch:    {args.files_per_batch}")
    print(f"Output:         {args.output}")

    # --------------------------------------------------------
    # Cargar referencias
    # --------------------------------------------------------

    references = load_txt_files(args.directory)

    # --------------------------------------------------------
    # Dividir referencias
    # --------------------------------------------------------

    batches = [
        references[i : i + args.files_per_batch]
        for i in range(0, len(references), args.files_per_batch)
    ]

    number_of_batches = len(batches)

    print(f"\nReference batches: {number_of_batches}")

    # --------------------------------------------------------
    # Distribuir cantidad de prompts
    # --------------------------------------------------------

    distribution = distribute_prompts(
        args.number,
        number_of_batches,
    )

    print("\nPrompt distribution:")

    for index, amount in enumerate(distribution, start=1):
        print(
            f"  Batch {index}: "
            f"{len(batches[index - 1])} references → "
            f"{amount} prompts"
        )

    # --------------------------------------------------------
    # Generación
    # --------------------------------------------------------

    all_prompts = []

    for batch_index, (batch, amount) in enumerate(
        zip(batches, distribution),
        start=1,
    ):

        if amount == 0:
            continue

        print("\n" + "=" * 70)
        print(f"BATCH {batch_index}/{number_of_batches}")
        print("=" * 70)

        print(f"\nReferences: " f"{batch[0]['name']} → {batch[-1]['name']}")

        print(f"Prompts assigned to this batch: {amount}")

        # Cada prompt se genera mediante una llamada independiente.
        # Las mismas referencias del batch se vuelven a enviar
        # en cada llamada.
        for prompt_number in range(1, amount + 1):

            print("\n" + "-" * 70)
            print(
                f"Generating prompt "
                f"{prompt_number}/{amount} "
                f"in batch {batch_index}..."
            )
            print("-" * 70)

            generation_prompt = build_generation_prompt(batch)

            success = False
            max_attempts = 3

            for attempt in range(1, max_attempts + 1):

                try:
                    prompts = ask_ollama(
                        args.model,
                        generation_prompt,
                    )

                    if len(prompts) != 1:
                        raise RuntimeError(
                            f"Expected exactly 1 prompt, "
                            f"but received {len(prompts)}."
                        )

                    success = True
                    break

                except Exception as e:

                    print(f"\n  ERROR on attempt " f"{attempt}/{max_attempts}:")

                    print(f"  {e}")

                    if attempt < max_attempts:

                        print("\n  Retrying in 3 seconds...")

                        time.sleep(3)

            if not success:

                print("\nERROR: Could not generate this prompt.")

                print(
                    "Already generated prompts have been "
                    "preserved in the output file."
                )

                sys.exit(1)

            # La respuesta contiene exactamente un prompt.
            all_prompts.append(prompts[0])

            # Guardar inmediatamente después de cada prompt
            # para no perder progreso si una llamada posterior falla.
            save_json(
                args.output,
                all_prompts,
            )

            print(f"\n✓ Prompt generated: " f"{len(all_prompts)}/{args.number}")

            print(f"✓ Saved: {args.output}")

    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    print(f"\nTotal prompts generated: " f"{len(all_prompts)}")

    print(f"JSON saved to: " f"{args.output}")

    print()


if __name__ == "__main__":
    main()
