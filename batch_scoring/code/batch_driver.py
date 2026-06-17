"""
Batch scoring driver for Azure ML Batch Endpoint.

Calls the online VLM endpoint with p004 classification context, processes
buildings one at a time, and returns a predictions DataFrame.

Self-contained — does not import project code. Runs on Azure ML compute.
"""

import os
import json
import base64
import re
import pandas as pd
from typing import List
from pathlib import Path
from io import BytesIO
from PIL import Image
from openai import OpenAI


# ── INIT ────────────────────────────────────────────────────────────────

def init():
    """Initialize OpenAI client to call the online endpoint."""
    global openai_client, model_name, deployment_name

    api_url = os.environ.get("ONLINE_API_URL", "")
    api_key = os.environ.get("ONLINE_API_KEY", "")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-VL-32B-Instruct")
    deployment_name = os.environ.get("ONLINE_DEPLOYMENT", "")

    if not api_url or not api_key:
        raise ValueError("ONLINE_API_URL and ONLINE_API_KEY must be set")

    base_url = api_url if api_url.endswith("/v1") else f"{api_url}/v1"

    openai_client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers={"azureml-model-deployment": deployment_name},
    )

    print(f"[INIT] Batch scorer ready — endpoint: {api_url}")


# ── SYSTEM PROMPT ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a building-use classifier specialized in the Bogota, Colombia 10-class
facade taxonomy. Your job is to analyze street-level facade images and determine
the building's primary use type from visible evidence only.

## Reasoning Protocol

1. **OBSERVE**: Scan all provided images. Identify visible architectural
   features: number of floors, facade materials, window types, entrance style,
   signage, balconies, construction state, and surrounding context.

2. **VERTICAL INSPECTION** (multi-story buildings only): Check every floor
   individually from top to bottom — NOT just the ground floor. In Bogota,
   ground-floor commerce with upper-floor residential is the default
   mixed-use pattern. For upper floors, look specifically for: balconies
   with laundry/plants, domestic curtains (not office blinds), residential-
   style windows, clotheslines, satellite dishes, separate residential
   entrance door.

3. **CROSS-CHECK**: Compare what you see against the class definitions
   provided in the user message. Pay special attention to "confusing
   neighbors" — classes that look similar but have key distinctions.

4. **EVIDENCE**: For each candidate class, list specific visible features
   that support it. Evidence must be observable in the images, not assumed.

5. **DECIDE**: Select the ONE primary class that best matches the visible
   evidence. If the building has genuine mixed use, note secondary classes
   but keep one primary.

6. **SELF-CHECK**: Before returning, verify:
   - Your evidence is based on what you can SEE, not what you assume
   - You checked confusing-neighbor distinctions
   - If images are insufficient, you flagged UNKNOWN_OR_INSUFFICIENT_EVIDENCE
   - If confidence is low, you set requiere_revision=true

## Output Rules

- Return ONLY valid JSON. No markdown, no explanation outside the JSON.
- One primary class (is_primary=true). Optional secondary classes (max 2).
- Every label must come from the taxonomy enum provided in the user message.
- Confidence must be 0-1. Be honest: 0.90+ = unambiguous, 0.70-0.89 = likely
  correct, 0.50-0.69 = plausible but uncertain, <0.50 = guess.
- caracteristicas_visibles: observable features in Spanish, 1-8 items.

## Review Flag Rules

Set requiere_revision=true when ANY of:
- max_confidence < 0.70
- Two top classes within 0.10 confidence of each other (ambiguous)
- Primary label is COMERCIAL_3, DOTACIONAL_2, or MOLES_1 (suggested classes)
- Images are dark, blurred, obstructed >50%, or facade not visible
- Building appears abandoned or facade is unreadable

## Language

- caracteristicas_visibles: Spanish (es)
- Field names: as specified in the user message (label, confidence, etc.)
- Do NOT translate class names. Use exactly the enum values provided.

## Important

This is NOT a general-purpose VLM task. You are applying a specific
taxonomic classification to Bogota building facades. When in doubt between
two classes, re-read their definitions in the user message and check the
distinguishing features. If still uncertain, mark for review rather than
guessing."""


# ── P004 CLASSIFICATION CONTEXT ─────────────────────────────────────────

TAXONOMY_DEFINITIONS = """
## Bogota Building Use Taxonomy (10 classes + UNKNOWN)

### RESIDENCIAL_1 — Vivienda
Buildings for housing: houses, apartments, condominiums, residential towers.
Visual: balconies with laundry/plants, residential curtains, single-door
entrance with intercom, NO commercial signage. May have metal security
grilles on ground floor (common in all Bogota buildings, NOT a commercial
indicator).
NEGATIVE: If ANY floor shows non-residential indicators (professional
plaques, office directory, glass-door entrance, AC units on windows,
business-hours signage, window blinds instead of curtains, absence of
domestic elements like laundry/plants on ALL floors), consider MIXTO_2.
A building with a residential facade that has been partially converted
to offices is MIXTO_2, not RESIDENCIAL_1.

### COMERCIAL_1 — Locales / Retail
Buildings where ALL floors are dedicated to retail/commercial use.
TRUE indicators (at least one required):
  - Visible commercial signage with business name (letrero con nombre del
    negocio, horario, promociones)
  - High customer flow: people entering/exiting, vehicles loading/unloading
  - Product display windows (vitrinas) visible from street
  - Public-access commercial entrance (puerta abierta al publico)
NOT indicators (present in residential too):
  - Metal security grilles (rejas metalicas) — universal in Bogota
  - Ground-floor appearance without signage
  - Roll-up shutters without business signage
NEGATIVE: If ANY upper floor shows residential indicators, classify as
MIXTO_1. A building is only COMERCIAL_1 if NO floor shows residential use
and at least one TRUE commercial indicator is present.

### COMERCIAL_2 — Oficinas / Professional Services
Offices, consulting, corporate buildings, banks, government offices.
Visual: professional plaques (lawyer, accountant), uniform office windows
with blinds, lobby directory, corporate logo. NO retail displays.

### COMERCIAL_3 — Servicios / Hospitality (SUGERIDO)
Hotels, motels, restaurants.
Visual: hotel reception, restaurant with tables/menu, motel with discreet
entrance and rate signage. SUGGESTED CLASS — always route to review.

### DOTACIONAL_1 — Servicios Comunales / Community Facilities
Community centers, health clinics, hospitals, religious temples, sports centers.
Visual: institutional signage, hospital emergency entrance, church cross/dome,
community center with official crest. Absorbs health, religious, recreational.

### DOTACIONAL_2 — Educacion / Education (SUGERIDO)
Schools, universities, kindergartens, training centers.
Visual: school signage, playground with swings, visible classrooms, students
in uniform. SUGGESTED CLASS — always route to review.

### MOLES_1 — Construcciones en Obra / Under Construction (SUGERIDO)
Large buildings under construction (>4 floors or >10,000 m2, stages 2-3).
Visual: construction mesh (green/orange/blue), scaffolding, tower crane,
bare concrete, curaduria/construction license notice. Transient state.
SUGGESTED CLASS — always route to review.

### RURAL_1 — Rurales / Rural
Rural/agricultural structures: sheds, barns, stables, chicken coops, silos.
Visual: wood/zinc structure, wire fencing, pasture, visible animals, unpaved
roads, non-urban context, single-story rustic construction.

### MIXTO_1 — Residencial + Comercial 1
Mixed residential AND retail in same structure. This is the most common
mixed-use pattern in Bogota: ground-floor store + upper-floor housing.
TRUE indicators (both required):
  a) Ground floor: commercial signage with business name, product display
     window, or visible customer flow AND a commercial entrance
  b) Upper floors: residential indicators — balconies with laundry/plants,
     domestic curtains (not office blinds), residential-style windows,
     clotheslines, clothes hung to dry, satellite dishes, separate
     residential entrance door
NOT indicators:
  - Metal security grilles on ground floor without commercial signage
  - Ground-floor appearance without letreros or customer flow
CRITICAL: if ground floor has TRUE commercial indicators AND upper floors
show residential indicators → MIXTO_1 (not COMERCIAL_1).

### MIXTO_2 — Residencial + Comercial 2
Mixed residential AND offices in same structure. Houses or residential
buildings partially converted to professional offices, consulting rooms,
or service practices.

DEFINITION: A building with RESIDENTIAL architecture (house, apartment
building — not purpose-built office tower) where part of the space has
been visibly converted to professional use. The residential character
comes from the BUILDING TYPE, not just currently-visible domestic items.
A house-shaped building with office plaques is MIXTO_2 even if no laundry
or plants are currently visible on balconies.

PRIMARY indicators (at least one required):
  - Professional plaque or sign on residential facade (abogado, contador,
    medico, consultorio, oficina, any business/professional name)
  - Office directory board or multiple small name plaques at entrance
  - Glass-door professional entrance on a house/apartment-style building
  - Window decals or stickers with business names/hours

SECONDARY indicators (look for these when plaques are not clearly readable):
  - Air conditioning split units on windows (common in converted offices)
  - Uniform blinds (persianas) instead of domestic curtains on SOME floors
  - Multiple doorbells/buzzers labeled with business names
  - Professional-grade entrance door (glass, aluminum frame) contrasting
    with residential-style windows above
  - Building has residential architecture (house shape, apartment layout)
    but lacks current domestic indicators on some floors

NEGATIVE: If the building has NO architectural residential character
(purpose-built office tower, uniform office windows on ALL floors, lobby
directory, corporate architecture) → COMERCIAL_2, not MIXTO_2.

CRITICAL: even if professional plaques are too small to read clearly,
secondary indicators + residential architecture are sufficient for MIXTO_2.

### UNKNOWN_OR_INSUFFICIENT_EVIDENCE
Use when images do not support a reliable decision (dark, fully obstructed,
no facade visible, absurd image, facade >80% covered).
"""

FLOOR_ANALYSIS_INSTRUCTION = """
## REQUIRED: Floor-by-Floor Analysis

Before selecting any class, you MUST list each visible floor and describe
exactly what you see. This is NOT optional:

1. Piso 1 (planta baja):
   - Letreros comerciales: [visible / no visible / ambiguo]
     If visible: list each business name or sign text you can read.
   - Vitrinas o exhibicion de productos: [visible / no visible]
   - Puerta de acceso: [comercial abierta al publico / residencial / ambiguo]
   - Actividad: [clientes entrando/saliendo / vehiculos cargando / sin actividad]
   - Rejas metalicas: [presentes / no presentes] (note: NOT a commercial indicator)

2. Piso 2:
   - Balcones: [presentes / no presentes]
     If present: [con ropa tendida/plantas / vacios / cerrados]
   - Cortinas: [domesticas / persianas de oficina / no visibles]
   - Tipo de ventana: [residencial / comercial / no se puede determinar]
   - Indicadores residenciales: [ropa tendida / plantas / antenas / ninguno]

3. Piso 3 (if exists): [same as piso 2]

4. Pisos adicionales: [same as piso 2]
"""

DECISION_TABLE = """
## Decision Table

Based on your floor-by-floor analysis, apply this table in order:

### COMERCIAL_1 / MIXTO_1 / RESIDENCIAL_1

| Ground Floor                    | Upper Floors              | Classification      | Confidence       |
|---------------------------------|---------------------------|---------------------|-------------------|
| Letreros VISIBLES + actividad   | Residencial VISIBLE        | MIXTO_1             | 0.80 - 0.95      |
| Letreros VISIBLES + actividad   | NO residencial (oficina)   | COMERCIAL_1         | 0.75 - 0.90      |
| Letreros VISIBLES + actividad   | AMBIGUO (no se ve bien)    | MIXTO_1 + REVIEW    | 0.60 - 0.75      |
| SIN letreros (solo rejas)       | Residencial VISIBLE        | RESIDENCIAL_1       | 0.75 - 0.90      |
| SIN letreros (solo rejas)       | AMBIGUO                    | RESIDENCIAL_1 + REVIEW | 0.55 - 0.70  |
| AMBIGUO (letreros ilegibles)    | Residencial VISIBLE        | RESIDENCIAL_1 + REVIEW | 0.60 - 0.75  |
| AMBIGUO (letreros ilegibles)    | AMBIGUO                    | UNKNOWN + REVIEW    | < 0.50           |
| Comercial visible               | Comercial visible          | COMERCIAL_1         | 0.80 - 0.95      |

### MIXTO_2 (Residencial + Oficinas)

| Facade Indicators                                   | Classification      | Confidence       |
|-----------------------------------------------------|---------------------|-------------------|
| Placa profesional VISIBLE + elementos residenciales | MIXTO_2             | 0.80 - 0.90      |
| Placa profesional VISIBLE + SIN residencial claro   | MIXTO_2             | 0.70 - 0.85      |
| Placa ILEGIBLE + indicadores SECUNDARIOS presentes  | MIXTO_2 + REVIEW    | 0.60 - 0.75      |
| Placa ILEGIBLE + SIN indicadores secundarios        | RESIDENCIAL_1 + REVIEW | 0.55 - 0.70  |
| SIN placa + indicadores SECUNDARIOS visibles        | MIXTO_2 + REVIEW    | 0.55 - 0.70      |
| SIN placa + SIN indicadores secundarios             | RESIDENCIAL_1       | 0.75 - 0.90      |

Indicadores SECUNDARIOS: AC splits on windows, uniform blinds (not domestic
curtains), glass/aluminum entrance door contrasting with residential windows,
absence of domestic elements (no laundry, no plants, no residential curtains)
on floors that appear non-residential, multiple labeled doorbells/buzzers,
visible office furniture through windows.

CRITICAL RULES:
- Metal security grilles (rejas) on ground floor are NOT commercial evidence.
  If the ONLY ground-floor indicator is rejas, treat as "SIN letreros" above.
- If you CANNOT read the letrero/placa text clearly, treat as "AMBIGUO" or
  "ILEGIBLE", not "VISIBLE".
- When the decision table says "+ REVIEW", requiere_revision MUST be true.
- Default: if uncertain between COMERCIAL_1 and MIXTO_1, prefer MIXTO_1.
- Default: if uncertain between RESIDENCIAL_1 and MIXTO_1, flag REVIEW.
- Default: if uncertain between RESIDENCIAL_1 and MIXTO_2, check for secondary
  office indicators. If ANY present → MIXTO_2 + REVIEW. If none → REVIEW
  (don't confidently guess RESIDENCIAL_1 when office conversion is possible).
- MIXTO_2 vs COMERCIAL_2: MIXTO_2 has RESIDENTIAL ARCHITECTURE (house,
  apartment building) with office conversion elements. COMERCIAL_2 is a
  purpose-built office building (corporate architecture, uniform office
  windows, lobby directory). A house-shaped building with office plaques
  is MIXTO_2 even if no domestic elements (laundry, plants) are visible.
- Building SCALE matters: small buildings (1-3 floors, house-scale, narrow
  facade) with professional signs are MIXTO_2. COMERCIAL_2 implies larger
  buildings (3+ floors of uniform office windows, wider facade, corporate
  design). A small converted house with a professional plaque is MIXTO_2,
  not COMERCIAL_2.
- Absence of domestic elements alone is NOT enough to call a building
  COMERCIAL_2. You need POSITIVE purpose-built-office architecture.
"""

BATCH_OUTPUT_INSTRUCTION = """
## BATCH OUTPUT FORMAT

For this batch processing run, return ONLY a compact JSON with:
{
  "label": "one of [RESIDENCIAL_1, COMERCIAL_1, COMERCIAL_2, COMERCIAL_3, DOTACIONAL_1, DOTACIONAL_2, MOLES_1, RURAL_1, MIXTO_1, MIXTO_2, UNKNOWN_OR_INSUFFICIENT_EVIDENCE]",
  "confidence": 0.XX
}

Apply ALL the reasoning, floor-by-floor analysis, and decision table rules above.
But output ONLY the label and confidence — no other fields.
No markdown fences. No additional text. Just the JSON object.
"""

FULL_USER_PROMPT = (
    TAXONOMY_DEFINITIONS
    + "\n---\n"
    + FLOOR_ANALYSIS_INSTRUCTION
    + "\n---\n"
    + DECISION_TABLE
    + "\n---\n"
    + BATCH_OUTPUT_INSTRUCTION
)


# ── IMAGE ENCODING ──────────────────────────────────────────────────────

# Set to 0 to keep native resolution (default). Set BATCH_IMAGE_MAX_DIM=400 to
# resize to 400px (legacy behavior, useful when token budget is tight).
MAX_IMAGE_SIZE = int(os.getenv("BATCH_IMAGE_MAX_DIM", "0"))


def _encode_image(image_path: Path) -> str:
    """Encode an image to a data: URL, resizing and converting to JPEG if needed.

    Resize is skipped when MAX_IMAGE_SIZE <= 0 (default). Set the
    BATCH_IMAGE_MAX_DIM environment variable to opt back into 400px resize.
    """
    ext = image_path.suffix.lower()
    mime = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}.get(ext, "jpeg")

    img = Image.open(image_path)
    w, h = img.size

    # Resize if MAX_IMAGE_SIZE > 0 and image is larger
    if MAX_IMAGE_SIZE > 0 and max(w, h) > MAX_IMAGE_SIZE:
        factor = MAX_IMAGE_SIZE / max(w, h)
        img = img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)

    # Ensure RGB for JPEG output (handle RGBA, LA, P, PA modes)
    if img.mode in ("RGBA", "LA", "P", "PA"):
        if img.mode == "P":
            img = img.convert("RGBA")
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        mask = img.split()[-1] if img.mode in ("RGBA", "LA", "PA") else None
        rgb_img.paste(img, mask=mask)
        img = rgb_img
    elif img.mode != "RGB":
        img = img.convert("RGB")

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{data}"


# ── JSON EXTRACTION ─────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    """Extract the first valid JSON object from VLM response text."""
    text = text.strip()

    # Remove markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")

    # Find the first { ... } pair
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


# ── CLASSIFY ────────────────────────────────────────────────────────────

def _classify_building(building_id: str, image_paths: List[Path]) -> dict:
    """Send all images for one building in a single VLM call with p004 context."""
    content = [{"type": "text", "text": FULL_USER_PROMPT}]

    for path in image_paths:
        data_url = _encode_image(path)
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    response = openai_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=512,
    )

    text = response.choices[0].message.content
    parsed = _extract_json(text)

    if parsed:
        label = parsed.get("label", "PARSE_ERROR")
        confidence = parsed.get("confidence", 0)
    else:
        label = "PARSE_ERROR"
        confidence = 0
        print(f"[WARN] {building_id}: JSON parse failed: {text[:150]}")

    return {"label": label, "confidence": confidence}


# ── BATCH RUN ───────────────────────────────────────────────────────────

def run(mini_batch: List[str]) -> pd.DataFrame:
    """Process a mini-batch of files from the data asset.

    Expects _task.json files mixed with image files. Groups images by
    building directory and classifies each building once.
    """
    results = []
    processed = set()

    # Group _task.json files by parent directory
    for file_path_str in mini_batch:
        file_path = Path(file_path_str)

        if file_path.name != "_task.json":
            continue

        building_dir = file_path.parent
        building_id = building_dir.name

        if building_id in processed:
            continue
        processed.add(building_id)

        try:
            with open(file_path, "r") as f:
                task = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            results.append({
                "building_id": building_id,
                "label": "ERROR",
                "confidence": 0,
                "images_used": 0,
                "status": f"task_read_error: {str(e)[:100]}",
            })
            continue

        image_paths = [building_dir / name for name in task.get("image_files", [])]

        # Verify all images exist
        missing = [p for p in image_paths if not p.exists()]
        if missing:
            results.append({
                "building_id": building_id,
                "label": "ERROR",
                "confidence": 0,
                "images_used": len(image_paths),
                "status": f"missing_images: {[p.name for p in missing]}",
            })
            continue

        try:
            pred = _classify_building(building_id, image_paths)
            results.append({
                "building_id": building_id,
                "label": pred["label"],
                "confidence": pred["confidence"],
                "images_used": len(image_paths),
                "status": "success",
            })
            print(f"[OK] {building_id} ({len(image_paths)} img) -> {pred['label']} ({pred['confidence']})")

        except Exception as e:
            results.append({
                "building_id": building_id,
                "label": "ERROR",
                "confidence": 0,
                "images_used": len(image_paths),
                "status": f"error: {str(e)[:150]}",
            })
            print(f"[ERR] {building_id}: {e}")

    print(f"[DONE] Processed {len(results)} buildings in this mini-batch")
    return pd.DataFrame(results)
