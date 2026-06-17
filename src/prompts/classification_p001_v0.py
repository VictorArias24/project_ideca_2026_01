"""
Prompt p001 v0: Original version before 2026-06-14 COMERCIAL_1/MIXTO_1 refinements.

This is the prompt as it existed before:
- Added VERTICAL INSPECTION step in system_prompt.txt
- Enhanced COMERCIAL_1 with "ALL floors" requirement
- Enhanced MIXTO_1 with CRITICAL rule
- Added "Bogota Urban Pattern" note

This file is preserved for reference only. Active prompt is p001.py.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SYSTEM_PROMPT_PATH = _REPO_ROOT / "specs" / "system_prompt.txt"
_SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text().strip()

TAXONOMY_DEFINITIONS = """
## Bogota Building Use Taxonomy (10 classes + UNKNOWN)

### RESIDENCIAL_1 — Vivienda
Buildings for housing: houses, apartments, condominiums, residential towers.
Visual: balconies with laundry/plants, residential curtains, single-door
entrance with intercom, NO commercial signage.

### COMERCIAL_1 — Locales / Retail
Retail stores, shops, direct-to-public commerce.
Visual: product display windows, "abierto" signage, commercial security
grilles, public-access entrance.

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
Mixed residential AND retail in same structure. Ground-floor store + upper
housing.
Visual: display window + commercial sign on ground floor, residential
balconies/curtains above, two separate entrances.

### MIXTO_2 — Residencial + Comercial 2
Mixed residential AND offices in same structure.
Visual: professional plaque on residential facade, office entrance next to
residential entrance, building directory in residential building.

### UNKNOWN_OR_INSUFFICIENT_EVIDENCE
Use when images do not support a reliable decision (dark, fully obstructed,
no facade visible, absurd image, facade >80% covered).
"""

OUTPUT_SCHEMA_INSTRUCTION = """
Return ONLY valid JSON with this structure:
{
  "building_id": "string",
  "caracteristicas_visibles": ["visible feature in Spanish", ...],
  "clases": [
    {
      "label": "one of the 11 enum values above",
      "confidence": 0.0-1.0,
      "evidencia_visual": ["specific evidence from images", ...],
      "is_primary": true
    }
  ],
  "requiere_revision": true/false,
  "razon_revision": ["reason if requiere_revision=true"],
  "model_id": "Qwen2.5-VL-32B-Instruct",
  "perfil_modelo": "80gb_high_accuracy",
  "schema_version": "1.0.0",
  "taxonomy_version": "1.0.0"
}

Rules:
- Exactly ONE class must have "is_primary": true.
- Maximum 3 classes total.
- "confidence" must be between 0 and 1.
- "evidencia_visual" must cite specific visible elements from the images.
- "requiere_revision": true when:
  * max_confidence < 0.70
  * Top-2 classes have confidence difference < 0.10
  * Primary label is COMERCIAL_3, DOTACIONAL_2, or MOLES_1
  * Primary label is UNKNOWN_OR_INSUFFICIENT_EVIDENCE
  * Images are dark, blurred, obstructed >50%, or facade not visible
- "caracteristicas_visibles": list 1-12 observable building features
"""


def get_system_prompt() -> str:
    return _SYSTEM_PROMPT


def get_taxonomy_definitions() -> str:
    return TAXONOMY_DEFINITIONS


def get_output_schema_instruction() -> str:
    return OUTPUT_SCHEMA_INSTRUCTION


def build_user_prompt(building_id: str) -> str:
    return (
        TAXONOMY_DEFINITIONS
        + "\n---\n"
        + OUTPUT_SCHEMA_INSTRUCTION
        + "\n---\n"
        + f"Building ID: {building_id}\n\n"
        + "Classify the PRIMARY building using the taxonomy above "
        + "and return the JSON."
    )


def build_messages(
    image_data_urls: list[str],
    building_id: str = "unknown",
) -> list[dict]:
    content = [{"type": "text", "text": build_user_prompt(building_id)}]

    for url in image_data_urls:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": url}
            }
        )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
