"""
Prompt p001: Full taxonomy classification prompt.

Loads the system prompt from specs/system_prompt.txt and uses
the same concise 11-class taxonomy definitions proven in Phase 02
endpoint tests. No new prompt content — just repackaging what works.

Language: Spanish evidence (caracteristicas_visibles, evidencia_visual),
          English label IDs (RESIDENCIAL_1, etc.).
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# System prompt loaded from version-controlled file (Phase 02 artifact)
_SYSTEM_PROMPT_PATH = _REPO_ROOT / "specs" / "system_prompt.txt"
_SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text().strip()

# Taxonomy definitions — same concisetext proven in Phase 02 test scripts.
# Full operational definitions live in specs/taxonomy_operational.yaml.
TAXONOMY_DEFINITIONS = """
## Bogota Building Use Taxonomy (10 classes + UNKNOWN)

### RESIDENCIAL_1 — Vivienda
Buildings for housing: houses, apartments, condominiums, residential towers.
Visual: balconies with laundry/plants, residential curtains, single-door
entrance with intercom, NO commercial signage.

### COMERCIAL_1 — Locales / Retail
Buildings where ALL floors are dedicated to retail/commercial use. Every
visible floor shows commercial activity: product display windows, commercial
signage, security grilles, public-access entrances.
NEGATIVE: If upper floors show residential indicators (balconies, laundry,
domestic curtains, residential entrance), this is MIXTO_1, NOT COMERCIAL_1.
A building is only COMERCIAL_1 if NO floor shows residential use.

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
Visual: display window + commercial sign on ground floor, AND residential
balconies/curtains/clotheslines on upper floors, two separate entrances
(one commercial, one residential).
CRITICAL: if you see commerce on ground floor AND ANY residential
indicators above, classify as MIXTO_1, not COMERCIAL_1.

### MIXTO_2 — Residencial + Comercial 2
Mixed residential AND offices in same structure.
Visual: professional plaque on residential facade, office entrance next to
residential entrance, building directory in residential building.

### UNKNOWN_OR_INSUFFICIENT_EVIDENCE
Use when images do not support a reliable decision (dark, fully obstructed,
no facade visible, absurd image, facade >80% covered).

## Bogota Urban Pattern
In Bogota multi-story buildings, ground-floor commercial + upper-floor
residential is extremely common. ALWAYS check upper floors for residential
indicators before classifying as purely commercial. When in doubt between
COMERCIAL_1 and MIXTO_1, prefer MIXTO_1 and set requiere_revision=true.
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
    """Return the system prompt (model role, reasoning protocol, rules)."""
    return _SYSTEM_PROMPT


def get_taxonomy_definitions() -> str:
    """Return concise 11-class taxonomy definitions."""
    return TAXONOMY_DEFINITIONS


def get_output_schema_instruction() -> str:
    """Return the JSON output format and validation rules."""
    return OUTPUT_SCHEMA_INSTRUCTION


def build_user_prompt(building_id: str) -> str:
    """Build the full user prompt: taxonomy + schema + instruction."""
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
    """Build the full message list for the VLM API call.

    Returns list of messages with system prompt + user content (taxonomy + images).
    """
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
