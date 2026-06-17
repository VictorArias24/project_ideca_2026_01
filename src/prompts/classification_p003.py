"""
Prompt p003: Floor-by-floor structured observation with confidence guardrails.

Version 3 — built from p002 with p002 test-failure analysis:

Key refinements over p002:
  - FORCES floor-by-floor observation BEFORE any classification decision.
    The model must describe what it sees on each floor independently.
  - Decision table: maps floor-level evidence to classification outcome
    (e.g., signage below + residential above → MIXTO_1).
  - Confidence calibration: if any floor is ambiguous, confidence drops.
  - Guardrail: if neither commercial nor residential indicators are clear,
    flag for review instead of guessing RESIDENCIAL_1.
  - NO default guess. The model must cite specific evidence per floor
    or flag UNKNOWN.

Language: Spanish evidence (caracteristicas_visibles, evidencia_visual),
          English label IDs (RESIDENCIAL_1, etc.).
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
entrance with intercom, NO commercial signage. May have metal security
grilles on ground floor (common in all Bogota buildings, NOT a commercial
indicator).

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
Mixed residential AND offices in same structure.
Visual: professional plaque on residential facade, office entrance next to
residential entrance, building directory in residential building.

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

CRITICAL RULES:
- Metal security grilles (rejas) on ground floor are NOT commercial evidence.
  If the ONLY ground-floor indicator is rejas, treat as "SIN letreros" above.
- If you CANNOT read the letrero text clearly, treat as "AMBIGUO", not "VISIBLE".
- When the decision table says "+ REVIEW", requiere_revision MUST be true.
- Default: if uncertain between COMERCIAL_1 and MIXTO_1, prefer MIXTO_1.
- Default: if uncertain between RESIDENCIAL_1 and MIXTO_1, flag REVIEW.
"""

OUTPUT_SCHEMA_INSTRUCTION = """
Return ONLY valid JSON with this structure:
{
  "building_id": "string",
  "analisis_por_piso": {
    "piso_1": "description of ground floor observations in Spanish",
    "piso_2": "description of second floor in Spanish",
    "pisos_adicionales": "description of additional floors if any, or empty string"
  },
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
- "confidence" must be between 0 and 1. Use the Decision Table ranges above.
  Do NOT default to 0.85. Vary confidence based on evidence clarity.
- "analisis_por_piso" is REQUIRED. You must fill piso_1, piso_2, and
  pisos_adicionales (empty string if only 2 floors). Describe what you
  actually SEE, not what you assume.
- "evidencia_visual" must cite specific visible elements from the images.
- "requiere_revision": true when:
  * max_confidence < 0.70
  * Top-2 classes have confidence difference < 0.10
  * Primary label is COMERCIAL_3, DOTACIONAL_2, or MOLES_1
  * Primary label is UNKNOWN_OR_INSUFFICIENT_EVIDENCE
  * Images are dark, blurred, obstructed >50%, or facade not visible
  * Decision Table says "+ REVIEW"
  * Ground floor indicators are ambiguous (no readable letreros, no clear
    customer flow, only rejas)
- "caracteristicas_visibles": list 1-12 observable building features
"""


def get_system_prompt() -> str:
    """Return the system prompt (model role, reasoning protocol, rules)."""
    return _SYSTEM_PROMPT


def get_taxonomy_definitions() -> str:
    """Return refined 11-class taxonomy definitions (p003)."""
    return TAXONOMY_DEFINITIONS


def get_output_schema_instruction() -> str:
    """Return the JSON output format including floor analysis and decision table."""
    return FLOOR_ANALYSIS_INSTRUCTION + "\n---\n" + DECISION_TABLE + "\n---\n" + OUTPUT_SCHEMA_INSTRUCTION


def build_user_prompt(building_id: str) -> str:
    """Build the full user prompt: taxonomy + floor analysis + decision table + schema."""
    return (
        TAXONOMY_DEFINITIONS
        + "\n---\n"
        + get_output_schema_instruction()
        + "\n---\n"
        + f"Building ID: {building_id}\n\n"
        + "STEP 1: Analyze each floor using the Floor-by-Floor Analysis above.\n"
        + "STEP 2: Apply the Decision Table to select the primary class and confidence.\n"
        + "STEP 3: Return the JSON with analisis_por_piso filled in.\n"
        + "REMEMBER: rejas metalicas are NOT commercial indicators in Bogota.\n"
        + "If you cannot read letreros clearly, they are AMBIGUO, not VISIBLE."
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
