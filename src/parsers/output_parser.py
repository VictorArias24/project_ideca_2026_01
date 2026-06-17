"""
Output parser: Extracts and validates JSON from VLM responses.

Implements the parse pipeline from Phase 02 test scripts as reusable
functions: extract JSON, validate against taxonomy schema, check required
fields, exactly-one-primary constraint, valid labels, confidence range.

Usage:
    from src.parsers.output_parser import parse_and_validate
    result, error = parse_and_validate(raw_vlm_text)
    if result:
        print(result["clases"][0]["label"])
"""

import json
import re
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Valid labels from locked Phase 01 taxonomy + UNKNOWN
VALID_LABELS = frozenset({
    "RESIDENCIAL_1", "COMERCIAL_1", "COMERCIAL_2", "COMERCIAL_3",
    "DOTACIONAL_1", "DOTACIONAL_2", "MOLES_1", "RURAL_1",
    "MIXTO_1", "MIXTO_2", "UNKNOWN_OR_INSUFFICIENT_EVIDENCE",
})

VALID_PROFILES = frozenset({"24gb_fast", "80gb_high_accuracy", "api_baseline"})

REQUIRED_FIELDS = [
    "building_id", "caracteristicas_visibles", "clases",
    "requiere_revision", "model_id", "perfil_modelo",
    "schema_version", "taxonomy_version",
]


def extract_json(text: str) -> Optional[str]:
    """Extract JSON string from model output.

    Strategy 1: Find JSON code block (```json ... ``` or ``` ... ```)
    Strategy 2: Find outermost {} pair by tracking brace depth.
    """
    # Strategy 1: code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 2: outermost {} pair
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def validate_output(data: dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate parsed JSON against output schema requirements.

    Returns (is_valid, error_message).
    """
    # Required top-level fields
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing required field: {field}"

    # building_id
    bid = data["building_id"]
    if not isinstance(bid, str) or len(bid) == 0:
        return False, "building_id must be a non-empty string"

    # caracteristicas_visibles
    cv = data["caracteristicas_visibles"]
    if not isinstance(cv, list) or len(cv) < 1 or len(cv) > 12:
        return False, (
            f"caracteristicas_visibles must be 1-12 items, "
            f"got {len(cv) if isinstance(cv, list) else type(cv).__name__}"
        )

    # clases
    clases = data.get("clases", [])
    if not isinstance(clases, list) or len(clases) < 1 or len(clases) > 3:
        return False, (
            f"clases must be 1-3 items, "
            f"got {len(clases) if isinstance(clases, list) else type(clases).__name__}"
        )

    primary_count = 0
    for i, clase in enumerate(clases):
        # label
        label = clase.get("label", "")
        if label not in VALID_LABELS:
            return False, f"Invalid label in clases[{i}]: {label}"

        # confidence
        conf = clase.get("confidence")
        if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
            return False, f"Invalid confidence in clases[{i}]: {conf}"

        # evidencia_visual
        ev = clase.get("evidencia_visual", [])
        if not isinstance(ev, list) or len(ev) < 1 or len(ev) > 8:
            return False, (
                f"evidencia_visual in clases[{i}] must be 1-8 items, "
                f"got {len(ev) if isinstance(ev, list) else type(ev).__name__}"
            )

        # is_primary
        if clase.get("is_primary"):
            primary_count += 1

    if primary_count != 1:
        return False, f"Exactly 1 primary class required, got {primary_count}"

    # requiere_revision
    if not isinstance(data.get("requiere_revision"), bool):
        return False, "requiere_revision must be boolean"

    # perfil_modelo
    if data.get("perfil_modelo") not in VALID_PROFILES:
        return False, f"Invalid perfil_modelo: {data.get('perfil_modelo')}"

    return True, None


def parse_and_validate(text: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Full parse pipeline: extract JSON -> parse -> validate.

    Returns (parsed_dict_or_None, error_message_or_None).
    """
    json_str = extract_json(text)
    if json_str is None:
        return None, "No JSON found in response"

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"

    is_valid, error = validate_output(data)
    if not is_valid:
        return None, error

    return data, None
