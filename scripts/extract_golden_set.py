"""
Extract a bootstrap golden set from the expert-annotated docx.

Parses Victor_Casos Modelo Usos x Fachadas.docx to map each embedded
image to its expert-defined section, producing a ground-truth JSON file
for Phase 05 evaluation.

Output: experiments/golden_set.json
"""

import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCX_PATH = REPO / "docs" / "Victor_Casos Modelo Usos x Fachadas.docx"
OUTPUT_PATH = REPO / "experiments" / "golden_set.json"

# Section text (or sub-section) → label.
# The docx has parent headings (e.g. "PREDIOS CLARAMENTE RESIDENCIALES POR TIPOLOGÍAS")
# and sub-headings (e.g. "TIPOLOGÍA 1 ∞ Estrato 1"). Both map to the same label.
SECTION_LABEL_MAP = {
    "PREDIOS CLARAMENTE RESIDENCIALES POR TIPOLOGÍAS": "RESIDENCIAL_1",
    "PREDIOS CLARAMENTE COMERCIALES": "COMERCIAL_1",
    "PREDIOS DE OFICINAS": "COMERCIAL_2",
    "PREDIOS CLARAMENTE DOTACIONALES": "DOTACIONAL_1",
    "PREDIOS TIPO MIXTO COMERCIO CON VIVIENDA": "MIXTO_1",
    "PREDIOS RESIDENCIALES CON USO DE OFICINAS (COMERCIAL)": "MIXTO_2",
    "PREDIOS EN AVANCES DE OBRA": "MOLES_1",
    "BODEGAS": None,  # parent — resolved by sub-sections
}

# Sub-section overrides. These map specific sub-headings to labels.
# Any sub-heading NOT in this dict inherits the parent section label.
SUB_SECTION_OVERRIDES = {
    # Residential sub-strata
    "TIPOLOGÍA 1": "RESIDENCIAL_1",
    "TIPOLOGÍA 2": "RESIDENCIAL_1",
    "TIPOLOGÍA 3": "RESIDENCIAL_1",
    "TIPOLOGÍA 4": "RESIDENCIAL_1",
    "TIPOLOGÍA 5": "RESIDENCIAL_1",
    "TIPOLOGÍA 6": "RESIDENCIAL_1",
    # BODEGAS children
    "Templo de culto": "DOTACIONAL_1",
    # AVANCES DE OBRA explanatory text — maps to parent
    "Avisos de curaduría u otras entidades": "MOLES_1",
}

# Sub-section patterns that should NOT be treated as new sections.
# These are descriptive text, not section changes.
SKIP_SUB_PATTERNS = [
    r"^Podrían ser",
    r"^Industrias",
    r"^Comercio artesanal",
    r"^Depósitos",
    r"^Bodegas",
    r"^y \d+$",
]


def _resolve_label(section_text: str) -> str | None:
    """Map a section or sub-section text to a taxonomy label."""
    text = section_text.strip()
    if not text:
        return None

    # Exact match
    if text in SECTION_LABEL_MAP:
        label = SECTION_LABEL_MAP[text]
        return label if label is not None else _resolve_sub(text)

    # Sub-section match
    return _resolve_sub(text)


def _resolve_sub(text: str) -> str | None:
    for pattern, label in SUB_SECTION_OVERRIDES.items():
        if pattern in text:
            return label

    for pattern in SKIP_SUB_PATTERNS:
        if re.search(pattern, text):
            return None

    return None


def _build_rId_to_filename(z: zipfile.ZipFile) -> dict[str, str]:
    """Parse word/_rels/document.xml.rels to map rId → media/filename."""
    rels_xml = z.read("word/_rels/document.xml.rels")
    root = ET.fromstring(rels_xml)

    rmap = {}
    ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    for rel in root.findall(f"{{{ns}}}Relationship"):
        rId = rel.get("Id")
        target = rel.get("Target")
        if target and "image" in rel.get("Type", ""):
            filename = Path(target).name
            rmap[rId] = filename
    return rmap


def extract() -> dict[str, dict[str, str]]:
    """Walk docx paragraphs, map rId-embedded images → label.

    Returns dict of {filename: {label, section}}.
    """
    with zipfile.ZipFile(DOCX_PATH) as z:
        rId_to_file = _build_rId_to_filename(z)
        doc_xml = z.read("word/document.xml")
        root = ET.fromstring(doc_xml)

    ns_w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ns_a = "http://schemas.openxmlformats.org/drawingml/2006/main"
    ns_r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    paras = root.findall(f".//{{{ns_w}}}p")
    current_section: str = ""
    current_parent: str = ""
    golden: dict[str, dict[str, str]] = {}

    for para in paras:
        # Detect section heading
        pPr = para.find(f"{{{ns_w}}}pPr")
        style_val = None
        if pPr is not None:
            pStyle = pPr.find(f"{{{ns_w}}}pStyle")
            if pStyle is not None:
                style_val = pStyle.get(f"{{{ns_w}}}val")

        # Gather paragraph text
        texts = []
        for r in para.findall(f"{{{ns_w}}}r"):
            for t in r.findall(f"{{{ns_w}}}t"):
                if t.text:
                    texts.append(t.text)
        text = "".join(texts).strip()

        # Section heading: style matches "Prrafodelista" and has text
        if style_val == "Prrafodelista" and text:
            # Check if this is a known parent section
            if text in SECTION_LABEL_MAP:
                current_section = text
                current_parent = text
            else:
                # Sub-section: keep parent, overlay sub text
                current_section = text
            continue

        # Find image embed references
        blips = para.findall(f".//{{{ns_a}}}blip")
        embeds = [b.get(f"{{{ns_r}}}embed") for b in blips if b.get(f"{{{ns_r}}}embed")]

        if not embeds:
            continue

        # Resolve label
        label = _resolve_label(current_section)
        if label is None:
            section_parent = current_parent or current_section
            label = _resolve_label(section_parent)

        if label is None:
            continue

        for rId in embeds:
            filename = rId_to_file.get(rId)
            if filename and filename not in golden:
                golden[filename] = {
                    "label": label,
                    "section": current_section or current_parent,
                }

    return golden


def main():
    golden = extract()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    # Stats
    labels = {}
    for v in golden.values():
        labels[v["label"]] = labels.get(v["label"], 0) + 1

    print(f"Golden set: {len(golden)} images → {OUTPUT_PATH}")
    print("Per label:")
    for label, count in sorted(labels.items()):
        print(f"  {label}: {count}")
    print(f"\nSkipped: {59 - len(golden)} images (BODEGAS/ambiguous)")


if __name__ == "__main__":
    main()
