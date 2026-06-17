#!/usr/bin/env python3
"""Test the deployed VLM endpoint with a single building image.

Loads system prompt from specs/system_prompt.txt and includes
concise 10-class taxonomy definitions in the user message.

Usage:
    python scripts/test_endpoint_single.py
"""

import os
import sys
import time
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import requests as http_requests

load_dotenv()

api_url = os.getenv("API_URL")
api_key = os.getenv("API_KEY")
deployment_name = os.getenv("DEPLOYMENT_NAME")

if not all([api_url, api_key, deployment_name]):
    print("ERROR: Missing environment variables. Check your .env file:")
    print(f"  API_URL: {'SET' if api_url else 'MISSING'}")
    print(f"  API_KEY: {'SET' if api_key else 'MISSING'}")
    print(f"  DEPLOYMENT_NAME: {'SET' if deployment_name else 'MISSING'}")
    sys.exit(1)

client = OpenAI(
    base_url=f"{api_url}/v1",
    api_key=api_key,
    default_headers={"azureml-model-deployment": deployment_name},
)

MODEL_NAME = "Qwen/Qwen2.5-VL-32B-Instruct"

# Load system prompt
repo_root = Path(__file__).resolve().parents[1]
system_prompt_path = repo_root / "specs" / "system_prompt.txt"
with open(system_prompt_path) as f:
    SYSTEM_PROMPT = f.read().strip()

# Concise 10-class taxonomy (abridged from specs/taxonomy_operational.yaml)
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

# Output schema description
OUTPUT_SCHEMA = """
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
"""

# Test image: recognizable commercial/office building
test_image = (
    "https://raw.githubusercontent.com/"
    "EbookFoundation/free-programming-books/main/"
    "more/free-programming-cheatsheets.md"
)
# Use a reliable test image
test_image = "https://picsum.photos/id/0/800/600"


def download_and_encode(url: str) -> str:
    """Download image from URL and return base64 data URL."""
    headers = {"User-Agent": "Phase02-Test/1.0"}
    resp = http_requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    img_b64 = base64.b64encode(resp.content).decode()
    ct = resp.headers.get("content-type", "image/jpeg")
    return f"data:{ct};base64,{img_b64}"


# Download and encode image
print("Downloading test image...")
try:
    image_data_url = download_and_encode(test_image)
    print(f"  Image encoded: {len(image_data_url)} chars")
except Exception as e:
    print(f"  Download failed: {e}")
    print("  Using local fallback — create a test image or provide a URL")
    sys.exit(1)

print("=" * 60)
print("TEST: Single-Image Inference")
print("=" * 60)
print(f"Endpoint:    {api_url}")
print(f"Model:       {MODEL_NAME}")
print(f"System prompt: {len(SYSTEM_PROMPT)} chars from {system_prompt_path}")
print()

start = time.time()

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        TAXONOMY_DEFINITIONS
                        + "\n---\n"
                        + OUTPUT_SCHEMA
                        + "\n---\n"
                        + "Classify the building in the image below using the "
                        + "taxonomy and return the JSON."
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ],
    max_tokens=512,
)

latency = time.time() - start

print(f"Latency: {latency:.1f}s")
print(f"Tokens:  {response.usage}")
print()
print("Response:")
raw = response.choices[0].message.content.strip()

# Strip markdown code fences if present
if raw.startswith("```"):
    lines = raw.split("\n")
    lines = [l for l in lines if not l.startswith("```")]
    raw = "\n".join(lines)

print(raw[:600])
print()

# Try to parse JSON
try:
    parsed = json.loads(raw)
    print("JSON parsed successfully!")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    json_ok = True
except json.JSONDecodeError:
    print("JSON parse failed (expected without schema enforcement — Phase 03)")
    json_ok = False

# Record results
with open("experiments/phase02-results.txt", "a") as f:
    f.write(f"\nsingle_image_test:\n")
    f.write(f"  latency_seconds: {latency:.1f}\n")
    f.write(f"  tokens: {response.usage.total_tokens if response.usage else 0}\n")
    f.write(f"  json_parsed: {json_ok}\n")
    f.write(f"  timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ')}\n")

print(f"Results appended to experiments/phase02-results.txt")
