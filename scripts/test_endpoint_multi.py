#!/usr/bin/env python3
"""Test the endpoint with multiple images simulating a building with 4 views.

Loads system prompt from specs/system_prompt.txt and includes
concise 10-class taxonomy definitions in the user message.

Usage:
    python scripts/test_endpoint_multi.py
"""

import os
import sys
import time
import json
import base64
import io
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import requests as http_requests
from PIL import Image

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

# 4 building images simulating multi-view input
# NOTE: Using non-building images for infrastructure verification.
# Real building images will be used in Phase 05 evaluation.
building_image_urls = [
    "https://picsum.photos/id/0/800/600",
    "https://picsum.photos/id/10/800/600",
    "https://picsum.photos/id/20/800/600",
    "https://picsum.photos/id/30/800/600",
]


def download_and_encode(url: str, max_dim: int = 400) -> str:
    """Download image, resize to max_dim, return base64 data URL."""
    headers = {"User-Agent": "Phase02-Test/1.0"}
    resp = http_requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    fmt = img.format or "JPEG"
    img.save(buf, format=fmt, quality=75)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{img_b64}"


# Download and encode images
print("Downloading test images...")
encoded_images = []
for i, url in enumerate(building_image_urls):
    try:
        data_url = download_and_encode(url)
        encoded_images.append(data_url)
        print(f"  [{i+1}/{len(building_image_urls)}] {len(data_url)} chars")
    except Exception as e:
        print(f"  [{i+1}/{len(building_image_urls)}] FAIL: {e}")

if not encoded_images:
    print("No images downloaded. Aborting.")
    sys.exit(1)

print()
print("=" * 60)
print("TEST: Multi-Image Inference")
print("=" * 60)
print(f"Endpoint:      {api_url}")
print(f"Model:         {MODEL_NAME}")
print(f"Images:        {len(encoded_images)}")
print(f"System prompt: {len(SYSTEM_PROMPT)} chars from {system_prompt_path}")
print()

# Build user message: taxonomy + schema + images
content = [
    {
        "type": "text",
        "text": (
            TAXONOMY_DEFINITIONS
            + "\n---\n"
            + OUTPUT_SCHEMA
            + "\n---\n"
            + "The following images show buildings. Classify the PRIMARY "
            + "building using the taxonomy above and return the JSON."
        ),
    }
]

for data_url in encoded_images:
    content.append({"type": "image_url", "image_url": {"url": data_url}})

print("Sending request...")
start = time.time()

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ],
    max_tokens=256,
)

latency = time.time() - start

print(f"\nLatency: {latency:.1f}s")
print(f"Tokens:  {response.usage}")
print(f"\nResponse:")
raw = response.choices[0].message.content.strip()

# Strip markdown code fences if present
if raw.startswith("```"):
    lines = raw.split("\n")
    lines = [l for l in lines if not l.startswith("```")]
    raw = "\n".join(lines)

print(raw[:700])

# Try to parse JSON
try:
    parsed = json.loads(raw)
    print("\nJSON parsed successfully!")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    json_ok = True
    # Check label validity
    labels_in_response = set()
    for c in parsed.get("clases", []):
        labels_in_response.add(c.get("label", ""))
    valid_labels = {
        "RESIDENCIAL_1", "COMERCIAL_1", "COMERCIAL_2", "COMERCIAL_3",
        "DOTACIONAL_1", "DOTACIONAL_2", "MOLES_1", "RURAL_1",
        "MIXTO_1", "MIXTO_2", "UNKNOWN_OR_INSUFFICIENT_EVIDENCE",
    }
    invalid = labels_in_response - valid_labels
    if invalid:
        print(f"WARNING: Labels not in taxonomy: {invalid}")
    else:
        print(f"All labels valid: {labels_in_response}")
except json.JSONDecodeError:
    print("\nJSON parse failed (expected without schema enforcement — Phase 03)")
    json_ok = False

# Record results
with open("experiments/phase02-results.txt", "a") as f:
    f.write(f"\nmulti_image_test:\n")
    f.write(f"  image_count: {len(encoded_images)}\n")
    f.write(f"  latency_seconds: {latency:.1f}\n")
    f.write(f"  latency_per_image: {latency/len(encoded_images):.1f}\n")
    f.write(f"  tokens: {response.usage.total_tokens if response.usage else 0}\n")
    f.write(f"  json_parsed: {json_ok}\n")
    f.write(f"  timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
    a100_cost_per_hour = 3.60
    cost_per_call = (latency / 3600) * a100_cost_per_hour
    cost_per_1000 = cost_per_call * 1000
    f.write(f"  cost_per_call_usd: {cost_per_call:.4f}\n")
    f.write(f"  cost_per_1000_buildings_usd: {cost_per_1000:.2f}\n")

print(f"\nResults appended to experiments/phase02-results.txt")
