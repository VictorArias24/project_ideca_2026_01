#!/usr/bin/env python3
"""
End-to-end classification: images -> prompt -> VLM call -> parse -> validate.

Ties together the Phase 03 modules:
  - prompts.classification_p001  (system prompt + taxonomy + schema)
  - src.adapters.foundry_adapter (VLM client)
  - src.parsers.output_parser    (JSON extraction + validation)
  - src.parsers.repair_policy    (retry on parse failure)

Usage:
    python scripts/classify_building.py --building-id b001 \\
        --images /path/to/front.jpg /path/to/side.jpg
"""

import sys
import os
import json
import time
import base64
import io
import argparse
from pathlib import Path
from typing import Optional

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters.foundry_adapter import FoundryVlmAdapter
from src.parsers.output_parser import parse_and_validate
from src.parsers.repair_policy import RepairPolicy
from src.prompts.classification_p001 import build_messages
from src.utils.image_encoding import image_to_data_url


def classify_building(
    adapter: FoundryVlmAdapter,
    image_paths: list[str],
    building_id: str = "unknown",
) -> dict:
    """Run the full classification pipeline for one building.

    Returns dict with keys: result, error, attempts, latency_s, building_id.
    """
    # Encode images
    image_data_urls = [image_to_data_url(p) for p in image_paths]

    # Build messages (system prompt + taxonomy + schema + images)
    messages = build_messages(image_data_urls, building_id)

    # Build repair policy
    def vlm_call(msgs):
        return adapter.classify_rawmsgs)["text"]

    repair = RepairPolicy(vlm_call=vlm_call)

    # Classify with repair
    start = time.time()
    result, error, attempts = repair.classify_with_repair(messages)
    elapsed = time.time() - start

    return {
        "result": result,
        "error": error,
        "attempts": attempts,
        "latency_s": elapsed,
        "building_id": building_id,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Classify a building's use type from facade images."
    )
    parser.add_argument(
        "--building-id", default="unknown", help="Building identifier"
    )
    parser.add_argument(
        "--images", nargs="+", required=True, help="Paths to facade images (1-4)"
    )
    parser.add_argument(
        "--max-dim", type=int, default=400,
        help="Max image dimension before encoding (default: 400)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output result as JSON only"
    )
    args = parser.parse_args()

    # Validate image paths
    for path in args.images:
        if not os.path.isfile(path):
            print(f"ERROR: Image not found: {path}", file=sys.stderr)
            sys.exit(1)

    if len(args.images) < 1 or len(args.images) > 8:
        print("ERROR: 1-8 images required", file=sys.stderr)
        sys.exit(1)

    # Init adapter
    adapter = FoundryVlmAdapter()

    # Encode images
    if not args.json:
        print(f"Encoding {len(args.images)} image(s)...")
    data_urls = [image_to_data_url(p, args.max_dim) for p in args.images]

    # Build messages
    messages = build_messages(data_urls, args.building_id)

    # VLM call wrapper
    def vlm_call(msgs):
        return adapter.classify_rawmsgs)["text"]

    repair = RepairPolicy(vlm_call=vlm_call)

    # Run
    if not args.json:
        print("Classifying...")
    start = time.time()
    result, error, attempts = repair.classify_with_repair(messages)
    elapsed = time.time() - start

    if args.json:
        output = {
            "building_id": args.building_id,
            "result": result,
            "error": error,
            "attempts": attempts,
            "latency_s": elapsed,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"\nLatency:  {elapsed:.1f}s")
        print(f"Attempts: {attempts}")
        if result:
            print(f"\nClassification:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\nFAILED: {error}")
            sys.exit(1)


if __name__ == "__main__":
    main()
