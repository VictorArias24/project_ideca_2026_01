#!/usr/bin/env python3
"""
Prompt experiment harness: measure parse success rate on real building images.

Uses the 4 Bogota building images from azureml-demo/images/.
Runs classification with prompt p001, measures:
  - Parse success rate (target: >= 80%)
  - Latency per building
  - Token usage and cost estimate

Only creates prompt p002 if p001 falls below 80% parse success.

Usage:
    python scripts/run_prompt_experiments.py

Requires:
    - Endpoint running (scale to 100% traffic first)
    - Building images in experiments/test_images/ (copied from azureml-demo/images/)
"""

import sys
import json
import time
import copy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters.foundry_adapter import FoundryVlmAdapter
from src.parsers.output_parser import parse_and_validate
from src.parsers.repair_policy import RepairPolicy, basic_json_repair
from src.prompts.classification_p001 import build_messages, get_system_prompt
from src.utils.image_encoding import image_to_data_url

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
TEST_IMAGES_DIR = EXPERIMENTS_DIR / "test_images"

# Test buildings: each is a set of facade views for one building.
# Uses images from azureml-demo/images/ (pre-downloaded Bogota building photos).
# Each "building" here uses all 4 images as views of the same building.
TEST_BUILDINGS = [
    {
        "building_id": "bogota-001",
        "images": [
            "102808000096_679.jpg",
            "209106005025_1108.jpg",
            "209106005025_1109.jpg",
            "209106005025_1110.jpg",
        ],
    },
]


def run_building(adapter: FoundryVlmAdapter, building: dict, prompt_label: str) -> dict:
    """Run classification for one building and return result dict."""
    building_id = building["building_id"]

    # Encode images
    data_urls = []
    for img_name in building["images"]:
        img_path = TEST_IMAGES_DIR / img_name
        if not img_path.exists():
            return {
                "building_id": building_id,
                "prompt_label": prompt_label,
                "parse_success": False,
                "error": f"Image not found: {img_name}",
                "latency_s": 0,
                "result": None,
                "raw_text": "",
                "attempts": 0,
            }
        data_urls.append(image_to_data_url(img_path))

    # Build messages
    messages = build_messages(data_urls, building_id)

    # VLM call wrapper
    def vlm_call(msgs):
        return adapter.classify_rawmsgs)["text"]

    repair = RepairPolicy(vlm_call=vlm_call)

    start = time.time()
    result, error, attempts = repair.classify_with_repair(messages)
    elapsed = time.time() - start

    # Get raw text for diagnostics
    raw_text = ""
    try:
        resp = adapter.classify(messages)
        raw_text = resp["text"][:300]
    except Exception:
        pass

    return {
        "building_id": building_id,
        "prompt_label": prompt_label,
        "parse_success": result is not None,
        "error": error,
        "latency_s": elapsed,
        "attempts": attempts,
        "result": result,
        "raw_text": raw_text,
    }


def main():
    # Verify images exist
    if not TEST_IMAGES_DIR.exists():
        print("ERROR: Test images directory not found.")
        print(f"  Expected: {TEST_IMAGES_DIR}")
        print("  Copy images from azureml-demo/images/ to experiments/test_images/")
        print(f"  Run: cp -r /home/viarias/workspace/azureml-demo/images/ {TEST_IMAGES_DIR}")
        sys.exit(1)

    for building in TEST_BUILDINGS:
        for img_name in building["images"]:
            if not (TEST_IMAGES_DIR / img_name).exists():
                print(f"ERROR: Missing image: {img_name}")
                print(f"  Copy from azureml-demo/images/ to {TEST_IMAGES_DIR}")
                sys.exit(1)

    print("=" * 60)
    print("PROMPT EXPERIMENT: p001")
    print("=" * 60)
    print(f"Buildings:  {len(TEST_BUILDINGS)}")
    print(f"Total images: {sum(len(b['images']) for b in TEST_BUILDINGS)}")
    print(f"Prompt:     prompts/classification_p001.py")
    print()

    adapter = FoundryVlmAdapter()

    if not adapter.health_check():
        print("ERROR: Endpoint not reachable. Scale up the endpoint first.")
        sys.exit(1)

    all_results = {}
    for prompt_label in ["p001"]:
        print(f"Running prompt {prompt_label}...")
        results = []

        for building in TEST_BUILDINGS:
            if len(TEST_BUILDINGS) > 1:
                print(f"  {building['building_id']}...", end=" ")
            r = run_building(adapter, building, prompt_label)
            results.append(r)

            status = "PASS" if r["parse_success"] else f"FAIL ({r['error']})"
            print(f"{status} | {r['latency_s']:.1f}s | {r['attempts']} attempt(s)")

        all_results[prompt_label] = results

        # Summary
        success = sum(1 for r in results if r["parse_success"])
        total = len(results)
        rate = success / total * 100 if total > 0 else 0
        avg_latency = sum(r["latency_s"] for r in results) / total if total > 0 else 0

        print(f"\n  Summary: {success}/{total} parse success ({rate:.0f}%)")
        print(f"  Avg latency: {avg_latency:.1f}s")
        print()

    # Save results
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    output_path = EXPERIMENTS_DIR / "prompt_experiment_results.json"

    # Clean results for JSON serialization
    def clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(v) for v in obj]
        elif isinstance(obj, frozenset):
            return list(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(clean(all_results), f, indent=2, default=str, ensure_ascii=False)

    print(f"Results saved to {output_path}")

    # Decision: does p001 meet the 80% threshold?
    p001_results = all_results["p001"]
    p001_success = sum(1 for r in p001_results if r["parse_success"])
    p001_rate = p001_success / len(p001_results) * 100

    if p001_rate >= 80:
        print(f"\nPASS: p001 meets 80% parse success threshold ({p001_rate:.0f}%).")
        print("No p002 variant needed. Proceed to Phase 04.")
    else:
        print(f"\nBELOW THRESHOLD: p001 parse success = {p001_rate:.0f}% (< 80%).")
        print("Create prompts/classification_p002.py with adjustments:")
        print("  1. JSON-first instruction (output format before taxonomy)")
        print("  2. Stronger constraint: 'You MUST return ONLY valid JSON'")
        print("  3. Include example JSON output")
        print("  4. Simplify taxonomy to label + one-line description")
        print("Then re-run this script to compare p002 against p001.")


if __name__ == "__main__":
    main()
