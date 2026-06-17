#!/usr/bin/env python3
"""
Evaluation harness: Run golden set through the classify endpoint and compute metrics.

Usage:
    python scripts/run_eval.py [--limit N] [--timeout S]

Requires the FastAPI server to be running on localhost:7860.
Results are written to experiments/phase05-results.json.
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
Evals_PATH = REPO / "evals"
TEST_IMAGES = REPO / "experiments" / "test_images"
API_URL = "http://localhost:7860"

VALID_LABELS = frozenset({
    "RESIDENCIAL_1", "COMERCIAL_1", "COMERCIAL_2", "COMERCIAL_3",
    "DOTACIONAL_1", "DOTACIONAL_2", "MOLES_1", "RURAL_1",
    "MIXTO_1", "MIXTO_2", "UNKNOWN_OR_INSUFFICIENT_EVIDENCE",
})


def load_golden_set(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def classify_building(building_id: str, image_file: str, timeout: int = 120) -> dict | None:
    """Send classify request to the running server using a file path."""
    img_path = REPO / "experiments" / image_file
    if not img_path.exists():
        print(f"  [SKIP] file not found: {img_path}")
        return None

    with open(img_path, "rb") as f:
        import base64
        b64 = base64.b64encode(f.read()).decode()
        ext = img_path.suffix.lower().replace(".", "")
        if ext == "jpg":
            ext = "jpeg"
        data_url = f"data:image/{ext};base64,{b64}"

    try:
        resp = requests.post(
            f"{API_URL}/classify",
            json={"building_id": building_id, "image_urls": [data_url]},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] {building_id}: {e}")
        return None


def get_primary_label(classification: dict) -> str:
    clases = classification.get("clases", [])
    for c in clases:
        if c.get("is_primary"):
            return c.get("label", "UNKNOWN")
    if clases:
        return clases[0].get("label", "UNKNOWN")
    return "UNKNOWN"


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute classification metrics from true and predicted label lists."""
    all_labels = sorted(set(y_true) | set(y_pred))

    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n > 0 else 0

    class_metrics = {}
    for label in all_labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp, "fp": fp, "fn": fn,
        }

    f1_values = [m["f1"] for m in class_metrics.values() if m["support"] > 0]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0

    # Weighted F1
    total = sum(m["support"] for m in class_metrics.values())
    weighted_f1 = sum(m["f1"] * m["support"] for m in class_metrics.values()) / total if total > 0 else 0

    # Confusion matrix as dict of dicts
    confusion = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "total_samples": n,
        "total_correct": correct,
        "per_class": class_metrics,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run golden set evaluation")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0=all)")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    golden_path = Evals_PATH / "golden_set.jsonl"
    items = load_golden_set(golden_path)
    if args.limit > 0:
        items = items[: args.limit]

    print(f"Evaluating {len(items)} golden set entries...\n")

    results = []
    y_true, y_pred = [], []
    labels = Counter()
    parse_failures = 0
    api_errors = 0
    total_latency = 0

    for i, item in enumerate(items, 1):
        bid = item["building_id"]
        label = item["label"]
        img = item["image_file"]
        labels[label] += 1

        print(f"[{i}/{len(items)}] {bid} ({label}) ... ", end="", flush=True)
        start = time.monotonic()

        pred = classify_building(bid, img, timeout=args.timeout)
        if pred is None:
            api_errors += 1
            y_pred.append("ERROR")
            y_true.append(label)
            results.append({"building_id": bid, "true_label": label, "predicted_label": "ERROR", "error": "api_error"})
            print("API ERROR")
            continue

        elapsed = time.monotonic() - start
        total_latency += elapsed

        classification = pred.get("classification", {})
        if isinstance(classification, dict) and "error" in classification:
            parse_failures += 1
            predicted = "PARSE_ERROR"
        else:
            predicted = pred.get("primary_label") or get_primary_label(classification)

        y_true.append(label)
        y_pred.append(predicted)

        status = "✓" if predicted == label else "✗"
        print(f"{status} {predicted} ({elapsed:.0f}s)")

        results.append({
            "building_id": bid,
            "true_label": label,
            "predicted_label": predicted,
            "confidence": pred.get("max_confidence"),
            "requires_review": pred.get("requires_review"),
            "latency_s": round(elapsed, 1),
        })

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}\n")

    metrics = compute_metrics(y_true, y_pred)
    metrics["parse_failures"] = parse_failures
    metrics["api_errors"] = api_errors
    metrics["avg_latency_s"] = round(total_latency / len(results), 1) if results else 0
    metrics["total_latency_s"] = round(total_latency, 0)

    print(f"Accuracy:   {metrics['accuracy']:.1%} ({metrics['total_correct']}/{metrics['total_samples']})")
    print(f"Macro F1:   {metrics['macro_f1']:.3f}")
    print(f"Weighted F1:{metrics['weighted_f1']:.3f}")
    print(f"Parse fails:{metrics['parse_failures']}")
    print(f"API errors: {metrics['api_errors']}")
    print(f"Avg latency:{metrics['avg_latency_s']}s")
    print()

    print("Per-class metrics:")
    print(f"{'Label':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Supp':>5}")
    print("-" * 58)
    for label in sorted(metrics["per_class"]):
        m = metrics["per_class"][label]
        if m["support"] > 0:
            print(f"{label:<30} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['support']:5}")
    print()

    # Confusion matrix
    print("Confusion Matrix (rows=true, cols=predicted):")
    cm_labels = sorted(set(y_true) | set(y_pred))
    header = f"{'':>30}" + "".join(f"{l[:6]:>8}" for l in cm_labels)
    print(header)
    for t in cm_labels:
        row = f"{t[:30]:>30}"
        for p in cm_labels:
            row += f"{metrics['confusion_matrix'].get(t, {}).get(p, 0):>8}"
        print(row)

    # Per-class errors
    print(f"\n{'='*50}")
    print("ERROR DETAIL (wrong predictions)")
    for r in results:
        if r["predicted_label"] != r["true_label"]:
            print(f"  {r['building_id']}: true={r['true_label']} pred={r['predicted_label']} conf={r.get('confidence', '?')}")

    # Save results
    output_path = Path(args.output) if args.output else (REPO / "experiments" / "phase05-results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt_version": "p003",
            "model": "Qwen2.5-VL-32B-Instruct",
            "metrics": metrics,
            "predictions": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
