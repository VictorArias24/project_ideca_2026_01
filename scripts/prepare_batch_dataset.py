"""
Prepare a 100-building batch dataset from the azureml-demo source.

Filters buildings with 2+ images and performs stratified sampling
by image count. Copies building directories to data/packages/ and
generates _task.json files.

Usage:
    python scripts/prepare_batch_dataset.py
"""

import json
import random
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

random.seed(42)

ORIGEN = Path("/home/viarias/workspace/azureml-demo/batch_scoring/test_data")
DESTINO = Path("data/packages")
N_BUILDINGS = 100
MIN_IMAGES = 2


def preparar() -> None:
    if not ORIGEN.exists():
        raise FileNotFoundError(f"Source not found: {ORIGEN}")

    DESTINO.mkdir(parents=True, exist_ok=True)

    # Group buildings by image count
    por_cantidad = defaultdict(list)
    for carpeta in sorted(ORIGEN.iterdir()):
        if carpeta.is_dir():
            n = len(list(carpeta.glob("*.jpg")))
            if n >= MIN_IMAGES:
                por_cantidad[n].append(carpeta)

    total_multi = sum(len(v) for v in por_cantidad.values())
    print(f"Total buildings with {MIN_IMAGES}+ images: {total_multi}")

    # Stratified proportional sampling, minimum 1 per stratum
    seleccionados = []
    for n, edificios in sorted(por_cantidad.items()):
        proporcion = len(edificios) / total_multi
        n_seleccionar = max(1, min(len(edificios), round(N_BUILDINGS * proporcion)))
        muestra = random.sample(edificios, n_seleccionar)
        seleccionados.extend(muestra)

    # Adjust to exactly N_BUILDINGS
    if len(seleccionados) > N_BUILDINGS:
        seleccionados = seleccionados[:N_BUILDINGS]
    elif len(seleccionados) < N_BUILDINGS:
        others = [d for d in sum(por_cantidad.values(), []) if d not in set(seleccionados)]
        needed = N_BUILDINGS - len(seleccionados)
        seleccionados.extend(random.sample(others, min(needed, len(others))))

    random.shuffle(seleccionados)

    # Copy and generate _task.json
    total_imagenes = 0
    dist = Counter()

    for carpeta in seleccionados:
        dest_dir = DESTINO / carpeta.name
        shutil.copytree(carpeta, dest_dir, dirs_exist_ok=True)

        imagenes = sorted(dest_dir.glob("*.jpg"))
        n_img = len(imagenes)
        total_imagenes += n_img
        dist[n_img] += 1

        tarea = {
            "building_id": carpeta.name,
            "num_images": n_img,
            "image_files": [img.name for img in imagenes],
        }
        with open(dest_dir / "_task.json", "w") as f:
            json.dump(tarea, f)

    print(f"\nDataset ready in {DESTINO.resolve()}")
    print(f"\nSummary:")
    print(f"  Buildings: {len(seleccionados)}")
    print(f"  Total images: {total_imagenes}")
    print(f"  Avg images/building: {total_imagenes / len(seleccionados):.1f}")
    print(f"  Distribution:")
    for k in sorted(dist):
        print(f"    {k} image(s): {dist[k]} buildings")

    # Save manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "source": str(ORIGEN),
        "total_buildings": len(seleccionados),
        "total_images": total_imagenes,
        "avg_images_per_building": total_imagenes / len(seleccionados),
        "distribution": {str(k): v for k, v in sorted(dist.items())},
        "buildings": sorted([c.name for c in seleccionados]),
    }
    with open(DESTINO / "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest: {DESTINO / '_manifest.json'}")


if __name__ == "__main__":
    preparar()
