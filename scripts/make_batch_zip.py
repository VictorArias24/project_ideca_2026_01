"""
Create a ZIP file from data/packages/ for batch processing upload.

Usage:
    python scripts/make_batch_zip.py

Output:
    data/packages.zip — ready for upload to /batch page
"""

import zipfile
from pathlib import Path

SOURCE = Path("data/packages")
OUTPUT = Path("data/packages.zip")


def main():
    if not SOURCE.exists():
        print(f"ERROR: {SOURCE} not found. Run scripts/prepare_batch_dataset.py first.")
        return

    buildings = [d for d in SOURCE.iterdir() if d.is_dir() and (d / "_task.json").exists()]
    if not buildings:
        print("ERROR: No building directories with _task.json found.")
        return

    OUTPUT.unlink(missing_ok=True)

    total_files = 0
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for bdir in sorted(buildings):
            for f in sorted(bdir.iterdir()):
                if f.name == "_manifest.json":
                    continue
                arcname = f"{bdir.name}/{f.name}"
                zf.write(f, arcname)
                total_files += 1

    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"Created: {OUTPUT}")
    print(f"  Buildings: {len(buildings)}")
    print(f"  Files: {total_files}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Upload to /batch page in the web UI")


if __name__ == "__main__":
    main()
