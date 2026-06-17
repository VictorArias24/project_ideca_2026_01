"""
Shared image encoding: file path / URL → base64 data URL.

Extracted from scripts/classify_building.py and scripts/run_prompt_experiments.py.
Used by both the CLI scripts and the FastAPI ClassifyService.
"""

import base64
import io
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

logger = logging.getLogger(__name__)

MAX_DIM = 400
NO_RESIZE = 0
QUALITY = 75


def _resolve_max_dim(explicit: int | None = None) -> int:
    """Resolve max_dim: explicit arg > IMAGE_MAX_DIM env > MAX_DIM default (400).

    Set IMAGE_MAX_DIM=0 (or NO_RESIZE) to keep native resolution.
    """
    if explicit is not None:
        return explicit
    env_val = os.getenv("IMAGE_MAX_DIM")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            pass
    return MAX_DIM


def _ensure_rgb_for_jpeg(img: Image.Image, fmt: str) -> Image.Image:
    """Convert image to RGB if saving as JPEG and mode has alpha channel."""
    if fmt.upper() in ("JPEG", "JPG") and img.mode in ("RGBA", "P", "LA", "PA"):
        return img.convert("RGB")
    return img


def _resize_if_needed(img: Image.Image, max_dim: int) -> Image.Image:
    """Resize image so its longest side <= max_dim. Pass max_dim=0 to skip."""
    if max_dim <= 0:
        return img
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    ratio = max_dim / max(w, h)
    return img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)


def image_to_data_url(source: str | Path, max_dim: int | None = None) -> str:
    """Load an image from disk or URL, optionally resize, return a base64 data URL.

    If *source* starts with ``data:`` it is returned as-is (and may be re-encoded).
    If *source* looks like a URL (http/https), it is fetched via requests.
    Otherwise it is treated as a local file path.

    Args:
        source: File path, http(s) URL, or data: URI.
        max_dim: Maximum width/height before encoding. ``None`` (default) reads
            the ``IMAGE_MAX_DIM`` env var, falling back to 400 px. Pass ``0`` to
            keep native resolution (no resize).

    Returns:
        Base64 data URL string, e.g. ``data:image/jpeg;base64,...``
    """
    resolved = _resolve_max_dim(max_dim)
    source_str = str(source)

    if source_str.startswith("data:"):
        return _data_url_resize(source_str, resolved)

    if source_str.startswith(("http://", "https://")):
        return _url_to_data_url(source_str, resolved)

    return _file_to_data_url(Path(source_str), resolved)


def _file_to_data_url(path: Path, max_dim: int) -> str:
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    img = _resize_if_needed(img, max_dim)

    buf = io.BytesIO()
    fmt = img.format or "JPEG"
    img = _ensure_rgb_for_jpeg(img, fmt)
    img.save(buf, format=fmt, quality=QUALITY)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{img_b64}"


def _url_to_data_url(url: str, max_dim: int) -> str:
    import requests

    resp = requests.get(url, timeout=30, stream=True)
    resp.raise_for_status()

    img = Image.open(io.BytesIO(resp.content))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    img = _resize_if_needed(img, max_dim)

    buf = io.BytesIO()
    fmt = img.format or "JPEG"
    img = _ensure_rgb_for_jpeg(img, fmt)
    img.save(buf, format=fmt, quality=QUALITY)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{img_b64}"


def _data_url_resize(data_url: str, max_dim: int) -> str:
    header, encoded = data_url.split(",", 1)
    img_data = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_data))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    img = _resize_if_needed(img, max_dim)

    buf = io.BytesIO()
    fmt = img.format or "JPEG"
    img = _ensure_rgb_for_jpeg(img, fmt)
    img.save(buf, format=fmt, quality=QUALITY)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{img_b64}"
