"""
Shared image encoding: file path / URL → base64 data URL.

Extracted from scripts/classify_building.py and scripts/run_prompt_experiments.py.
Used by both the CLI scripts and the FastAPI ClassifyService.
"""

import base64
import io
import logging
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

logger = logging.getLogger(__name__)

MAX_DIM = 400
QUALITY = 75


def _ensure_rgb_for_jpeg(img: Image.Image, fmt: str) -> Image.Image:
    """Convert image to RGB if saving as JPEG and mode has alpha channel."""
    if fmt.upper() in ("JPEG", "JPG") and img.mode in ("RGBA", "P", "LA", "PA"):
        return img.convert("RGB")
    return img


def image_to_data_url(source: str | Path, max_dim: int = MAX_DIM) -> str:
    """Load an image from disk or URL, resize, return a base64 data URL.

    If *source* starts with ``data:`` it is returned as-is.
    If *source* looks like a URL (http/https), it is fetched via requests.
    Otherwise it is treated as a local file path.

    Args:
        source: File path, http(s) URL, or data: URI.
        max_dim: Maximum width/height before encoding (default 400 px).

    Returns:
        Base64 data URL string, e.g. ``data:image/jpeg;base64,...``
    """
    source_str = str(source)

    if source_str.startswith("data:"):
        return _data_url_resize(source_str, max_dim)

    if source_str.startswith(("http://", "https://")):
        return _url_to_data_url(source_str, max_dim)

    return _file_to_data_url(Path(source_str), max_dim)


def _file_to_data_url(path: Path, max_dim: int) -> str:
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

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

    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

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

    w, h = img.size
    if max(w, h) <= max_dim:
        return data_url

    ratio = max_dim / max(w, h)
    img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    fmt = img.format or "JPEG"
    img = _ensure_rgb_for_jpeg(img, fmt)
    img.save(buf, format=fmt, quality=QUALITY)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{img_b64}"
