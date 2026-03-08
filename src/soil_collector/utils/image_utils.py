"""Image loading, validation, resizing, and saving utilities."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}


def is_image_file(path: Path) -> bool:
    """Check if a file has a supported image extension."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def load_image(path: Path) -> Image.Image | None:
    """Load an image from disk. Returns None if corrupt or unreadable."""
    try:
        img = Image.open(path)
        img.load()  # Force full load to catch truncated files
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        logger.debug(f"Failed to load image {path}: {e}")
        return None


def check_resolution(img: Image.Image, min_shortest_side: int = 512) -> bool:
    """Return True if the shortest side of the image is >= min_shortest_side."""
    return min(img.size) >= min_shortest_side


def resize_image(
    img: Image.Image, max_longest_side: int = 1024, mode: str = "shortest_side"
) -> Image.Image:
    """Resize image if it exceeds max dimensions.

    mode='shortest_side': resize so shortest side = max_longest_side, keep aspect ratio.
    Only resizes if the image is larger than the target.
    """
    w, h = img.size
    shortest = min(w, h)

    if shortest <= max_longest_side:
        return img

    scale = max_longest_side / shortest
    new_w = round(w * scale)
    new_h = round(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def save_as_jpeg(img: Image.Image, path: Path, quality: int = 95) -> None:
    """Save an image as JPEG with the specified quality."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure RGB mode (no alpha channel)
    if img.mode != "RGB":
        img = img.convert("RGB")
    path = path.with_suffix(".jpg")
    img.save(path, "JPEG", quality=quality)


def collect_image_paths(directory: Path) -> list[Path]:
    """Recursively collect all image file paths from a directory."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file() and is_image_file(p))
