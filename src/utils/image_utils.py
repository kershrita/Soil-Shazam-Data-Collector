"""Image loading, validation, resizing, and saving utilities."""

from __future__ import annotations

import hashlib
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
        # Palette images with transparency must go through RGBA to avoid PIL warning
        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")
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
    if mode not in {"shortest_side", "longest_side"}:
        logger.warning(f"Unknown resize mode '{mode}', falling back to shortest_side")
        mode = "shortest_side"

    shortest = min(w, h)
    longest = max(w, h)
    metric = shortest if mode == "shortest_side" else longest
    if metric <= max_longest_side:
        return img

    scale_base = shortest if mode == "shortest_side" else longest
    scale = max_longest_side / scale_base
    new_w = round(w * scale)
    new_h = round(h * scale)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


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


def canonical_image_name(path: Path, root_dir: Path) -> str:
    """Return a stable filename suitable for flat output directories.

    If `path` is already directly under `root_dir`, the original filename is kept.
    For nested inputs (for example raw download folders), prepend a short hash of
    the relative parent folder to avoid collisions like many `Image_1.jpg` files.
    """
    try:
        rel = path.relative_to(root_dir)
    except ValueError:
        rel = Path(path.name)

    if str(rel.parent) in {"", "."}:
        return path.name

    parent_token = str(rel.parent).replace("\\", "/")
    prefix = hashlib.md5(parent_token.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{path.name}"
