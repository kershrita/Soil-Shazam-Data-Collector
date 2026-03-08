"""Resolution filter — discard too-small images, resize too-large ones."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from tqdm import tqdm

from soil_collector.utils.image_utils import (
    check_resolution,
    collect_image_paths,
    load_image,
    resize_image,
    save_as_jpeg,
)

logger = logging.getLogger(__name__)


def _unique_name(path: Path, input_dir: Path) -> str:
    """Generate a unique output name using relative path hash to avoid collisions.

    Images from different source/query folders may share the same filename
    (e.g., Image_1.jpg). We include a short hash of the relative path to
    make each output name unique.
    """
    try:
        rel = path.relative_to(input_dir)
    except ValueError:
        rel = path
    path_hash = hashlib.md5(str(rel.parent).encode()).hexdigest()[:8]
    return f"{path_hash}_{path.stem}"


def run_resolution_filter(
    input_dir: Path,
    output_dir: Path,
    min_shortest_side: int = 512,
    max_longest_side: int = 1024,
    jpeg_quality: int = 95,
) -> dict:
    """Filter images by resolution and resize oversized ones.

    - Discards images with shortest side < min_shortest_side.
    - Resizes images with shortest side > max_longest_side (shortest side → target).
    - Saves all passing images as JPEG.

    Returns stats dict with counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_image_paths(input_dir)

    stats = {"total": len(image_paths), "kept": 0, "discarded": 0, "resized": 0, "errors": 0}
    logger.info(f"Resolution filter: processing {stats['total']} images")

    # Collect existing output files to support resume
    existing = {p.stem for p in output_dir.rglob("*.jpg")}

    for path in tqdm(image_paths, desc="Resolution filter"):
        out_name = _unique_name(path, input_dir)
        if out_name in existing:
            stats["kept"] += 1
            continue

        img = load_image(path)
        if img is None:
            stats["errors"] += 1
            continue

        # Check minimum resolution
        if not check_resolution(img, min_shortest_side):
            stats["discarded"] += 1
            continue

        # Resize if oversized
        original_size = img.size
        img = resize_image(img, max_longest_side)
        if img.size != original_size:
            stats["resized"] += 1

        # Save as JPEG
        out_path = output_dir / f"{out_name}.jpg"
        save_as_jpeg(img, out_path, jpeg_quality)
        stats["kept"] += 1

    logger.info(
        f"Resolution filter done: {stats['kept']} kept, "
        f"{stats['discarded']} discarded (<{min_shortest_side}px), "
        f"{stats['resized']} resized, {stats['errors']} errors"
    )
    return stats
