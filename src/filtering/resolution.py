"""Resolution filter — discard too-small images, resize too-large ones."""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from utils import (
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


def _process_one(
    path: Path,
    input_dir: Path,
    output_dir: Path,
    existing: set[str],
    min_shortest_side: int,
    max_longest_side: int,
    resize_mode: str,
    jpeg_quality: int,
) -> str:
    """Process a single image. Returns a status string: kept, skipped, discarded, resized, error."""
    out_name = _unique_name(path, input_dir)
    if out_name in existing:
        return "skipped"

    img = load_image(path)
    if img is None:
        return "error"

    if not check_resolution(img, min_shortest_side):
        return "discarded"

    original_size = img.size
    img = resize_image(img, max_longest_side, mode=resize_mode)
    was_resized = img.size != original_size

    out_path = output_dir / f"{out_name}.jpg"
    save_as_jpeg(img, out_path, jpeg_quality)
    return "resized" if was_resized else "kept"


def run_resolution_filter(
    input_dir: Path,
    output_dir: Path,
    min_shortest_side: int = 512,
    max_longest_side: int = 1024,
    resize_mode: str = "shortest_side",
    jpeg_quality: int = 95,
    workers: int = 8,
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
    logger.info(f"Resolution filter: processing {stats['total']} images ({workers} workers)")

    # Collect existing output files to support resume
    existing = {p.stem for p in output_dir.rglob("*.jpg")}

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="resize") as pool:
        futures = {
            pool.submit(
                _process_one,
                path,
                input_dir,
                output_dir,
                existing,
                min_shortest_side,
                max_longest_side,
                resize_mode,
                jpeg_quality,
            ): path
            for path in image_paths
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Resolution filter"):
            path = futures[future]
            try:
                result = future.result()
                if result == "skipped":
                    stats["kept"] += 1
                elif result == "kept":
                    stats["kept"] += 1
                elif result == "resized":
                    stats["resized"] += 1
                    stats["kept"] += 1
                elif result == "discarded":
                    stats["discarded"] += 1
                elif result == "error":
                    stats["errors"] += 1
            except Exception as e:
                logger.error(f"Failed processing {path}: {e}")
                stats["errors"] += 1

    logger.info(
        f"Resolution filter done: {stats['kept']} kept, "
        f"{stats['discarded']} discarded (<{min_shortest_side}px), "
        f"{stats['resized']} resized, {stats['errors']} errors"
    )
    return stats
