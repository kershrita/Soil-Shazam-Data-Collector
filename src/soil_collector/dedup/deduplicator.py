"""Image deduplication via Perceptual Hashing."""

from __future__ import annotations

import json
import logging
import shutil
import warnings
from pathlib import Path

from tqdm import tqdm

from soil_collector.utils.image_utils import collect_image_paths

logger = logging.getLogger(__name__)


def _phash_dedup(image_dir: Path, threshold: int = 10) -> tuple[set[str], dict[str, list[str]]]:
    """Stage 1: Find duplicates via perceptual hashing (imagededup).

    Returns (set of filenames to REMOVE, dict mapping kept→[removed]).
    """
    from imagededup.methods import PHash

    logger.info(f"Stage 1 — Perceptual hashing (threshold={threshold})")
    hasher = PHash()

    # imagededup expects a directory path
    encodings = hasher.encode_images(image_dir=str(image_dir))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        duplicates = hasher.find_duplicates(
            encoding_map=encodings,
            max_distance_threshold=threshold,
        )

    # Determine which files to remove — keep the first in each group
    to_remove: set[str] = set()
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()

    for filename, dups in duplicates.items():
        if filename in to_remove:
            continue
        if filename in seen:
            continue
        seen.add(filename)
        removed_in_group: list[str] = []
        for dup in dups:
            if dup not in seen:
                to_remove.add(dup)
                seen.add(dup)
                removed_in_group.append(dup)
        if removed_in_group:
            groups[filename] = removed_in_group

    logger.info(f"Stage 1: {len(to_remove)} duplicates found via PHash")
    return to_remove, groups


def run_deduplication(
    input_dir: Path,
    output_dir: Path,
    phash_threshold: int = 10,
) -> dict:
    """Run perceptual hash deduplication.

    Copies surviving images to output_dir.
    Returns stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = input_dir / "images" if (input_dir / "images").exists() else input_dir

    image_paths = collect_image_paths(images_dir)
    stats = {"total": len(image_paths), "removed": 0, "kept": 0}

    if not image_paths:
        return stats

    phash_remove, phash_groups = _phash_dedup(images_dir, phash_threshold)
    stats["removed"] = len(phash_remove)

    surviving = [p for p in image_paths if p.name not in phash_remove]
    stats["kept"] = len(surviving)

    # Save group mappings
    groups_path = output_dir / "dedup_groups.json"
    groups_path.write_text(
        json.dumps(phash_groups, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Saved duplicate group mapping to {groups_path}")

    # Copy surviving images
    out_images_dir = output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(surviving, desc="Copying deduplicated images"):
        dest = out_images_dir / p.name
        if not dest.exists():
            shutil.copy2(p, dest)

    logger.info(
        f"Dedup done: {stats['kept']} kept, {stats['removed']} removed (PHash)"
    )
    return stats
