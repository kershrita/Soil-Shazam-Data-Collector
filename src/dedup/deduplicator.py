"""Image deduplication via Perceptual Hashing."""

from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from utils import collect_image_paths

logger = logging.getLogger(__name__)


def _find_duplicates_single_process(
    encodings: dict[str, str],
    threshold: int,
) -> dict[str, list[str]]:
    """Find duplicate filename groups without multiprocessing (Windows-safe)."""
    if not encodings:
        return {}

    names = list(encodings.keys())
    hash_ints: list[int | None] = []
    for name in names:
        try:
            hash_ints.append(int(encodings[name], 16))
        except (TypeError, ValueError):
            hash_ints.append(None)

    duplicates: dict[str, list[str]] = {}
    for i in tqdm(range(len(names)), desc="Finding duplicates (single-process)"):
        name_i = names[i]
        hi = hash_ints[i]
        dup_list: list[str] = []
        for j in range(i + 1, len(names)):
            hj = hash_ints[j]
            if hi is None or hj is None:
                continue
            distance = (hi ^ hj).bit_count()
            if distance <= threshold:
                dup_list.append(names[j])
        if dup_list:
            duplicates[name_i] = dup_list
    return duplicates


def _phash_dedup(
    image_dir: Path, threshold: int = 10, num_workers: int = 8,
) -> tuple[set[str], dict[str, list[str]]]:
    """Stage 1: Find duplicates via perceptual hashing (imagededup).

    Uses ThreadPoolExecutor for reliable parallel hashing on Windows
    (imagededup's built-in multiprocessing can deadlock).

    Returns (set of filenames to REMOVE, dict mapping kept→[removed]).
    """
    from imagededup.methods import PHash

    logger.info(f"Stage 1 — Perceptual hashing (threshold={threshold}, workers={num_workers})")
    hasher = PHash()
    image_files = collect_image_paths(image_dir)

    # Hash images in parallel using threads (avoids Windows multiprocessing deadlocks)
    encodings: dict[str, str] = {}
    errors = 0

    def _hash_one(path: Path) -> tuple[str, str | None]:
        try:
            h = hasher.encode_image(image_file=str(path))
            return path.name, h
        except Exception as e:
            logger.debug(f"Failed to hash {path.name}: {e}")
            return path.name, None

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(tqdm(
            pool.map(_hash_one, image_files),
            total=len(image_files),
            desc="Hashing images (PHash)",
        ))

    for name, h in results:
        if h is not None:
            encodings[name] = h
        else:
            errors += 1

    if errors:
        logger.warning(f"Skipped {errors} images that failed to hash")

    # Avoid imagededup multiprocessing retrieval on Windows-restricted environments.
    duplicates = _find_duplicates_single_process(encodings, threshold)

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
    num_workers: int = 8,
) -> dict:
    """Run perceptual hash deduplication.

    Copies surviving images to output_dir using parallel I/O.
    Returns stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = input_dir / "images" if (input_dir / "images").exists() else input_dir
    out_images_dir = output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(images_dir)
    stats = {"total": len(image_paths), "removed": 0, "kept": 0}

    if not image_paths:
        for stale in collect_image_paths(out_images_dir):
            stale.unlink(missing_ok=True)
        (output_dir / "dedup_groups.json").write_text("{}", encoding="utf-8")
        return stats

    phash_remove, phash_groups = _phash_dedup(images_dir, phash_threshold, num_workers)
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

    # Synchronize output directory with current surviving set
    surviving_names = {p.name for p in surviving}
    for existing in collect_image_paths(out_images_dir):
        if existing.name not in surviving_names:
            existing.unlink(missing_ok=True)

    # Copy surviving images — parallel I/O

    copy_workers = min(num_workers, 16)

    def _copy(p: Path) -> None:
        dest = out_images_dir / p.name
        shutil.copy2(p, dest)

    with ThreadPoolExecutor(max_workers=copy_workers) as pool:
        list(tqdm(pool.map(_copy, surviving), total=len(surviving), desc="Copying deduplicated images"))

    logger.info(
        f"Dedup done: {stats['kept']} kept, {stats['removed']} removed (PHash)"
    )
    return stats
