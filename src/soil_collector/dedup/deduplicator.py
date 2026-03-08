"""Two-stage image deduplication: Perceptual Hashing → CLIP embeddings."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from soil_collector.utils.clip_model import CLIPModel
from soil_collector.utils.image_utils import collect_image_paths, load_image

logger = logging.getLogger(__name__)


def _phash_dedup(image_dir: Path, threshold: int = 10) -> set[str]:
    """Stage 1: Find duplicates via perceptual hashing (imagededup).

    Returns set of filenames (stems) to REMOVE (keep one per group).
    """
    from imagededup.methods import PHash

    logger.info(f"Stage 1 — Perceptual hashing (threshold={threshold})")
    hasher = PHash()

    # imagededup expects a directory path
    encodings = hasher.encode_images(image_dir=str(image_dir))
    duplicates = hasher.find_duplicates(
        encoding_map=encodings,
        max_distance_threshold=threshold,
    )

    # Determine which files to remove — keep the first in each group
    to_remove: set[str] = set()
    seen: set[str] = set()

    for filename, dups in duplicates.items():
        if filename in to_remove:
            continue
        if filename in seen:
            continue
        seen.add(filename)
        for dup in dups:
            if dup not in seen:
                to_remove.add(dup)
                seen.add(dup)

    logger.info(f"Stage 1: {len(to_remove)} duplicates found via PHash")
    return to_remove


def _clip_dedup(
    image_paths: list[Path],
    clip_model: CLIPModel,
    cosine_threshold: float = 0.95,
) -> set[str]:
    """Stage 2: Find semantic duplicates via CLIP embedding cosine similarity.

    Returns set of filenames to REMOVE.
    """
    logger.info(f"Stage 2 — CLIP embedding dedup (threshold={cosine_threshold})")

    if len(image_paths) < 2:
        return set()

    # Encode all images
    images = []
    valid_paths = []
    for p in tqdm(image_paths, desc="Loading images for CLIP dedup"):
        img = load_image(p)
        if img is not None:
            images.append(img)
            valid_paths.append(p)

    if len(images) < 2:
        return set()

    logger.info(f"Encoding {len(images)} images for CLIP dedup...")
    all_features = clip_model.encode_images(images)
    features_np = all_features.numpy()

    # Compute pairwise cosine similarity (features are already normalized)
    logger.info("Computing pairwise cosine similarities...")
    sim_matrix = features_np @ features_np.T

    # Find duplicates above threshold (upper triangle only)
    to_remove: set[str] = set()
    removed_indices: set[int] = set()

    for i in range(len(valid_paths)):
        if i in removed_indices:
            continue
        for j in range(i + 1, len(valid_paths)):
            if j in removed_indices:
                continue
            if sim_matrix[i, j] >= cosine_threshold:
                # Remove the second one
                to_remove.add(valid_paths[j].name)
                removed_indices.add(j)

    logger.info(f"Stage 2: {len(to_remove)} semantic duplicates found via CLIP")
    return to_remove


def run_deduplication(
    input_dir: Path,
    labels_path: Path,
    output_dir: Path,
    clip_model: CLIPModel,
    phash_threshold: int = 10,
    clip_cosine_threshold: float = 0.95,
) -> dict:
    """Run two-stage deduplication: PHash then CLIP.

    Copies surviving images to output_dir, updates labels.
    Returns stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = input_dir / "images" if (input_dir / "images").exists() else input_dir

    image_paths = collect_image_paths(images_dir)
    stats = {"total": len(image_paths), "removed_phash": 0, "removed_clip": 0, "kept": 0}

    if not image_paths:
        return stats

    # Stage 1: Perceptual hashing
    phash_remove = _phash_dedup(images_dir, phash_threshold)
    stats["removed_phash"] = len(phash_remove)

    # Filter out PHash duplicates
    surviving_paths = [p for p in image_paths if p.name not in phash_remove]
    logger.info(f"After PHash: {len(surviving_paths)} images remain")

    # Stage 2: CLIP embedding dedup
    clip_remove = _clip_dedup(surviving_paths, clip_model, clip_cosine_threshold)
    stats["removed_clip"] = len(clip_remove)

    # Final surviving set
    all_remove = phash_remove | clip_remove
    surviving = [p for p in image_paths if p.name not in all_remove]
    stats["kept"] = len(surviving)

    # Copy surviving images
    out_images_dir = output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(surviving, desc="Copying deduplicated images"):
        dest = out_images_dir / p.name
        if not dest.exists():
            shutil.copy2(p, dest)

    # Update labels
    if labels_path.exists():
        try:
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            surviving_names = {p.name for p in surviving}
            labels = [entry for entry in labels if entry.get("image") in surviving_names]
            out_labels = output_dir / "labels.json"
            out_labels.write_text(
                json.dumps(labels, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"Updated labels: {len(labels)} entries saved to {out_labels}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error updating labels: {e}")

    logger.info(
        f"Dedup done: {stats['kept']} kept, "
        f"{stats['removed_phash']} removed (PHash), "
        f"{stats['removed_clip']} removed (CLIP)"
    )
    return stats
