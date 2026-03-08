"""CLIP-based soil image filtering — keep only soil-related images."""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from soil_collector.utils.clip_model import CLIPModel
from soil_collector.utils.image_utils import collect_image_paths, load_image

logger = logging.getLogger(__name__)


def run_clip_filter(
    input_dir: Path,
    output_dir: Path,
    log_path: Path,
    clip_model: CLIPModel,
    positive_prompts: list[str],
    negative_prompts: list[str],
    threshold: float = 0.30,
    flagged_stems: set[str] | None = None,
) -> dict:
    """Filter images using CLIP — keep only soil-related images.

    Uses positive soil prompts vs negative prompts. Keeps images where
    avg positive similarity > threshold and positive > negative.

    Also removes any images flagged by overlay filter (watermark/text).

    Returns stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_image_paths(input_dir)
    flagged_stems = flagged_stems or set()

    stats = {"total": len(image_paths), "kept": 0, "discarded_soil": 0,
             "discarded_overlay": 0, "errors": 0}

    if not image_paths:
        return stats

    logger.info(f"CLIP soil filter: processing {len(image_paths)} images (threshold={threshold})")

    # Pre-encode text prompts
    pos_features = clip_model.encode_texts(positive_prompts)
    neg_features = clip_model.encode_texts(negative_prompts)

    # Existing output files for resume support
    existing = {p.stem for p in output_dir.rglob("*.jpg")}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "positive_score", "negative_score", "kept"])

        batch_size = clip_model.batch_size
        for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP soil filter"):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            valid_paths = []
            for p in batch_paths:
                # Skip overlay-flagged (watermark/text)
                if p.stem in flagged_stems:
                    stats["discarded_overlay"] += 1
                    writer.writerow([p.name, "N/A", "N/A", "overlay"])
                    continue
                # Skip already processed
                if p.stem in existing:
                    stats["kept"] += 1
                    continue

                img = load_image(p)
                if img is not None:
                    images.append(img)
                    valid_paths.append(p)
                else:
                    stats["errors"] += 1

            if not images:
                continue

            img_features = clip_model.encode_images(images)
            pos_scores = (img_features @ pos_features.T).mean(dim=1)
            neg_scores = (img_features @ neg_features.T).mean(dim=1)

            for path, pos_score, neg_score in zip(valid_paths, pos_scores, neg_scores):
                pos_val = pos_score.item()
                neg_val = neg_score.item()
                keep = pos_val >= threshold and pos_val > neg_val

                writer.writerow([path.name, f"{pos_val:.4f}", f"{neg_val:.4f}", keep])

                if keep:
                    dest = output_dir / path.name
                    shutil.copy2(path, dest)
                    stats["kept"] += 1
                else:
                    stats["discarded_soil"] += 1

    logger.info(
        f"CLIP filter done: {stats['kept']} kept, "
        f"{stats['discarded_soil']} discarded (not soil), "
        f"{stats['discarded_overlay']} discarded (overlay), "
        f"{stats['errors']} errors"
    )
    return stats
