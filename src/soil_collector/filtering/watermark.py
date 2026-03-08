"""CLIP-based watermark and logo detection."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from tqdm import tqdm

from soil_collector.utils.clip_model import CLIPModel
from soil_collector.utils.image_utils import collect_image_paths, load_image

logger = logging.getLogger(__name__)


def run_watermark_filter(
    input_dir: Path,
    log_path: Path,
    clip_model: CLIPModel,
    watermark_prompts: list[str],
    clean_prompts: list[str],
    margin: float = 0.05,
) -> set[str]:
    """Identify watermarked images using CLIP.

    Compares average similarity to watermark prompts vs clean prompts.
    Returns set of filenames (stems) that are detected as watermarked.
    """
    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        return set()

    logger.info(f"Watermark filter: scanning {len(image_paths)} images")

    # Pre-encode text prompts
    wm_features = clip_model.encode_texts(watermark_prompts)
    clean_features = clip_model.encode_texts(clean_prompts)

    watermarked = set()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "watermark_score", "clean_score", "is_watermarked"])

        batch_size = clip_model.batch_size
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Watermark filter"):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            valid_paths = []
            for p in batch_paths:
                img = load_image(p)
                if img is not None:
                    images.append(img)
                    valid_paths.append(p)

            if not images:
                continue

            img_features = clip_model.encode_images(images)
            wm_scores = (img_features @ wm_features.T).mean(dim=1)
            clean_scores = (img_features @ clean_features.T).mean(dim=1)

            for path, wm_score, clean_score in zip(valid_paths, wm_scores, clean_scores):
                wm_val = wm_score.item()
                clean_val = clean_score.item()
                is_wm = wm_val - clean_val > margin

                writer.writerow([path.name, f"{wm_val:.4f}", f"{clean_val:.4f}", is_wm])

                if is_wm:
                    watermarked.add(path.stem)

    logger.info(f"Watermark filter: {len(watermarked)} watermarked images detected")
    return watermarked
