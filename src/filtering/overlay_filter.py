"""CLIP-based overlay detection — watermarks, logos, and text."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from tqdm import tqdm

from utils import CLIPModel, collect_image_paths, load_image

logger = logging.getLogger(__name__)


def run_overlay_filter(
    input_dir: Path,
    log_path: Path,
    clip_model: CLIPModel,
    overlay_prompts: list[str],
    clean_prompts: list[str],
    overlay_margin: float = 0.07,
) -> set[str]:
    """Identify images with watermarks, logos, or text overlays using CLIP.

    Compares average similarity to overlay prompts vs clean prompts.
    Returns set of filenames (stems) that are flagged.
    """
    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        return set()

    logger.info(f"Overlay filter: scanning {len(image_paths)} images")

    # Pre-encode text prompts
    overlay_features = clip_model.encode_texts(overlay_prompts)
    clean_features = clip_model.encode_texts(clean_prompts)

    flagged: set[str] = set()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "overlay_score", "clean_score", "flagged"])

        batch_size = clip_model.batch_size
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Overlay filter"):
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
            overlay_scores = (img_features @ overlay_features.T).mean(dim=1)
            clean_scores = (img_features @ clean_features.T).mean(dim=1)

            for path, ov_s, clean_s in zip(valid_paths, overlay_scores, clean_scores):
                ov_val = ov_s.item()
                clean_val = clean_s.item()
                is_flagged = ov_val - clean_val > overlay_margin

                writer.writerow([
                    path.name,
                    f"{ov_val:.4f}", f"{clean_val:.4f}",
                    is_flagged,
                ])

                if is_flagged:
                    flagged.add(path.stem)

    logger.info(f"Overlay filter: {len(flagged)} images flagged (watermark/text)")
    return flagged
