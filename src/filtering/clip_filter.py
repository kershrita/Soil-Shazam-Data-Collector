"""CLIP-based soil image filtering — keep only soil-related images."""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from utils import (
    CLIPModel,
    canonical_image_name,
    collect_image_paths,
    load_image,
    load_pipeline_manifest,
    mark_manifest_step,
    save_pipeline_manifest,
)

logger = logging.getLogger(__name__)


def run_clip_filter(
    input_dir: Path,
    output_dir: Path,
    log_path: Path,
    clip_model: CLIPModel,
    positive_prompts: list[str],
    negative_prompts: list[str],
    threshold: float = 0.30,
    flagged_names: set[str] | None = None,
    resume: bool = False,
    persist_kept_images: bool = True,
    manifest_path: Path | None = None,
) -> dict:
    """Filter images using CLIP — keep only soil-related images.

    Uses positive soil prompts vs negative prompts. Keeps images where
    avg positive similarity > threshold and positive > negative.

    Also removes any images flagged by overlay filter (watermark/text).
    When `persist_kept_images=False`, writes decision logs only (no image copies).

    Returns stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_image_paths(input_dir)
    flagged_names = flagged_names or set()
    name_root = input_dir / "images" if (input_dir / "images").is_dir() else input_dir
    manifest = load_pipeline_manifest(manifest_path) if manifest_path else None

    stats = {"total": len(image_paths), "kept": 0, "discarded_soil": 0,
             "discarded_overlay": 0, "errors": 0}

    if not image_paths:
        return stats

    logger.info(f"CLIP soil filter: processing {len(image_paths)} images (threshold={threshold})")

    # Pre-encode text prompts
    pos_features = clip_model.encode_texts(positive_prompts)
    neg_features = clip_model.encode_texts(negative_prompts)

    # Existing output files for resume support
    existing = {p.name for p in collect_image_paths(output_dir)} if resume else set()

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
                out_name = canonical_image_name(p, name_root)
                # Skip overlay-flagged (watermark/text)
                if out_name in flagged_names:
                    stats["discarded_overlay"] += 1
                    writer.writerow([out_name, "N/A", "N/A", "overlay"])
                    if manifest is not None:
                        mark_manifest_step(
                            manifest,
                            out_name,
                            "filter",
                            {
                                "kept": False,
                                "reason": "overlay",
                                "positive_score": None,
                                "negative_score": None,
                            },
                        )
                    continue
                # Skip already processed
                if out_name in existing:
                    stats["kept"] += 1
                    if manifest is not None:
                        mark_manifest_step(
                            manifest,
                            out_name,
                            "filter",
                            {
                                "kept": True,
                                "reason": "resume_existing",
                            },
                        )
                    continue

                img = load_image(p)
                if img is not None:
                    images.append(img)
                    valid_paths.append((p, out_name))
                else:
                    stats["errors"] += 1
                    if manifest is not None:
                        mark_manifest_step(
                            manifest,
                            out_name,
                            "filter",
                            {
                                "kept": False,
                                "reason": "load_error",
                            },
                        )

            if not images:
                continue

            img_features = clip_model.encode_images(images)
            pos_scores = (img_features @ pos_features.T).mean(dim=1)
            neg_scores = (img_features @ neg_features.T).mean(dim=1)

            for (path, out_name), pos_score, neg_score in zip(valid_paths, pos_scores, neg_scores):
                pos_val = pos_score.item()
                neg_val = neg_score.item()
                keep = pos_val >= threshold and pos_val > neg_val

                writer.writerow([out_name, f"{pos_val:.4f}", f"{neg_val:.4f}", keep])

                if keep:
                    if persist_kept_images:
                        dest = output_dir / out_name
                        shutil.copy2(path, dest)
                    stats["kept"] += 1
                    if manifest is not None:
                        mark_manifest_step(
                            manifest,
                            out_name,
                            "filter",
                            {
                                "kept": True,
                                "reason": "passed",
                                "positive_score": round(float(pos_val), 6),
                                "negative_score": round(float(neg_val), 6),
                            },
                        )
                else:
                    stats["discarded_soil"] += 1
                    if manifest is not None:
                        mark_manifest_step(
                            manifest,
                            out_name,
                            "filter",
                            {
                                "kept": False,
                                "reason": "low_soil_score",
                                "positive_score": round(float(pos_val), 6),
                                "negative_score": round(float(neg_val), 6),
                            },
                        )

    if manifest_path and manifest is not None:
        save_pipeline_manifest(manifest_path, manifest)

    logger.info(
        f"CLIP filter done: {stats['kept']} kept, "
        f"{stats['discarded_soil']} discarded (not soil), "
        f"{stats['discarded_overlay']} discarded (overlay), "
        f"{stats['errors']} errors"
    )
    return stats
