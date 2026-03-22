"""Sample images from the dataset for human evaluation."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

LABEL_CATEGORIES = [
    "soil_color", "soil_texture", "particle_size",
    "crack_presence", "rock_fraction", "surface_structure", "surface_roughness",
]


def create_eval_sample(
    dataset_dir: Path,
    output_dir: Path,
    n_accepted: int = 100,
    n_rejected: int = 30,
    seed: int = 42,
) -> Path:
    """Randomly sample images from labeled outputs for human annotation.

    Samples from:
    - Accepted images (final dataset) → measure label accuracy + filter precision
    - Rejected images (filtered out) → measure filter recall

    Returns path to the created sample file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "sample.json"

    rng = random.Random(seed)
    samples = []

    # --- Sample from accepted labels ---
    labels_path = dataset_dir / "labels_full.json"
    if not labels_path.exists():
        labels_path = dataset_dir / "labels.json"
    if not labels_path.exists():
        logger.error(f"No labels file found in {dataset_dir}")
        return sample_path

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    n_accepted = min(n_accepted, len(labels))
    selected = rng.sample(labels, n_accepted)

    for entry in selected:
        predicted = {cat: entry.get(cat, "unknown") for cat in LABEL_CATEGORIES}

        # Extract confidence scores if available
        confidence = {}
        if "scores" in entry:
            for cat in LABEL_CATEGORIES:
                if cat in entry["scores"]:
                    confidence[cat] = entry["scores"][cat].get("confidence", None)

        samples.append({
            "image": entry["image"],
            "source": "accepted",
            "predicted": predicted,
            "confidence": confidence,
            "ground_truth": None,  # filled in by annotator
            "is_soil": None,      # filled in by annotator
        })

    # --- Sample from rejected images (for filter recall) ---
    # Rejected images are in deduped but not in filtered.
    deduped_root = dataset_dir.parent / "deduped"
    filtered_root = dataset_dir.parent / "filtered"
    deduped_images_dir = deduped_root / "images" if (deduped_root / "images").is_dir() else deduped_root
    filtered_images_dir = filtered_root / "images" if (filtered_root / "images").is_dir() else filtered_root

    if deduped_images_dir.exists() and filtered_images_dir.exists():
        deduped_stems = {
            p.stem for p in deduped_images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        }
        filtered_stems = {
            p.stem for p in filtered_images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        }
        rejected_stems = deduped_stems - filtered_stems

        if rejected_stems:
            n_rejected = min(n_rejected, len(rejected_stems))
            selected_rejected = rng.sample(sorted(rejected_stems), n_rejected)

            for stem in selected_rejected:
                # Find actual filename
                for ext in (".jpg", ".jpeg", ".png", ".webp"):
                    candidate = deduped_images_dir / f"{stem}{ext}"
                    if candidate.exists():
                        samples.append({
                            "image": candidate.name,
                            "source": "rejected",
                            "predicted": None,
                            "confidence": None,
                            "ground_truth": None,
                            "is_soil": None,
                        })
                        break

            logger.info(f"Sampled {n_rejected} rejected images for filter recall")
        else:
            logger.warning("No rejected images found for filter recall sampling")
    else:
        logger.warning("Deduped/filtered dirs not found — skipping rejected sampling")

    # Save sample
    sample_path.write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    accepted_count = sum(1 for s in samples if s["source"] == "accepted")
    rejected_count = sum(1 for s in samples if s["source"] == "rejected")
    logger.info(
        f"Evaluation sample created: {len(samples)} images "
        f"({accepted_count} accepted, {rejected_count} rejected) -> {sample_path}"
    )
    return sample_path
