"""Sample images from the dataset for human evaluation."""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

LABEL_CATEGORIES = [
    "soil_color",
    "soil_texture",
    "particle_size",
    "crack_presence",
    "rock_fraction",
    "surface_structure",
    "surface_roughness",
]


def _rejected_from_soil_log(soil_log_path: Path) -> list[str]:
    if not soil_log_path.exists():
        return []
    try:
        with open(soil_log_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return sorted(
                {
                    str(row.get("filename", "")).strip()
                    for row in reader
                    if str(row.get("filename", "")).strip()
                    and str(row.get("kept", "")).strip().lower() != "true"
                }
            )
    except OSError:
        return []


def _rejected_from_directory_diff(dataset_dir: Path) -> list[str]:
    deduped_root = dataset_dir.parent / "deduped"
    filtered_root = dataset_dir.parent / "filtered"
    deduped_images_dir = deduped_root / "images" if (deduped_root / "images").is_dir() else deduped_root
    filtered_images_dir = filtered_root / "images" if (filtered_root / "images").is_dir() else filtered_root

    if not (deduped_images_dir.exists() and filtered_images_dir.exists()):
        return []

    deduped_stems = {
        p.stem
        for p in deduped_images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }
    filtered_stems = {
        p.stem
        for p in filtered_images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }

    rejected_stems = deduped_stems - filtered_stems
    rejected_names: list[str] = []
    for stem in sorted(rejected_stems):
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = deduped_images_dir / f"{stem}{ext}"
            if candidate.exists():
                rejected_names.append(candidate.name)
                break
    return rejected_names


def create_eval_sample(
    dataset_dir: Path,
    output_dir: Path,
    n_accepted: int = 100,
    n_rejected: int = 30,
    seed: int = 42,
) -> Path:
    """Randomly sample images from labeled outputs for human annotation.

    Samples from:
    - Accepted images (final dataset): measure label accuracy + filter precision
    - Rejected images (filtered out): measure filter recall

    Returns path to the created sample file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "sample.json"

    rng = random.Random(seed)
    samples: list[dict] = []

    # Sample from accepted labels.
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
        confidence = {}
        if "scores" in entry:
            for cat in LABEL_CATEGORIES:
                if cat in entry["scores"]:
                    confidence[cat] = entry["scores"][cat].get("confidence", None)

        samples.append(
            {
                "image": entry["image"],
                "source": "accepted",
                "predicted": predicted,
                "confidence": confidence,
                "ground_truth": None,
                "is_soil": None,
            }
        )

    # Sample from rejected images.
    soil_log_path = dataset_dir.parent / "logs" / "soil_filter.csv"
    rejected_names = _rejected_from_soil_log(soil_log_path)
    if not rejected_names:
        rejected_names = _rejected_from_directory_diff(dataset_dir)

    if rejected_names:
        n_rejected = min(n_rejected, len(rejected_names))
        selected_rejected = rng.sample(rejected_names, n_rejected)
        for name in selected_rejected:
            samples.append(
                {
                    "image": name,
                    "source": "rejected",
                    "predicted": None,
                    "confidence": None,
                    "ground_truth": None,
                    "is_soil": None,
                }
            )
        logger.info(f"Sampled {n_rejected} rejected images for filter recall")
    else:
        logger.warning("No rejected images found for filter recall sampling")

    sample_path.write_text(
        json.dumps(samples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    accepted_count = sum(1 for s in samples if s["source"] == "accepted")
    rejected_count = sum(1 for s in samples if s["source"] == "rejected")
    logger.info(
        f"Evaluation sample created: {len(samples)} images "
        f"({accepted_count} accepted, {rejected_count} rejected) -> {sample_path}"
    )
    return sample_path
