"""CLIP-based multi-feature soil labeling."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from utils import CLIPModel, collect_image_paths, load_image

logger = logging.getLogger(__name__)


def run_clip_labeling(
    input_dir: Path,
    output_dir: Path,
    clip_model: CLIPModel,
    label_prompts: dict[str, dict[str, list[str] | str]],
) -> list[dict]:
    """Label filtered soil images using CLIP similarity scoring.

    Args:
        input_dir: Directory with filtered soil images.
        output_dir: Directory to copy images and save labels.
        clip_model: Loaded CLIPModel instance.
        label_prompts: Nested dict — {category: {label: prompt_or_list}}.
            Each label may map to a single prompt string or a list of prompts.
            When a list is given, scores are averaged across prompts (ensemble).

    Returns:
        List of label dicts, one per image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        logger.warning("No images found for labeling")
        return []

    logger.info(f"CLIP labeling: processing {len(image_paths)} images across {len(label_prompts)} categories")

    # Pre-encode all prompts per category.
    # Each label may have a list of prompts; scores are averaged across them (ensemble).
    category_data: dict[str, dict] = {}
    for category, labels_dict in label_prompts.items():
        label_names = []
        features_per_label = []
        for label_name, prompts in labels_dict.items():
            if isinstance(prompts, str):
                prompts = [prompts]
            label_names.append(label_name)
            features_per_label.append(clip_model.encode_texts(prompts))  # (n_prompts, dim)
        category_data[category] = {
            "labels": label_names,
            "features_per_label": features_per_label,
        }

    # Load existing labels for resume
    labels_path = output_dir / "labels.json"
    existing_labels = {}
    if labels_path.exists():
        try:
            data = json.loads(labels_path.read_text(encoding="utf-8"))
            existing_labels = {entry["image"]: entry for entry in data}
        except (json.JSONDecodeError, KeyError):
            pass

    all_labels: list[dict] = list(existing_labels.values())
    existing_images = set(existing_labels.keys())

    batch_size = clip_model.batch_size
    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP labeling"):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        valid_paths = []

        for p in batch_paths:
            if p.name in existing_images:
                continue
            img = load_image(p)
            if img is not None:
                images.append(img)
                valid_paths.append(p)

        if not images:
            continue

        # Encode images once
        img_features = clip_model.encode_images(images)

        # Score against each category
        for idx, path in enumerate(valid_paths):
            entry: dict = {"image": path.name, "scores": {}}
            img_feat = img_features[idx : idx + 1]  # (1, embed_dim)

            for category, cdata in category_data.items():
                scores = {}
                for label, label_feats in zip(cdata["labels"], cdata["features_per_label"]):
                    # Ensemble: average similarity across all prompts for this label
                    sim = (img_feat @ label_feats.T).mean().item()
                    scores[label] = round(sim, 4)
                # Sort by score descending
                sorted_labels = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                best_label = sorted_labels[0][0]
                best_score = sorted_labels[0][1]

                entry[category] = best_label
                entry["scores"][category] = {
                    "assigned": best_label,
                    "confidence": best_score,
                    "all_scores": scores,
                }
                if len(sorted_labels) > 1:
                    entry["scores"][category]["runner_up"] = sorted_labels[1][0]
                    entry["scores"][category]["runner_up_score"] = sorted_labels[1][1]

            all_labels.append(entry)

            # Copy image to output
            dest = images_dir / path.name
            if not dest.exists():
                shutil.copy2(path, dest)

    # Save labels
    labels_path.write_text(
        json.dumps(all_labels, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(f"CLIP labeling done: {len(all_labels)} images labeled, saved to {labels_path}")
    return all_labels
