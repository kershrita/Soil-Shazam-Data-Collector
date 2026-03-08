"""Dataset export — merge labels with corrections, produce final JSON + CSV."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_export(
    input_dir: Path,
    corrections_path: Path | None,
    output_dir: Path,
) -> dict:
    """Export the final clean dataset.

    - Merges auto-labels with manual corrections.
    - Copies images with sequential naming (soil_00001.jpg, ...).
    - Exports labels.json and labels.csv.
    - Prints distribution statistics.

    Returns stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # Load labels
    labels_path = input_dir / "labels.json"
    if not labels_path.exists():
        logger.error(f"No labels.json found in {input_dir}")
        return {"total": 0}

    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    # Apply corrections if available
    if corrections_path and corrections_path.exists():
        corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
        corrections_map = {}
        for c in corrections:
            if not c.get("_correct", True):
                corrections_map[c["image"]] = c

        applied = 0
        for entry in labels:
            if entry["image"] in corrections_map:
                corr = corrections_map[entry["image"]]
                for key, value in corr.items():
                    if key not in ("image", "_correct", "scores"):
                        entry[key] = value
                applied += 1

        if applied:
            logger.info(f"Applied {applied} manual corrections")

    # Determine image source directory
    images_in = input_dir / "images" if (input_dir / "images").exists() else input_dir

    # Copy images with sequential naming and update labels
    categories = [k for k in labels[0].keys() if k not in ("image", "scores")]
    final_labels = []

    for i, entry in enumerate(tqdm(labels, desc="Exporting dataset")):
        old_name = entry["image"]
        new_name = f"soil_{i + 1:05d}.jpg"

        src = images_in / old_name
        dst = images_out / new_name

        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

        # Build clean label entry (without raw scores for the main export)
        clean_entry = {"image": new_name}
        for cat in categories:
            clean_entry[cat] = entry.get(cat, "unknown")
        final_labels.append(clean_entry)

    # Save JSON
    json_path = output_dir / "labels.json"
    json_path.write_text(
        json.dumps(final_labels, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save CSV
    csv_path = output_dir / "labels.csv"
    df = pd.DataFrame(final_labels)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Also save full labels with scores for advanced use
    full_json_path = output_dir / "labels_full.json"
    full_json_path.write_text(
        json.dumps(labels, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Print distribution statistics
    stats = {"total": len(final_labels)}
    logger.info(f"\n{'=' * 60}")
    logger.info(f"DATASET EXPORT COMPLETE: {len(final_labels)} images")
    logger.info(f"{'=' * 60}")

    for cat in categories:
        dist = df[cat].value_counts()
        stats[cat] = dist.to_dict()
        logger.info(f"\n{cat}:")
        for label, count in dist.items():
            pct = count / len(df) * 100
            bar = "█" * int(pct / 2)
            logger.info(f"  {label:15s}: {count:5d} ({pct:5.1f}%) {bar}")

    logger.info(f"\nOutput files:")
    logger.info(f"  Images:      {images_out}")
    logger.info(f"  Labels JSON: {json_path}")
    logger.info(f"  Labels CSV:  {csv_path}")
    logger.info(f"  Full scores: {full_json_path}")

    return stats
