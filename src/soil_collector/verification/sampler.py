"""Random sampling and HTML report generation for manual verification."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def run_verification_sampling(
    input_dir: Path,
    output_dir: Path,
    sample_fraction: float = 0.07,
    seed: int = 42,
) -> Path:
    """Sample a fraction of labeled images and generate an HTML report.

    Args:
        input_dir: Directory with deduped images + labels.json.
        output_dir: Directory to write the HTML report.
        sample_fraction: Fraction of images to sample (0.0 to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Path to the generated HTML report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = input_dir / "labels.json"
    if not labels_path.exists():
        logger.error(f"No labels.json found in {input_dir}")
        return output_dir / "verification_report.html"

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    n_sample = max(1, int(len(labels) * sample_fraction))

    random.seed(seed)
    sampled = random.sample(labels, min(n_sample, len(labels)))

    logger.info(f"Sampled {len(sampled)} / {len(labels)} images for verification")

    # Determine image directory
    images_dir = input_dir / "images"
    if not images_dir.exists():
        images_dir = input_dir

    # Build label categories (exclude 'image' and 'scores')
    categories = [k for k in sampled[0].keys() if k not in ("image", "scores")]

    # Generate HTML report
    html = _build_html_report(sampled, categories, images_dir)
    report_path = output_dir / "verification_report.html"
    report_path.write_text(html, encoding="utf-8")

    # Also save the sampled entries for correction tracking
    sample_path = output_dir / "verification_sample.json"
    sample_path.write_text(
        json.dumps(sampled, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Create empty corrections template
    corrections_path = output_dir / "corrections.json"
    if not corrections_path.exists():
        template = [
            {
                "image": entry["image"],
                **{cat: entry.get(cat, "") for cat in categories},
                "_correct": True,
            }
            for entry in sampled
        ]
        corrections_path.write_text(
            json.dumps(template, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Corrections template saved to {corrections_path}")

    logger.info(f"Verification report saved to {report_path}")
    return report_path


def _build_html_report(
    samples: list[dict],
    categories: list[str],
    images_dir: Path,
) -> str:
    """Generate an HTML report showing sampled images with their labels."""

    rows = []
    for entry in samples:
        img_name = entry["image"]
        img_path = images_dir / img_name

        # Build label cells
        label_cells = []
        for cat in categories:
            value = entry.get(cat, "N/A")
            confidence = ""
            if "scores" in entry and cat in entry["scores"]:
                conf = entry["scores"][cat].get("confidence", 0)
                confidence = f" ({conf:.2f})"
            label_cells.append(f"<td><strong>{value}</strong>{confidence}</td>")

        rows.append(f"""
        <tr>
            <td><img src="{img_path.as_posix()}" style="max-width:300px; max-height:300px;"></td>
            <td>{img_name}</td>
            {"".join(label_cells)}
        </tr>""")

    header_cells = "".join(f"<th>{cat}</th>" for cat in categories)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Soil Dataset Verification Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4a7c59; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        img {{ border-radius: 4px; }}
        h1 {{ color: #333; }}
        .stats {{ background: #e8f5e9; padding: 12px; border-radius: 4px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Soil Dataset — Verification Report</h1>
    <div class="stats">
        <strong>Sampled images:</strong> {len(samples)}<br>
        <strong>Instructions:</strong> Review images and labels. Edit <code>corrections.json</code>
        to fix any mislabeled entries (set <code>_correct</code> to <code>false</code> and update label values).
    </div>
    <table>
        <tr>
            <th>Image</th>
            <th>Filename</th>
            {header_cells}
        </tr>
        {"".join(rows)}
    </table>
</body>
</html>"""
