"""Generate a shareable Markdown evaluation report."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LABEL_CATEGORIES = [
    "soil_color", "soil_texture", "particle_size",
    "crack_presence", "rock_fraction", "surface_structure", "surface_roughness",
]


def generate_report(eval_dir: Path) -> Path:
    """Generate a Markdown report from computed metrics.

    Reads metrics.json and produces report.md.
    Returns path to the report file.
    """
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        logger.error(f"No metrics.json found in {eval_dir}. Run eval-report first.")
        return eval_dir / "report.md"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report_path = eval_dir / "report.md"

    lines = []
    _header(lines, metrics)
    _filter_section(lines, metrics)
    _label_section(lines, metrics)
    _calibration_section(lines, metrics)
    _confusion_section(lines, metrics)
    _methodology_section(lines, metrics)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved to {report_path}")
    return report_path


def _header(lines: list[str], metrics: dict):
    sample = metrics.get("sample_size", {})
    summary = metrics.get("summary", {})

    lines.append("# Soil Image Collection Pipeline — Accuracy Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    overall = summary.get("overall_label_accuracy")
    precision = summary.get("filter_precision")
    recall = summary.get("filter_recall")

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **Filter Precision** (accepted images that are soil) | {_pct(precision)} |")
    lines.append(f"| **Filter Recall** (soil images correctly accepted) | {_pct(recall)} |")
    lines.append(f"| **Overall Label Accuracy** (avg across categories) | {_pct(overall)} |")
    lines.append(f"| Total images annotated | {sample.get('total_annotated', 0)} |")
    lines.append(f"| Accepted images evaluated | {sample.get('accepted_annotated', 0)} |")
    lines.append(f"| Rejected images evaluated | {sample.get('rejected_annotated', 0)} |")
    lines.append("")


def _filter_section(lines: list[str], metrics: dict):
    filt = metrics.get("filter", {})
    if not filt:
        return

    lines.append("## Filter Performance")
    lines.append("")
    lines.append("The soil/non-soil CLIP filter was evaluated by manually checking")
    lines.append("a random sample of accepted and rejected images.")
    lines.append("")
    lines.append("| Metric | Count | Rate |")
    lines.append("|--------|-------|------|")

    tp = filt.get("true_positives", 0)
    fp = filt.get("false_positives", 0)
    fn = filt.get("false_negatives", "—")
    tn = filt.get("true_negatives", "—")
    precision = filt.get("precision")
    recall = filt.get("recall")
    f1 = filt.get("f1")

    lines.append(f"| True Positives (correctly accepted soil) | {tp} | — |")
    lines.append(f"| False Positives (non-soil accepted) | {fp} | — |")
    lines.append(f"| False Negatives (soil incorrectly rejected) | {fn} | — |")
    lines.append(f"| True Negatives (non-soil correctly rejected) | {tn} | — |")
    lines.append(f"| **Precision** | — | {_pct(precision)} |")
    lines.append(f"| **Recall** | — | {_pct(recall)} |")
    if f1 is not None:
        lines.append(f"| **F1 Score** | — | {_pct(f1)} |")
    lines.append("")


def _label_section(lines: list[str], metrics: dict):
    labels = metrics.get("labels", {})
    if not labels:
        return

    lines.append("## Label Accuracy by Category")
    lines.append("")
    lines.append("Per-category accuracy of CLIP-assigned labels vs human ground truth.")
    lines.append("")
    lines.append("| Category | Accuracy | Correct / Total |")
    lines.append("|----------|----------|-----------------|")

    # Sort by accuracy descending
    sorted_cats = sorted(labels.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True)
    for cat, data in sorted_cats:
        acc = data.get("accuracy")
        correct = data.get("correct", 0)
        total = data.get("total", 0)
        lines.append(f"| {cat} | {_pct(acc)} | {correct} / {total} |")

    lines.append("")

    # Per-label detail for each category
    for cat in LABEL_CATEGORIES:
        data = labels.get(cat, {})
        per_label = data.get("per_label", {})
        if not per_label:
            continue

        lines.append(f"### {cat}")
        lines.append("")
        lines.append("| Label | Precision | Recall | Support |")
        lines.append("|-------|-----------|--------|---------|")
        for label, stats in sorted(per_label.items()):
            lines.append(
                f"| {label} | {_pct(stats.get('precision'))} "
                f"| {_pct(stats.get('recall'))} "
                f"| {stats.get('support', 0)} |"
            )
        lines.append("")


def _calibration_section(lines: list[str], metrics: dict):
    calibration = metrics.get("calibration", {})
    if not calibration:
        return

    lines.append("## Confidence Calibration")
    lines.append("")
    lines.append("Accuracy of predictions grouped by CLIP confidence score.")
    lines.append("Higher confidence should correlate with higher accuracy")
    lines.append("for a well-calibrated model.")
    lines.append("")

    # Aggregate across categories
    all_bins = None
    for cat, data in calibration.items():
        if all_bins is None:
            all_bins = data.get("bins", [])
            break

    if not all_bins:
        return

    lines.append("| Confidence Bin |" + " | ".join(
        cat.replace("_", " ").title() for cat in LABEL_CATEGORIES
    ) + " |")
    lines.append("|" + "---|" * (len(LABEL_CATEGORIES) + 1))

    for i, bin_label in enumerate(all_bins):
        row = f"| {bin_label} |"
        for cat in LABEL_CATEGORIES:
            cat_data = calibration.get(cat, {})
            accs = cat_data.get("accuracy", [])
            counts = cat_data.get("counts", [])
            if i < len(accs) and accs[i] is not None:
                row += f" {_pct(accs[i])} (n={counts[i]}) |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")


def _confusion_section(lines: list[str], metrics: dict):
    labels = metrics.get("labels", {})
    if not labels:
        return

    lines.append("## Confusion Matrices")
    lines.append("")
    lines.append("Rows = ground truth, columns = predicted.")
    lines.append("")

    for cat in LABEL_CATEGORIES:
        data = labels.get(cat, {})
        cm = data.get("confusion_matrix", {})
        if not cm:
            continue

        all_labels = sorted(set(
            list(cm.keys()) +
            [pred for gt_counts in cm.values() for pred in gt_counts.keys()]
        ))

        lines.append(f"### {cat}")
        lines.append("")
        header = "| GT \\ Pred |" + " | ".join(all_labels) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(all_labels) + 1))

        for gt in all_labels:
            row = f"| **{gt}** |"
            for pred in all_labels:
                count = cm.get(gt, {}).get(pred, 0)
                if gt == pred and count > 0:
                    row += f" **{count}** |"
                else:
                    row += f" {count} |"
            lines.append(row)
        lines.append("")


def _methodology_section(lines: list[str], metrics: dict):
    sample = metrics.get("sample_size", {})
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Sampling**: Random stratified sample from the final dataset")
    lines.append(f"- **Accepted images annotated**: {sample.get('accepted_annotated', 0)} "
                 f"(confirmed as soil: {sample.get('soil_confirmed', 0)})")
    lines.append(f"- **Rejected images annotated**: {sample.get('rejected_annotated', 0)}")
    lines.append("- **Ground truth**: Human expert annotation via the pipeline's evaluation UI")
    lines.append("- **Label accuracy**: Measured as exact match between CLIP prediction and human label")
    lines.append("- **Filter metrics**: Precision = TP/(TP+FP), Recall = TP/(TP+FN)")
    lines.append("- **Confidence calibration**: Predictions grouped by CLIP confidence score bins")
    lines.append("")


def _pct(value) -> str:
    """Format a 0-1 float as percentage string."""
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"
