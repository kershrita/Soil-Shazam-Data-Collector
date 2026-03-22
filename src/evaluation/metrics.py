"""Compute accuracy metrics from human annotations vs CLIP predictions."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

LABEL_CATEGORIES = [
    "soil_color", "soil_texture", "particle_size",
    "crack_presence", "rock_fraction", "surface_structure", "surface_roughness",
]


def compute_metrics(eval_dir: Path) -> dict:
    """Compute evaluation metrics from completed annotations.

    Reads sample.json (with ground_truth filled in) and computes:
    - Filter precision and recall
    - Per-category label accuracy, top-2 accuracy
    - Confusion matrices per category
    - Confidence calibration (accuracy vs confidence bins)
    - Overall summary statistics

    Returns a dict with all metrics.
    """
    sample_path = eval_dir / "sample.json"
    if not sample_path.exists():
        logger.error(f"No sample.json found in {eval_dir}")
        return {}

    samples = json.loads(sample_path.read_text(encoding="utf-8"))

    # Split by source
    accepted = [s for s in samples if s["source"] == "accepted" and s["is_soil"] is not None]
    rejected = [s for s in samples if s["source"] == "rejected" and s["is_soil"] is not None]

    if not accepted:
        logger.error("No annotated accepted samples found")
        return {}

    metrics = {}

    # --- Filter metrics ---
    metrics["filter"] = _compute_filter_metrics(accepted, rejected)

    # --- Label accuracy (soil-confirmed samples that also have predictions) ---
    soil_confirmed = [s for s in samples if s.get("is_soil") is True and s.get("ground_truth")]
    label_eval_samples = [
        s for s in soil_confirmed
        if isinstance(s.get("predicted"), dict) and s.get("predicted")
    ]
    if label_eval_samples:
        metrics["labels"] = _compute_label_metrics(label_eval_samples)
        metrics["calibration"] = _compute_calibration(label_eval_samples)
    else:
        logger.warning("No soil-confirmed samples with both predictions and ground truth labels")
        metrics["labels"] = {}
        metrics["calibration"] = {}

    # --- Summary ---
    metrics["summary"] = _compute_summary(metrics, accepted, rejected)
    metrics["sample_size"] = {
        "total_annotated": len(accepted) + len(rejected),
        "accepted_annotated": len(accepted),
        "rejected_annotated": len(rejected),
        "soil_confirmed": len(soil_confirmed),
        "soil_confirmed_with_predictions": len(label_eval_samples),
        "soil_confirmed_without_predictions": len(soil_confirmed) - len(label_eval_samples),
    }

    # Save metrics
    metrics_path = eval_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    logger.info(f"Metrics saved to {metrics_path}")

    return metrics


def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 4)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _compute_filter_metrics(accepted: list[dict], rejected: list[dict]) -> dict:
    """Compute filter precision and recall."""
    # Precision: of accepted images, how many are truly soil?
    true_positives = sum(1 for s in accepted if s["is_soil"] is True)
    false_positives = sum(1 for s in accepted if s["is_soil"] is False)
    precision = true_positives / len(accepted) if accepted else 0

    result = {
        "precision": round(precision, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "accepted_evaluated": len(accepted),
    }

    # Recall: of rejected images, how many were actually soil?  (false negatives)
    if rejected:
        false_negatives = sum(1 for s in rejected if s["is_soil"] is True)
        true_negatives = sum(1 for s in rejected if s["is_soil"] is False)
        total_true_soil = true_positives + false_negatives
        recall = true_positives / total_true_soil if total_true_soil > 0 else None
        result.update({
            "recall": round(recall, 4) if recall is not None else None,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "rejected_evaluated": len(rejected),
        })

        if recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
            result["f1"] = round(f1, 4)
    else:
        result["recall"] = None
        result["rejected_evaluated"] = 0

    return result


def _compute_label_metrics(samples: list[dict]) -> dict:
    """Compute per-category label accuracy and confusion matrices."""
    results = {}

    for cat in LABEL_CATEGORIES:
        correct = 0
        total = 0
        confusion: dict[str, Counter] = defaultdict(Counter)
        per_label_correct: Counter = Counter()
        per_label_total: Counter = Counter()

        for s in samples:
            pred = (s.get("predicted") or {}).get(cat)
            gt = (s.get("ground_truth") or {}).get(cat)
            if pred is None or gt is None:
                continue

            total += 1
            per_label_total[gt] += 1
            confusion[gt][pred] += 1

            if pred == gt:
                correct += 1
                per_label_correct[gt] += 1

        accuracy = correct / total if total > 0 else 0

        # Per-label precision and recall
        all_labels = sorted(set(list(per_label_total.keys()) +
                               [pred for gt_counts in confusion.values()
                                for pred in gt_counts.keys()]))

        per_label = {}
        for label in all_labels:
            tp = confusion.get(label, Counter()).get(label, 0)
            # Recall: tp / total for this ground truth label
            gt_total = per_label_total.get(label, 0)
            label_recall = tp / gt_total if gt_total > 0 else 0
            # Precision: tp / total predicted as this label
            pred_total = sum(confusion[gt].get(label, 0) for gt in confusion)
            label_precision = tp / pred_total if pred_total > 0 else 0

            per_label[label] = {
                "precision": round(label_precision, 4),
                "recall": round(label_recall, 4),
                "support": gt_total,
            }

        # Build confusion matrix as nested dict
        confusion_dict = {
            gt: dict(counts) for gt, counts in confusion.items()
        }

        results[cat] = {
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total,
            "per_label": per_label,
            "confusion_matrix": confusion_dict,
        }

    return results


def _compute_calibration(samples: list[dict]) -> dict:
    """Compute confidence calibration: group predictions by confidence, check accuracy."""
    bins = [(0.0, 0.20), (0.20, 0.22), (0.22, 0.24), (0.24, 0.26),
            (0.26, 0.28), (0.28, 0.30), (0.30, 1.0)]
    bin_labels = ["<0.20", "0.20-0.22", "0.22-0.24", "0.24-0.26",
                  "0.26-0.28", "0.28-0.30", ">0.30"]

    calibration = {}

    for cat in LABEL_CATEGORIES:
        bin_correct = [0] * len(bins)
        bin_total = [0] * len(bins)

        for s in samples:
            pred = (s.get("predicted") or {}).get(cat)
            gt = (s.get("ground_truth") or {}).get(cat)
            conf = (s.get("confidence") or {}).get(cat)

            if pred is None or gt is None or conf is None:
                continue

            for i, (lo, hi) in enumerate(bins):
                if lo <= conf < hi:
                    bin_total[i] += 1
                    if pred == gt:
                        bin_correct[i] += 1
                    break

        calibration[cat] = {
            "bins": bin_labels,
            "accuracy": [
                round(bin_correct[i] / bin_total[i], 4) if bin_total[i] > 0 else None
                for i in range(len(bins))
            ],
            "counts": bin_total,
        }

    return calibration


def _compute_summary(metrics: dict, accepted: list, rejected: list) -> dict:
    """Compute overall summary statistics."""
    label_metrics = metrics.get("labels", {})
    if not label_metrics:
        return {"overall_label_accuracy": None}

    accuracies = [v["accuracy"] for v in label_metrics.values() if v.get("accuracy") is not None]
    overall_accuracy = round(sum(accuracies) / len(accuracies), 4) if accuracies else None

    # Category ranking (best to worst)
    category_ranking = sorted(
        [(cat, v["accuracy"]) for cat, v in label_metrics.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "overall_label_accuracy": overall_accuracy,
        "filter_precision": metrics.get("filter", {}).get("precision"),
        "filter_recall": metrics.get("filter", {}).get("recall"),
        "category_ranking": category_ranking,
    }
