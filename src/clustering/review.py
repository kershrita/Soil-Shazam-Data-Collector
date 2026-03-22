"""Cluster-assisted review queue generation for accepted soil images."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score

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


def run_cluster_review(
    labeled_dir: Path,
    eval_dir: Path,
    output_root: Path,
    cluster_cfg: dict[str, Any] | None = None,
    max_images: int = 0,
) -> dict[str, Any]:
    """Build cluster metadata, conservative suggestions, and a review queue.

    This command is advisory-only and does not mutate labels.
    """
    cluster_cfg = cluster_cfg or {}

    k_min = int(cluster_cfg.get("k_min", 18))
    k_max = int(cluster_cfg.get("k_max", 40))
    pca_dim_cfg = int(cluster_cfg.get("pca_dim", 128))
    knn_k = max(1, int(cluster_cfg.get("knn_k", 7)))
    vote_ratio_min = float(cluster_cfg.get("vote_ratio_min", 0.80))
    similarity_min = float(cluster_cfg.get("similarity_min", 0.28))
    margin_min = float(cluster_cfg.get("margin_min", 0.03))
    outlier_quantile_max = float(cluster_cfg.get("outlier_quantile_max", 0.90))
    random_seed = int(cluster_cfg.get("random_seed", 42))

    if k_min < 2:
        raise ValueError("clustering.k_min must be >= 2")
    if k_max < k_min:
        raise ValueError("clustering.k_max must be >= clustering.k_min")
    if not (0.0 < vote_ratio_min <= 1.0):
        raise ValueError("clustering.vote_ratio_min must be in (0, 1]")
    if not (0.0 <= outlier_quantile_max < 1.0):
        raise ValueError("clustering.outlier_quantile_max must be in [0, 1)")

    labels_hash_before = _labels_hash_snapshot(labeled_dir)

    records, labels_path, _image_dir = load_labeled_records(labeled_dir, max_images=max_images)
    logger.info("Cluster review: loaded %s accepted records from %s", len(records), labels_path)

    output_root.mkdir(parents=True, exist_ok=True)
    embeddings_path = labeled_dir / "embeddings.npz"
    embeddings_meta_path = labeled_dir / "embeddings_meta.json"
    records, embeddings = load_required_embeddings(
        records=records,
        labeled_dir=labeled_dir,
    )

    if len(records) < 3:
        raise ValueError("Need at least 3 valid accepted images to cluster")

    image_to_idx = {r["image"]: idx for idx, r in enumerate(records)}

    sample_path = eval_dir / "sample.json"
    metrics_path = eval_dir / "metrics.json"

    if not sample_path.exists():
        raise ValueError(
            "Missing evaluation/sample.json. Run eval-sample and complete annotation before cluster-review."
        )
    if not metrics_path.exists():
        raise ValueError(
            "Missing evaluation/metrics.json. Run eval-report to compute category reliability before cluster-review."
        )

    seed_map = collect_annotation_seeds(sample_path, set(image_to_idx.keys()))
    min_seed_count = max(knn_k, int(cluster_cfg.get("min_seed_count", 10)))
    if len(seed_map) < min_seed_count:
        raise ValueError(
            f"Insufficient annotation seeds: found {len(seed_map)}, need at least {min_seed_count}. "
            "Annotate more accepted soil samples and run eval-report."
        )

    category_accuracy = load_category_accuracy(metrics_path)
    weak_weights = {
        cat: 1.0 + (1.0 - float(category_accuracy.get(cat, 0.70)))
        for cat in LABEL_CATEGORIES
    }
    review_only_categories = sorted(
        cat for cat in LABEL_CATEGORIES if float(category_accuracy.get(cat, 0.70)) < 0.60
    )

    pca_dim = max(2, min(pca_dim_cfg, embeddings.shape[0], embeddings.shape[1]))
    reducer = PCA(n_components=pca_dim, random_state=random_seed)
    reduced = reducer.fit_transform(embeddings)

    k_selection = select_best_k(
        reduced,
        k_min=k_min,
        k_max=k_max,
        random_state=random_seed,
    )
    best_k = int(k_selection["best_k"])

    cluster_model = KMeans(
        n_clusters=best_k,
        random_state=random_seed,
        n_init=10,
        max_iter=300,
    )
    cluster_ids = cluster_model.fit_predict(reduced)
    centroid_distances = np.linalg.norm(reduced - cluster_model.cluster_centers_[cluster_ids], axis=1)

    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        cluster_to_indices[int(cid)].append(idx)

    cluster_distance_p90: dict[int, float] = {}
    cluster_distance_outlier_threshold: dict[int, float] = {}
    for cid, idxs in cluster_to_indices.items():
        values = centroid_distances[idxs]
        cluster_distance_p90[cid] = float(np.quantile(values, 0.90)) if len(values) else 0.0
        cluster_distance_outlier_threshold[cid] = float(np.quantile(values, outlier_quantile_max)) if len(values) else 0.0

    exemplar_scores = np.zeros(len(records), dtype=np.float32)
    for idx, cid in enumerate(cluster_ids):
        p90 = cluster_distance_p90.get(int(cid), 0.0)
        dist = float(centroid_distances[idx])
        if p90 <= 0:
            exemplar_scores[idx] = 1.0
        else:
            exemplar_scores[idx] = float(max(0.0, min(1.0, 1.0 - (dist / p90))))

    seed_indices = sorted(image_to_idx[image] for image in seed_map if image in image_to_idx)
    seed_truth_by_index = {image_to_idx[image]: truth for image, truth in seed_map.items() if image in image_to_idx}
    seed_votes = compute_seed_votes(
        embeddings=embeddings,
        seed_indices=seed_indices,
        seed_truth_by_index=seed_truth_by_index,
        knn_k=knn_k,
    )

    cluster_consensus = compute_cluster_consensus(records, cluster_ids)

    cluster_seed_coverage = {}
    for cid, idxs in cluster_to_indices.items():
        if not idxs:
            cluster_seed_coverage[cid] = 0.0
            continue
        seed_count = sum(1 for i in idxs if i in seed_truth_by_index)
        cluster_seed_coverage[cid] = seed_count / len(idxs)

    suggestions_items: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        cid = int(cluster_ids[idx])
        predicted = record["predicted"]
        margins = record["margins"]

        image_suggestions: dict[str, dict[str, Any]] = {}
        reasons: list[str] = []
        flagged_categories: list[str] = []
        risk_score = 0.0

        for cat in LABEL_CATEGORIES:
            weight = weak_weights[cat]
            pred_label = str(predicted.get(cat, "unknown"))
            margin = margins.get(cat)
            cinfo = cluster_consensus.get(cid, {}).get(cat)
            sinfo = seed_votes[idx].get(cat)

            if cat in review_only_categories:
                risk_score += 1.0 * weight
                flagged_categories.append(cat)
                reasons.append(f"{cat}:review_only")
                continue

            if cinfo and pred_label != cinfo["label"]:
                risk_score += 0.65 * weight
                flagged_categories.append(cat)
                reasons.append(f"{cat}:pred_vs_cluster")

            if sinfo and pred_label != sinfo["label"]:
                risk_score += 0.85 * weight
                flagged_categories.append(cat)
                reasons.append(f"{cat}:pred_vs_seed")

            if cinfo and sinfo and cinfo["label"] != sinfo["label"]:
                risk_score += 0.75 * weight
                flagged_categories.append(cat)
                reasons.append(f"{cat}:cluster_seed_disagree")

            if margin is None:
                risk_score += 0.20 * weight
                reasons.append(f"{cat}:missing_margin")
            elif margin < margin_min:
                deficit = (margin_min - margin) / max(margin_min, 1e-9)
                risk_score += min(1.2, 0.35 + deficit) * weight
                flagged_categories.append(cat)
                reasons.append(f"{cat}:low_margin")

            allow, blocked_reasons = conservative_suggestion_gate(
                review_only=False,
                cluster_label=cinfo["label"] if cinfo else None,
                cluster_ratio=float(cinfo["ratio"]) if cinfo else 0.0,
                seed_label=sinfo["label"] if sinfo else None,
                seed_ratio=float(sinfo["ratio"]) if sinfo else 0.0,
                seed_similarity=float(sinfo["similarity"]) if sinfo else None,
                margin=margin,
                vote_ratio_min=vote_ratio_min,
                similarity_min=similarity_min,
                margin_min=margin_min,
            )

            if allow and cinfo and sinfo:
                conf = suggestion_confidence(
                    cluster_ratio=float(cinfo["ratio"]),
                    seed_ratio=float(sinfo["ratio"]),
                    seed_similarity=float(sinfo["similarity"]),
                    similarity_min=similarity_min,
                )
                image_suggestions[cat] = {
                    "label": cinfo["label"],
                    "confidence": round(conf, 4),
                    "cluster_ratio": round(float(cinfo["ratio"]), 4),
                    "seed_ratio": round(float(sinfo["ratio"]), 4),
                    "seed_similarity": round(float(sinfo["similarity"]), 4),
                    "margin": round(float(margin), 4) if margin is not None else None,
                }
            elif blocked_reasons:
                reasons.append(f"{cat}:suggestion_blocked:{'+'.join(blocked_reasons)}")

        outlier_threshold = cluster_distance_outlier_threshold.get(cid, 0.0)
        dist = float(centroid_distances[idx])
        is_outlier = bool(outlier_threshold > 0 and dist >= outlier_threshold)
        if is_outlier:
            excess = (dist - outlier_threshold) / max(outlier_threshold, 1e-9)
            risk_score += 0.90 + min(1.50, excess)
            reasons.append("cluster:outlier")

        uncertainty_signal_count = len(
            [
                r for r in reasons
                if (":pred_vs_" in r) or (":low_margin" in r) or ("cluster:outlier" in r) or (":review_only" in r)
            ]
        )

        review_item = {
            "image": record["image"],
            "cluster_id": cid,
            "priority_score": round(risk_score, 4),
            "centroid_distance": round(dist, 6),
            "exemplar_score": round(float(exemplar_scores[idx]), 6),
            "is_outlier": is_outlier,
            "seed_coverage": round(float(cluster_seed_coverage.get(cid, 0.0)), 4),
            "predicted": predicted,
            "flagged_categories": sorted(set(flagged_categories)),
            "suggested_labels": {cat: info["label"] for cat, info in image_suggestions.items()},
            "uncertainty_signal_count": uncertainty_signal_count,
            "reasons": sorted(set(reasons)),
        }
        review_items.append(review_item)

        if image_suggestions:
            suggestions_items.append(
                {
                    "image": record["image"],
                    "cluster_id": cid,
                    "predicted": predicted,
                    "suggestions": image_suggestions,
                }
            )

    review_items.sort(key=lambda x: x["priority_score"], reverse=True)

    assignments = []
    for idx, record in enumerate(records):
        cid = int(cluster_ids[idx])
        assignments.append(
            {
                "image": record["image"],
                "cluster_id": cid,
                "centroid_distance": round(float(centroid_distances[idx]), 6),
                "exemplar_score": round(float(exemplar_scores[idx]), 6),
                "seed_coverage": round(float(cluster_seed_coverage.get(cid, 0.0)), 4),
                "is_seed": idx in seed_truth_by_index,
            }
        )

    clusters = []
    for cid in sorted(cluster_to_indices.keys()):
        idxs = cluster_to_indices[cid]
        exemplars = sorted(idxs, key=lambda i: centroid_distances[i])[:5]
        clusters.append(
            {
                "cluster_id": cid,
                "size": len(idxs),
                "seed_count": sum(1 for i in idxs if i in seed_truth_by_index),
                "seed_coverage": round(float(cluster_seed_coverage.get(cid, 0.0)), 4),
                "mean_distance": round(float(np.mean(centroid_distances[idxs])), 6),
                "p90_distance": round(float(cluster_distance_p90.get(cid, 0.0)), 6),
                "consensus": cluster_consensus.get(cid, {}),
                "top_exemplars": [records[i]["image"] for i in exemplars],
            }
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    clusters_payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "k": best_k,
        "pca_dim": pca_dim,
        "n_images": len(records),
        "n_seeds": len(seed_truth_by_index),
        "clusters": clusters,
        "assignments": assignments,
    }

    suggestions_payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_only_categories": review_only_categories,
        "items": suggestions_items,
    }

    review_payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": review_items,
    }

    concentration = compute_priority_concentration(review_items)

    labels_hash_after = _labels_hash_snapshot(labeled_dir)
    labels_unchanged = labels_hash_before == labels_hash_after

    summary = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "labeled_dir": str(labeled_dir),
            "eval_dir": str(eval_dir),
            "labels_path": str(labels_path),
            "evaluation_sample": str(sample_path),
            "evaluation_metrics": str(metrics_path),
            "embeddings_path": str(embeddings_path),
            "embeddings_meta_path": str(embeddings_meta_path),
            "embedding_model_name": _read_embedding_meta_field(embeddings_meta_path, "model_name"),
            "embedding_pretrained": _read_embedding_meta_field(embeddings_meta_path, "pretrained"),
        },
        "settings": {
            "k_min": k_min,
            "k_max": k_max,
            "selected_k": best_k,
            "pca_dim": pca_dim,
            "knn_k": knn_k,
            "vote_ratio_min": vote_ratio_min,
            "similarity_min": similarity_min,
            "margin_min": margin_min,
            "outlier_quantile_max": outlier_quantile_max,
            "random_seed": random_seed,
        },
        "counts": {
            "accepted_images": len(records),
            "annotation_seeds": len(seed_truth_by_index),
            "clusters": len(clusters),
            "review_queue_items": len(review_items),
            "suggestion_items": len(suggestions_items),
            "suggested_category_slots": int(
                sum(len(item.get("suggestions", {})) for item in suggestions_items)
            ),
        },
        "quality_controls": {
            "review_only_categories": review_only_categories,
            "category_accuracy": {cat: round(float(category_accuracy.get(cat, 0.70)), 4) for cat in LABEL_CATEGORIES},
            "priority_concentration": concentration,
            "source_labels_unchanged": labels_unchanged,
        },
        "k_selection": k_selection,
        "artifacts": {
            "run_dir": str(run_dir),
            "clusters": str(run_dir / "clusters.json"),
            "review_queue": str(run_dir / "review_queue.json"),
            "suggestions": str(run_dir / "suggestions.json"),
            "summary": str(run_dir / "summary.json"),
        },
    }

    _write_json(run_dir / "clusters.json", clusters_payload)
    _write_json(run_dir / "review_queue.json", review_payload)
    _write_json(run_dir / "suggestions.json", suggestions_payload)
    _write_json(run_dir / "summary.json", summary)

    if not labels_unchanged:
        logger.warning("Source label artifacts changed during cluster-review (unexpected)")

    logger.info("Cluster review complete: %s", run_dir)
    return summary


def load_labeled_records(
    labeled_dir: Path,
    max_images: int = 0,
) -> tuple[list[dict[str, Any]], Path, Path]:
    """Load accepted labeled records from labeled output."""
    labels_full = labeled_dir / "labels_full.json"
    labels_basic = labeled_dir / "labels.json"

    labels_path = labels_full if labels_full.exists() else labels_basic
    if not labels_path.exists():
        raise ValueError(f"Missing labels file in {labeled_dir}. Expected labels_full.json or labels.json")

    image_dir = labeled_dir / "images" if (labeled_dir / "images").exists() else labeled_dir

    entries = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No accepted labels found in {labels_path}")

    records: list[dict[str, Any]] = []
    for entry in entries:
        image = str(entry.get("image", "")).strip()
        if not image:
            continue
        if not (image_dir / image).exists():
            continue

        predicted = {cat: str(entry.get(cat, "unknown")) for cat in LABEL_CATEGORIES}
        margins = {cat: extract_margin(entry, cat) for cat in LABEL_CATEGORIES}

        records.append(
            {
                "image": image,
                "predicted": predicted,
                "margins": margins,
            }
        )

    records.sort(key=lambda r: r["image"])
    if max_images > 0:
        records = records[:max_images]

    if not records:
        raise ValueError(
            f"No valid accepted records with image files found in {labels_path} and {image_dir}"
        )

    return records, labels_path, image_dir


def extract_margin(entry: dict[str, Any], category: str) -> float | None:
    """Extract confidence margin (assigned - runner_up) when available."""
    scores = entry.get("scores")
    if not isinstance(scores, dict):
        return None
    cat_scores = scores.get(category)
    if not isinstance(cat_scores, dict):
        return None
    conf = _to_float(cat_scores.get("confidence"))
    runner = _to_float(cat_scores.get("runner_up_score"))
    if conf is None or runner is None:
        return None
    return float(conf - runner)


def load_required_embeddings(
    records: list[dict[str, Any]],
    labeled_dir: Path,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Load persisted embeddings from the label step.

    Cluster review in v1 is strictly read-only and must reuse label embeddings.
    """
    embeddings_path = labeled_dir / "embeddings.npz"
    if not embeddings_path.exists():
        raise ValueError(
            f"Missing {embeddings_path}. Run the label step first to generate embeddings."
        )

    try:
        cached = np.load(embeddings_path)
        cached_images = [str(x) for x in cached["images"].tolist()]
        cached_embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
    except (OSError, ValueError, KeyError) as err:
        raise ValueError(
            f"Invalid embeddings cache at {embeddings_path}: {err}. Run label again."
        ) from err

    if cached_embeddings.ndim != 2 or len(cached_images) != cached_embeddings.shape[0]:
        raise ValueError(
            f"Invalid embedding shape in {embeddings_path}. Run label again to regenerate embeddings."
        )

    embedding_map = {name: cached_embeddings[idx] for idx, name in enumerate(cached_images)}
    expected_images = [record["image"] for record in records]
    missing = [name for name in expected_images if name not in embedding_map]
    if missing:
        raise ValueError(
            "Label embeddings are incomplete for the current accepted set "
            f"(missing {len(missing)} images). Run label again before cluster-review."
        )

    ordered = np.vstack([embedding_map[name] for name in expected_images]).astype(np.float32)
    logger.info(
        "Cluster review: loaded %s persisted label embeddings from %s",
        len(records),
        embeddings_path,
    )
    return records, ordered


def _read_embedding_meta_field(meta_path: Path, key: str) -> str | None:
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    value = payload.get(key)
    if value is None:
        return None
    return str(value)


def collect_annotation_seeds(
    sample_path: Path,
    known_images: set[str],
) -> dict[str, dict[str, str]]:
    """Collect accepted soil annotations with complete ground truth labels."""
    samples = json.loads(sample_path.read_text(encoding="utf-8"))
    seeds: dict[str, dict[str, str]] = {}

    for sample in samples:
        image = str(sample.get("image", "")).strip()
        if not image or image not in known_images:
            continue
        if sample.get("source") != "accepted":
            continue
        if sample.get("is_soil") is not True:
            continue
        gt = sample.get("ground_truth")
        if not isinstance(gt, dict):
            continue

        normalized: dict[str, str] = {}
        for cat in LABEL_CATEGORIES:
            value = str(gt.get(cat, "")).strip()
            if not value:
                normalized = {}
                break
            normalized[cat] = value

        if normalized:
            seeds[image] = normalized

    return seeds


def load_category_accuracy(metrics_path: Path) -> dict[str, float]:
    """Load per-category accuracy from evaluation metrics."""
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    labels = metrics.get("labels", {})
    result: dict[str, float] = {}
    for cat in LABEL_CATEGORIES:
        value = _to_float((labels.get(cat) or {}).get("accuracy"))
        if value is not None:
            result[cat] = float(value)
    return result


def select_best_k(
    features: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int = 42,
) -> dict[str, Any]:
    """Auto-select K using silhouette and Calinski-Harabasz signals."""
    if features.ndim != 2 or features.shape[0] < 3:
        raise ValueError("Need at least 3 samples to auto-select cluster count")

    n_samples = int(features.shape[0])
    min_k = max(2, int(k_min))
    max_k = min(int(k_max), n_samples - 1)

    if min_k > max_k:
        fallback_k = max(2, min(n_samples - 1, int(k_min), int(k_max)))
        model = KMeans(n_clusters=fallback_k, random_state=random_state, n_init=10, max_iter=300)
        labels = model.fit_predict(features)
        sil = float(silhouette_score(features, labels)) if len(set(labels)) > 1 else -1.0
        ch = float(calinski_harabasz_score(features, labels)) if len(set(labels)) > 1 else 0.0
        return {
            "best_k": fallback_k,
            "candidates": [
                {
                    "k": fallback_k,
                    "silhouette": round(sil, 6),
                    "calinski_harabasz": round(ch, 4),
                    "objective": round((sil + 1.0) / 2.0, 6),
                }
            ],
        }

    candidates: list[dict[str, float]] = []
    sample_size = min(n_samples, 1200)

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = model.fit_predict(features)

        if len(set(labels)) < 2:
            sil = -1.0
            ch = 0.0
        else:
            sil = float(
                silhouette_score(
                    features,
                    labels,
                    sample_size=sample_size,
                    random_state=random_state,
                )
            )
            ch = float(calinski_harabasz_score(features, labels))

        candidates.append(
            {
                "k": float(k),
                "silhouette": sil,
                "calinski_harabasz": ch,
            }
        )

    max_ch = max((c["calinski_harabasz"] for c in candidates), default=1.0)
    if max_ch <= 0:
        max_ch = 1.0

    best_k = int(candidates[0]["k"])
    best_obj = -math.inf
    out_rows: list[dict[str, Any]] = []

    for c in candidates:
        sil_norm = (c["silhouette"] + 1.0) / 2.0
        ch_norm = c["calinski_harabasz"] / max_ch
        objective = (0.7 * sil_norm) + (0.3 * ch_norm)

        k_val = int(c["k"])
        out_rows.append(
            {
                "k": k_val,
                "silhouette": round(float(c["silhouette"]), 6),
                "calinski_harabasz": round(float(c["calinski_harabasz"]), 4),
                "objective": round(float(objective), 6),
            }
        )

        if objective > best_obj or (abs(objective - best_obj) < 1e-9 and k_val < best_k):
            best_obj = objective
            best_k = k_val

    return {
        "best_k": best_k,
        "candidates": out_rows,
    }


def compute_cluster_consensus(
    records: list[dict[str, Any]],
    cluster_ids: np.ndarray,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Compute top predicted label and ratio per category inside each cluster."""
    buckets: dict[int, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    for idx, record in enumerate(records):
        cid = int(cluster_ids[idx])
        predicted = record.get("predicted", {})
        for cat in LABEL_CATEGORIES:
            label = str(predicted.get(cat, "unknown"))
            buckets[cid][cat][label] += 1

    result: dict[int, dict[str, dict[str, Any]]] = {}
    for cid, cat_counts in buckets.items():
        result[cid] = {}
        for cat, counts in cat_counts.items():
            total = sum(counts.values())
            if total <= 0:
                continue
            label, count = counts.most_common(1)[0]
            result[cid][cat] = {
                "label": label,
                "ratio": round(count / total, 6),
                "count": int(count),
                "total": int(total),
            }

    return result


def compute_seed_votes(
    embeddings: np.ndarray,
    seed_indices: list[int],
    seed_truth_by_index: dict[int, dict[str, str]],
    knn_k: int,
) -> list[dict[str, dict[str, Any]]]:
    """Find nearest annotation seeds and vote labels per category for each image."""
    n = embeddings.shape[0]
    if not seed_indices:
        return [{} for _ in range(n)]

    seed_emb = embeddings[seed_indices]
    sim = embeddings @ seed_emb.T

    top_k = max(1, min(knn_k, len(seed_indices)))
    top_idx = np.argpartition(sim, -top_k, axis=1)[:, -top_k:]

    seed_lookup = {local_idx: seed_indices[local_idx] for local_idx in range(len(seed_indices))}

    result: list[dict[str, dict[str, Any]]] = []
    for i in range(n):
        row_locals = top_idx[i]
        row_sims = sim[i, row_locals]
        order = np.argsort(-row_sims)

        nearest_global: list[int] = []
        nearest_sims: list[float] = []

        for ord_idx in order:
            local_seed_idx = int(row_locals[ord_idx])
            global_seed_idx = seed_lookup[local_seed_idx]
            nearest_global.append(global_seed_idx)
            nearest_sims.append(float(row_sims[ord_idx]))

        cat_votes: dict[str, dict[str, Any]] = {}
        for cat in LABEL_CATEGORIES:
            label_counts: Counter = Counter()
            label_sims: dict[str, list[float]] = defaultdict(list)

            for global_seed_idx, score in zip(nearest_global, nearest_sims):
                truth = seed_truth_by_index.get(global_seed_idx, {})
                label = truth.get(cat)
                if not label:
                    continue
                label_counts[label] += 1
                label_sims[label].append(score)

            if not label_counts:
                continue

            best_label, best_count = label_counts.most_common(1)[0]
            best_sims = label_sims.get(best_label, [])
            avg_sim = float(sum(best_sims) / len(best_sims)) if best_sims else 0.0

            cat_votes[cat] = {
                "label": best_label,
                "ratio": round(best_count / top_k, 6),
                "similarity": round(avg_sim, 6),
                "support": int(best_count),
            }

        result.append(cat_votes)

    return result


def conservative_suggestion_gate(
    review_only: bool,
    cluster_label: str | None,
    cluster_ratio: float,
    seed_label: str | None,
    seed_ratio: float,
    seed_similarity: float | None,
    margin: float | None,
    vote_ratio_min: float,
    similarity_min: float,
    margin_min: float,
) -> tuple[bool, list[str]]:
    """Strict gating for conservative label suggestions."""
    reasons: list[str] = []

    if review_only:
        reasons.append("review_only")
    if not cluster_label:
        reasons.append("no_cluster_consensus")
    if not seed_label:
        reasons.append("no_seed_vote")
    if cluster_label and seed_label and cluster_label != seed_label:
        reasons.append("cluster_seed_disagree")
    if cluster_ratio < vote_ratio_min:
        reasons.append("cluster_ratio_low")
    if seed_ratio < vote_ratio_min:
        reasons.append("seed_ratio_low")
    if seed_similarity is None or seed_similarity < similarity_min:
        reasons.append("seed_similarity_low")
    if margin is None or margin < margin_min:
        reasons.append("margin_low")

    return (len(reasons) == 0), reasons


def suggestion_confidence(
    cluster_ratio: float,
    seed_ratio: float,
    seed_similarity: float,
    similarity_min: float,
) -> float:
    """Compute a bounded confidence score for accepted suggestions."""
    sim_scaled = 0.0
    if seed_similarity is not None:
        denom = max(1e-9, 1.0 - similarity_min)
        sim_scaled = max(0.0, min(1.0, (seed_similarity - similarity_min) / denom))
    return (0.40 * cluster_ratio) + (0.35 * seed_ratio) + (0.25 * sim_scaled)


def compute_priority_concentration(review_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare uncertainty concentration in top-priority slice vs random baseline."""
    if not review_items:
        return {
            "top_slice_size": 0,
            "baseline_uncertainty_rate": 0.0,
            "top_slice_uncertainty_rate": 0.0,
            "concentration_factor": None,
            "target_factor": 1.5,
            "passes": False,
        }

    n = len(review_items)
    top_n = max(10, int(0.10 * n))
    top_n = min(top_n, n)

    def uncertain_score(item: dict[str, Any]) -> float:
        reasons = item.get("reasons", [])
        severe = sum(
            1 for r in reasons
            if (":pred_vs_" in r) or (":cluster_seed_disagree" in r) or (":low_margin" in r) or (":review_only" in r)
        )
        outlier = 1 if item.get("is_outlier") else 0
        return float(severe + outlier)

    scores = [uncertain_score(item) for item in review_items]
    threshold = float(np.quantile(scores, 0.75)) if scores else 0.0
    all_flags = [1 if s >= threshold and s > 0 else 0 for s in scores]
    top_flags = all_flags[:top_n]

    baseline_rate = float(sum(all_flags) / len(all_flags)) if all_flags else 0.0
    top_rate = float(sum(top_flags) / len(top_flags)) if top_flags else 0.0

    factor = None
    if baseline_rate > 0:
        factor = top_rate / baseline_rate

    target = 1.5
    passes = bool(factor is not None and factor >= target)

    return {
        "top_slice_size": top_n,
        "baseline_uncertainty_rate": round(baseline_rate, 6),
        "top_slice_uncertainty_rate": round(top_rate, 6),
        "high_uncertainty_threshold": round(threshold, 6),
        "concentration_factor": round(factor, 6) if factor is not None else None,
        "target_factor": target,
        "passes": passes,
    }


def _labels_hash_snapshot(labeled_dir: Path) -> dict[str, str | None]:
    """Hash source label artifacts to verify read-only behavior of cluster-review."""
    targets = [
        labeled_dir / "labels.json",
        labeled_dir / "labels_full.json",
    ]
    out: dict[str, str | None] = {}
    for path in targets:
        out[str(path)] = _sha256_file(path) if path.exists() else None
    return out


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
