"""Compute per-step statistics for the pipeline dashboard."""

from __future__ import annotations

from collections import Counter

from .helpers import (
    LABEL_CATEGORIES,
    build_soil_score_map,
    cache,
    extract_source,
    latest_cluster_run,
    list_dedup_removed_images,
    list_images,
    load_json_file,
    load_filter_log,
    load_labels,
)


def compute_step_stats(step_id: str, cfg: dict) -> dict:
    """Compute statistics for a pipeline step (cached)."""
    from .helpers import get_step

    cache_key = f"step_stats:{step_id}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    images = list_images(step_id, cfg)
    stats: dict = {"count": len(images)}
    step = get_step(step_id)

    if step_id == "cluster":
        run_dir = latest_cluster_run(cfg)
        if not run_dir:
            stats.update(
                {
                    "has_run": False,
                    "clusters": 0,
                    "review_queue_count": 0,
                    "suggestion_count": 0,
                    "suggested_slots": 0,
                }
            )
            cache.set(cache_key, stats)
            return stats

        summary = load_json_file(run_dir / "summary.json", {})
        review = load_json_file(run_dir / "review_queue.json", {})
        suggestions = load_json_file(run_dir / "suggestions.json", {})
        clusters_payload = load_json_file(run_dir / "clusters.json", {})

        counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
        quality = summary.get("quality_controls", {}) if isinstance(summary, dict) else {}

        review_items = review.get("items", []) if isinstance(review, dict) else []
        suggestion_items = suggestions.get("items", []) if isinstance(suggestions, dict) else []
        clusters_list = clusters_payload.get("clusters", []) if isinstance(clusters_payload, dict) else []

        stats.update(
            {
                "has_run": True,
                "run_id": summary.get("run_id", run_dir.name) if isinstance(summary, dict) else run_dir.name,
                "generated_at": summary.get("generated_at") if isinstance(summary, dict) else None,
                "count": int(counts.get("accepted_images", len(review_items))),
                "clusters": int(counts.get("clusters", len(clusters_list))),
                "review_queue_count": int(counts.get("review_queue_items", len(review_items))),
                "suggestion_count": int(counts.get("suggestion_items", len(suggestion_items))),
                "suggested_slots": int(
                    counts.get(
                        "suggested_category_slots",
                        sum(len(item.get("suggestions", {})) for item in suggestion_items),
                    )
                ),
                "review_only_categories": quality.get("review_only_categories", []) if isinstance(quality, dict) else [],
                "concentration_factor": (
                    (quality.get("priority_concentration") or {}).get("concentration_factor")
                    if isinstance(quality, dict)
                    else None
                ),
            }
        )

        flagged_dist: Counter = Counter()
        for item in review_items:
            for cat in item.get("flagged_categories", []):
                flagged_dist[cat] += 1
        if flagged_dist:
            stats["distributions"] = {"flagged_categories": dict(flagged_dist.most_common())}

    elif step_id == "download":
        sources: Counter = Counter()
        queries: Counter = Counter()
        for img in images:
            parts = img.split("/")
            if len(parts) >= 2:
                sources[parts[0]] += 1
            if len(parts) >= 3:
                queries[parts[1]] += 1
        stats["sources"] = dict(sources)
        stats["queries"] = dict(queries.most_common())
        stats["distributions"] = {src: {src: cnt} for src, cnt in sources.most_common()}

    elif step_id == "filter":
        soil_log = load_filter_log(cfg, "soil_filter.csv")
        overlay_log = load_filter_log(cfg, "overlay_filter.csv")
        if soil_log:
            kept = sum(
                1 for r in soil_log
                if r.get("kept", "").strip().lower() == "true"
            )
            stats["soil_filter"] = {
                "total": len(soil_log),
                "kept": kept,
                "rejected": len(soil_log) - kept,
            }
        if overlay_log:
            flagged = sum(
                1 for r in overlay_log
                if r.get("flagged", "").strip().lower() == "true"
            )
            stats["overlay_filter"] = {
                "total": len(overlay_log),
                "flagged": flagged,
                "clean": len(overlay_log) - flagged,
            }
        # Rejected images stats + per-source distributions
        soil_map = build_soil_score_map(cfg)
        overlay_count = sum(
            1 for info in soil_map.values()
            if info["kept"].lower() == "overlay"
        )
        rejected_total = sum(
            1 for info in soil_map.values()
            if info["kept"].lower() != "true"
        )
        stats["rejected_count"] = rejected_total
        stats["rejection_reasons"] = {
            "overlay": overlay_count,
            "low_soil_score": rejected_total - overlay_count,
        }
        # Build per-source kept/rejected distributions for chart
        kept_src: Counter = Counter(
            extract_source(fn) for fn, info in soil_map.items()
            if info["kept"].lower() == "true"
        )
        rejected_src: Counter = Counter(
            extract_source(fn) for fn, info in soil_map.items()
            if info["kept"].lower() != "true"
        )
        stats["distributions"] = {
            "kept": dict(kept_src.most_common()),
            "rejected": dict(rejected_src.most_common()),
        }
        all_filter_sources = sorted(set(kept_src) | set(rejected_src))
        stats["source_list"] = all_filter_sources

    elif step_id == "dedup":
        removed_names = list_dedup_removed_images(cfg)
        stats["removed_count"] = len(removed_names)

        kept_imgs = list_images(step_id, cfg)
        source_kept = Counter(extract_source(fn) for fn in kept_imgs)
        source_removed = Counter(extract_source(fn) for fn in removed_names)
        stats["distributions"] = {
            "kept": dict(source_kept.most_common()),
            "removed": dict(source_removed.most_common()),
        }
        stats["sources"] = {
            "kept": dict(source_kept.most_common()),
            "removed": dict(source_removed.most_common()),
        }
        stats["source_list"] = sorted(set(source_kept) | set(source_removed))

    elif step["has_labels"]:
        labels = load_labels(step_id, cfg)
        if labels:
            distributions = {}
            for cat in LABEL_CATEGORIES:
                dist = Counter(e.get(cat, "unknown") for e in labels.values())
                distributions[cat] = dict(dist.most_common())
            stats["distributions"] = distributions

    cache.set(cache_key, stats)
    return stats
