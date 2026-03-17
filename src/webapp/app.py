"""Flask web app for Soil Shazam Data Collector."""

from __future__ import annotations

import io
import json
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from PIL import Image

from .helpers import (
    LABEL_CATEGORIES,
    STEPS,
    _APP_DIR,
    _PROJECT_ROOT,
    build_overlay_score_map,
    build_resize_hash_map,
    build_soil_score_map,
    compute_filter_metrics,
    extract_source,
    find_raw_original,
    get_filter_thresholds,
    get_image_dimensions,
    get_rejection_reason,
    get_step,
    images_dir,
    list_dedup_removed_images,
    list_images,
    list_rejected_images,
    load_dedup_groups,
    load_full_labels,
    load_labels,
    load_yaml,
    resolve,
)
from .stats import compute_step_stats


# ─── App factory ──────────────────────────────────────────────────────────────

def create_app(config_dir: Path | None = None) -> Flask:
    config_dir = config_dir or _PROJECT_ROOT / "config"
    cfg = load_yaml(config_dir / "pipeline.yaml")
    prompts = load_yaml(config_dir / "prompts.yaml")

    # Extract valid label options from prompts
    label_options_all: dict[str, list[str]] = {}
    for cat in LABEL_CATEGORIES:
        label_options_all[cat] = list(prompts.get("labeling", {}).get(cat, {}).keys())

    app = Flask(
        __name__,
        template_folder=str(_APP_DIR / "templates"),
        static_folder=str(_APP_DIR / "static"),
    )

    # ─── Page routes ──────────────────────────────────────────────────────

    @app.route("/")
    def dashboard():
        step_stats = []
        for step in STEPS:
            stats = compute_step_stats(step["id"], cfg)
            step_stats.append({**step, "stats": stats})

        funnel_rows = []
        anomalies = []
        for idx in range(1, len(step_stats)):
            prev_step = step_stats[idx - 1]
            curr_step = step_stats[idx]
            prev_count = int(prev_step["stats"].get("count", 0))
            curr_count = int(curr_step["stats"].get("count", 0))
            conversion_pct = (curr_count / prev_count * 100.0) if prev_count else None
            drop_pct = (100.0 - conversion_pct) if conversion_pct is not None else None
            funnel_rows.append(
                {
                    "from_name": prev_step["name"],
                    "to_name": curr_step["name"],
                    "from_count": prev_count,
                    "to_count": curr_count,
                    "conversion_pct": conversion_pct,
                    "drop_pct": drop_pct,
                }
            )
            if conversion_pct is not None and conversion_pct < 35:
                anomalies.append(
                    {
                        "severity": "high",
                        "message": (
                            f"{curr_step['name']} retained only {conversion_pct:.1f}% "
                            f"of {prev_step['name']} output."
                        ),
                    }
                )

        filter_step = next((s for s in step_stats if s["id"] == "filter"), None)
        if filter_step:
            kept = int(filter_step["stats"].get("count", 0))
            rejected = int(filter_step["stats"].get("rejected_count", 0))
            total = kept + rejected
            if total:
                rejection_rate = rejected / total * 100.0
                if rejection_rate > 75:
                    anomalies.append(
                        {
                            "severity": "high",
                            "message": (
                                f"Filter rejection rate is {rejection_rate:.1f}% "
                                f"({rejected:,} rejected of {total:,})."
                            ),
                        }
                    )

        dedup_step = next((s for s in step_stats if s["id"] == "dedup"), None)
        if dedup_step:
            kept = int(dedup_step["stats"].get("count", 0))
            removed = int(dedup_step["stats"].get("removed_count", 0))
            total = kept + removed
            if total:
                removed_rate = removed / total * 100.0
                if removed_rate > 45:
                    anomalies.append(
                        {
                            "severity": "warn",
                            "message": (
                                f"Dedup removed {removed_rate:.1f}% "
                                f"({removed:,} of {total:,}) - review source overlap."
                            ),
                        }
                    )

        return render_template(
            "dashboard.html",
            steps=step_stats,
            categories=LABEL_CATEGORIES,
            funnel_rows=funnel_rows,
            anomalies=anomalies,
        )

    @app.route("/step/<step_id>")
    def step_browser(step_id):
        step = get_step(step_id)
        if not step:
            return "Step not found", 404

        view = request.args.get("view", "kept", type=str)
        stats = compute_step_stats(step_id, cfg)

        # Build label filter options from data
        label_options = {}
        if step["has_labels"]:
            for cat in LABEL_CATEGORIES:
                if "distributions" in stats and cat in stats["distributions"]:
                    label_options[cat] = sorted(stats["distributions"][cat].keys())
                else:
                    label_options[cat] = label_options_all.get(cat, [])
        filter_thresholds = get_filter_thresholds(cfg) if step_id == "filter" else None

        return render_template(
            "browser.html",
            step=step,
            stats=stats,
            view=view,
            categories=LABEL_CATEGORIES,
            label_options=label_options,
            label_options_all=label_options_all,
            steps=STEPS,
            filter_thresholds=filter_thresholds,
        )

    # ─── Image serving ────────────────────────────────────────────────────

    @app.route("/images/<step_id>/<path:filename>")
    def serve_image(step_id, filename):
        view = request.args.get("view", "kept", type=str)
        if step_id == "filter" and view == "rejected":
            deduped_dir = resolve(cfg["paths"]["deduped"])
            img_dir = deduped_dir / "images" if (deduped_dir / "images").is_dir() else deduped_dir
        elif step_id == "dedup" and view == "removed":
            resized_dir = resolve(cfg["paths"]["resized"])
            img_dir = resized_dir / "images" if (resized_dir / "images").is_dir() else resized_dir
        else:
            img_dir = images_dir(step_id, cfg)
        return send_from_directory(img_dir, filename)

    @app.route("/images/raw/<path:filename>")
    def serve_raw_image(filename):
        raw_dir = resolve(cfg["paths"]["raw"])
        return send_from_directory(raw_dir, filename)

    # ─── Thumbnail route ──────────────────────────────────────────────────

    @app.route("/thumbnails/<step_id>/<path:filename>")
    def serve_thumbnail(step_id, filename):
        """Serve a downscaled thumbnail (200px max) with disk caching."""
        view = request.args.get("view", "kept", type=str)
        if step_id == "filter" and view == "rejected":
            deduped_dir = resolve(cfg["paths"]["deduped"])
            img_dir = deduped_dir / "images" if (deduped_dir / "images").is_dir() else deduped_dir
        elif step_id == "dedup" and view == "removed":
            resized_dir = resolve(cfg["paths"]["resized"])
            img_dir = resized_dir / "images" if (resized_dir / "images").is_dir() else resized_dir
        else:
            img_dir = images_dir(step_id, cfg)

        src_path = img_dir / filename
        if not src_path.is_file():
            return "Not found", 404

        # Disk cache: store thumbnails in .thumbnails/ beside the images dir
        cache_dir = img_dir / ".thumbnails"
        fmt = "JPEG" if src_path.suffix.lower() in (".jpg", ".jpeg") else "PNG"
        ext = ".jpg" if fmt == "JPEG" else ".png"
        cache_path = cache_dir / (Path(filename).stem + ext)

        if cache_path.is_file() and cache_path.stat().st_mtime >= src_path.stat().st_mtime:
            mimetype = "image/jpeg" if fmt == "JPEG" else "image/png"
            return send_from_directory(cache_dir, cache_path.name,
                                       mimetype=mimetype,
                                       max_age=86400)

        try:
            with Image.open(src_path) as img:
                img.thumbnail((200, 200))
                cache_dir.mkdir(parents=True, exist_ok=True)
                img.save(cache_path, format=fmt, quality=80)
            mimetype = "image/jpeg" if fmt == "JPEG" else "image/png"
            return send_from_directory(cache_dir, cache_path.name,
                                       mimetype=mimetype,
                                       max_age=86400)
        except Exception:
            return send_from_directory(img_dir, filename)

    # ─── API routes ───────────────────────────────────────────────────────

    @app.route("/api/images/<step_id>")
    def api_images(step_id):
        step = get_step(step_id)
        if not step:
            return jsonify({"error": "Step not found"}), 404

        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 60, type=int)
        search = request.args.get("search", "", type=str).strip().lower()
        reason_filter = request.args.get("reason", "", type=str).strip().lower()
        view = request.args.get("view", "kept", type=str).strip().lower()
        decision_band = request.args.get("decision_band", "", type=str).strip().lower()
        failure_mode = request.args.get("failure_mode", "", type=str).strip().lower()
        nearest_boundary = request.args.get("nearest_boundary", "", type=str).strip().lower()

        # Label filters
        filters = {}
        for cat in LABEL_CATEGORIES:
            val = request.args.get(cat, "", type=str).strip()
            if val:
                filters[cat] = val

        # Filter step: rejected view
        if step_id == "filter" and view == "rejected":
            images = list_rejected_images(cfg)
            soil_map = build_soil_score_map(cfg)
            overlay_map = build_overlay_score_map(cfg)
            thresholds = get_filter_thresholds(cfg)
            metrics_cache: dict[str, dict] = {}

            def metrics_for(fn: str) -> dict:
                if fn not in metrics_cache:
                    metrics_cache[fn] = compute_filter_metrics(
                        fn, soil_map, overlay_map, thresholds
                    )
                return metrics_cache[fn]

            source_filter = request.args.get("source", "", type=str).strip().lower()
            if source_filter:
                images = [img for img in images if extract_source(img) == source_filter]

            if reason_filter:
                images = [
                    fn for fn in images
                    if get_rejection_reason(fn, soil_map, overlay_map)["reason"] == reason_filter
                ]

            if failure_mode:
                images = [
                    fn for fn in images
                    if metrics_for(fn).get("rejection_mode") == failure_mode
                ]

            if decision_band:
                images = [
                    fn for fn in images
                    if metrics_for(fn).get("decision_band") == decision_band
                ]

            if search:
                images = [img for img in images if search in img.lower()]

            total = len(images)
            start = (page - 1) * per_page
            page_images = images[start : start + per_page]

            results = []
            for img in page_images:
                reason_info = get_rejection_reason(img, soil_map, overlay_map)
                metrics = metrics_for(img)
                entry = {
                    "filename": img,
                    "url": f"/images/filter/{img}?view=rejected",
                    "thumb_url": f"/thumbnails/filter/{img}?view=rejected",
                    "rejection": reason_info,
                    "source": extract_source(img),
                    "filter_metrics": metrics,
                }
                results.append(entry)

            return jsonify({
                "images": results,
                "total": total,
                "page": page,
                "per_page": per_page,
                "pages": max(1, (total + per_page - 1) // per_page),
                "thresholds": thresholds,
            })

        # Dedup step: removed view
        if step_id == "dedup" and view == "removed":
            images = list_dedup_removed_images(cfg)

            source_filter = request.args.get("source", "", type=str).strip().lower()
            if source_filter:
                images = [img for img in images if extract_source(img) == source_filter]

            if search:
                images = [img for img in images if search in img.lower()]

            total = len(images)
            start = (page - 1) * per_page
            page_images = images[start : start + per_page]

            results = []
            for img in page_images:
                entry = {
                    "filename": img,
                    "url": f"/images/dedup/{img}?view=removed",
                    "thumb_url": f"/thumbnails/dedup/{img}?view=removed",
                    "removed": True,
                    "source": extract_source(img),
                }
                results.append(entry)

            return jsonify({
                "images": results,
                "total": total,
                "page": page,
                "per_page": per_page,
                "pages": max(1, (total + per_page - 1) // per_page),
            })

        images = list_images(step_id, cfg)
        labels = load_labels(step_id, cfg) if step["has_labels"] else {}

        # Download-specific filters: source and query
        if step_id == "download":
            source_filter = request.args.get("source", "", type=str).strip().lower()
            query_filter = request.args.get("query", "", type=str).strip()
            if source_filter:
                images = [img for img in images if img.split("/")[0].lower() == source_filter]
            if query_filter:
                images = [img for img in images if len(img.split("/")) >= 3 and img.split("/")[1] == query_filter]

        # Source filter for filter kept and dedup kept
        if step_id in ("filter", "dedup"):
            source_filter = request.args.get("source", "", type=str).strip().lower()
            if source_filter:
                images = [img for img in images if extract_source(img) == source_filter]

        # Filter kept view: overlay status filter
        if step_id == "filter" and view == "kept":
            wm_filter = request.args.get("watermark", "", type=str).strip().lower()
            soil_map = build_soil_score_map(cfg)
            wm_map = build_overlay_score_map(cfg)
            thresholds = get_filter_thresholds(cfg)
            metrics_cache: dict[str, dict] = {}

            def metrics_for(fn: str) -> dict:
                if fn not in metrics_cache:
                    metrics_cache[fn] = compute_filter_metrics(
                        fn, soil_map, wm_map, thresholds
                    )
                return metrics_cache[fn]

            if wm_filter:
                if wm_filter == "flagged":
                    images = [img for img in images if wm_map.get(img, {}).get("flagged", "").lower() == "true"]
                elif wm_filter == "clean":
                    images = [img for img in images if wm_map.get(img, {}).get("flagged", "").lower() != "true"]

            if nearest_boundary:
                images = [
                    img for img in images
                    if metrics_for(img).get("nearest_boundary") == nearest_boundary
                ]

            if decision_band:
                images = [
                    img for img in images
                    if metrics_for(img).get("decision_band") == decision_band
                ]

        if search:
            images = [img for img in images if search in img.lower()]

        if filters and labels:
            images = [
                img for img in images
                if all(labels.get(img, {}).get(c, "") == v for c, v in filters.items())
            ]

        total = len(images)
        start = (page - 1) * per_page
        page_images = images[start : start + per_page]

        results = []
        for img in page_images:
            entry = {
                "filename": img,
                "url": f"/images/{step_id}/{img}",
                "thumb_url": f"/thumbnails/{step_id}/{img}",
            }
            # Add source for filter, dedup, label steps
            if step_id in ("filter", "dedup", "label"):
                entry["source"] = extract_source(img)
            if step_id == "filter":
                if "metrics_for" in locals():
                    entry["filter_metrics"] = metrics_for(img)
            if labels:
                lbl = labels.get(img, {})
                entry["labels"] = {cat: lbl.get(cat, "") for cat in LABEL_CATEGORIES}
            results.append(entry)

        return jsonify({
            "images": results,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": max(1, (total + per_page - 1) // per_page),
        })

    @app.route("/api/image-detail/<step_id>/<path:filename>")
    def api_image_detail(step_id, filename):
        step = get_step(step_id)
        if not step:
            return jsonify({"error": "Step not found"}), 404

        view = request.args.get("view", "kept", type=str)
        result: dict = {"filename": filename, "url": f"/images/{step_id}/{filename}"}

        # Add source for filter/dedup/label
        if step_id in ("filter", "dedup", "label"):
            result["source"] = extract_source(filename)

        # Resize step: include before/after dimensions and original URL
        if step_id == "resize":
            resized_dir = images_dir("resize", cfg)
            dims = get_image_dimensions(resized_dir / filename)
            if dims:
                result["dimensions"] = dims
            hash_map = build_resize_hash_map(cfg)
            original = find_raw_original(filename, hash_map, cfg)
            if original:
                raw_dir = resolve(cfg["paths"]["raw"])
                rel = str(original.relative_to(raw_dir)).replace("\\", "/")
                result["original_url"] = f"/images/raw/{rel}"
                orig_dims = get_image_dimensions(original)
                if orig_dims:
                    result["original_dimensions"] = orig_dims
            return jsonify(result)

        # For dedup removed view
        if step_id == "dedup" and view == "removed":
            result["url"] = f"/images/dedup/{filename}?view=removed"
            result["removed"] = True
            result["source"] = extract_source(filename)
            # Find which kept image this was a duplicate of
            groups = load_dedup_groups(cfg)
            for kept, removed_list in groups.items():
                if filename in removed_list:
                    result["duplicate_of"] = kept
                    result["duplicate_of_url"] = f"/images/dedup/{kept}"
                    siblings = [r for r in removed_list if r != filename]
                    result["duplicate_siblings"] = [
                        {"filename": s, "url": f"/images/dedup/{s}?view=removed", "source": extract_source(s)}
                        for s in siblings
                    ]
                    break
            return jsonify(result)

        # For dedup kept view, include removed duplicates
        if step_id == "dedup" and view != "removed":
            groups = load_dedup_groups(cfg)
            removed_list = groups.get(filename, [])
            if removed_list:
                result["duplicates"] = [
                    {"filename": r, "url": f"/images/dedup/{r}?view=removed"}
                    for r in removed_list
                ]

        if step["has_labels"]:
            labels = load_labels(step_id, cfg)
            full_labels = load_full_labels(step_id, cfg)

            entry = labels.get(filename, {})
            result["labels"] = {cat: entry.get(cat, "") for cat in LABEL_CATEGORIES}

            full_entry = full_labels.get(filename, {})
            if "scores" in full_entry:
                result["scores"] = full_entry["scores"]

        if step_id == "filter":
            score_map = build_soil_score_map(cfg)
            overlay_map = build_overlay_score_map(cfg)
            thresholds = get_filter_thresholds(cfg)
            result["thresholds"] = thresholds
            result["filter_metrics"] = compute_filter_metrics(
                filename, score_map, overlay_map, thresholds
            )
            if filename in score_map:
                result["soil_scores"] = score_map[filename]
            if filename in overlay_map:
                result["overlay_scores"] = overlay_map[filename]
            if view == "rejected":
                result["rejection"] = get_rejection_reason(
                    filename, score_map, overlay_map
                )
                result["url"] = f"/images/filter/{filename}?view=rejected"

        return jsonify(result)

    # ─── Evaluation results route ─────────────────────────────────────────

    @app.route("/evaluation")
    def evaluation_page():
        """Evaluation results dashboard."""
        eval_dir = resolve(cfg["paths"]["dataset"]).parent / "evaluation"
        metrics_path = eval_dir / "metrics.json"
        if not metrics_path.exists():
            return render_template(
                "evaluation.html",
                steps=STEPS,
                categories=LABEL_CATEGORIES,
                metrics=None,
            )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return render_template(
            "evaluation.html",
            steps=STEPS,
            categories=LABEL_CATEGORIES,
            metrics=metrics,
        )

    @app.route("/api/eval/metrics")
    def api_eval_metrics():
        """Return evaluation metrics JSON."""
        eval_dir = resolve(cfg["paths"]["dataset"]).parent / "evaluation"
        metrics_path = eval_dir / "metrics.json"
        if not metrics_path.exists():
            return jsonify({"error": "No evaluation metrics found. Run: soil-shazam-data-collector eval-report"}), 404
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return jsonify(metrics)

    # ─── Evaluation / annotation routes ───────────────────────────────────

    @app.route("/annotate")
    def annotate_page():
        """Annotation UI for evaluation sampling."""
        return render_template(
            "annotate.html",
            categories=LABEL_CATEGORIES,
            label_options_all=label_options_all,
            steps=STEPS,
        )

    @app.route("/api/eval/sample")
    def api_eval_sample():
        """Return the evaluation sample with server-side filtering & pagination."""
        eval_dir = resolve(cfg["paths"]["dataset"]).parent / "evaluation"
        sample_path = eval_dir / "sample.json"
        if not sample_path.exists():
            return jsonify({
                "error": "No evaluation sample found. Run: soil-shazam-data-collector eval-sample"
            }), 404
        samples = json.loads(sample_path.read_text(encoding="utf-8"))

        # ── Filtering ──────────────────────────────────────────────────────
        soil_val = request.args.get("soil", "")
        category_val = request.args.get("category", "")
        correct_val = request.args.get("correct", "")

        filtered = []
        for s in samples:
            # soil filter
            if soil_val == "soil" and s.get("is_soil") is False:
                continue
            if soil_val == "not_soil" and s.get("is_soil") is not False:
                continue

            gt = s.get("ground_truth") or {}
            pr = s.get("predicted") or {}

            # category filter
            if category_val and gt:
                if category_val not in gt:
                    continue

            # correct / incorrect filter
            if correct_val and gt and pr:
                if category_val:
                    is_correct = pr.get(category_val) == gt.get(category_val)
                    if correct_val == "correct" and not is_correct:
                        continue
                    if correct_val == "incorrect" and is_correct:
                        continue
                else:
                    has_any = False
                    all_correct = True
                    for k in gt:
                        if k in pr:
                            has_any = True
                            if pr[k] != gt[k]:
                                all_correct = False
                    if correct_val == "correct" and (not has_any or not all_correct):
                        continue
                    if correct_val == "incorrect" and all_correct:
                        continue

            filtered.append(s)

        # ── Pagination ─────────────────────────────────────────────────────
        # ?all=1 bypasses pagination (used by the annotation UI which needs the full set)
        if request.args.get("all") == "1":
            return jsonify({
                "samples": filtered,
                "total": len(filtered),
                "page": 1,
                "per_page": len(filtered),
                "total_pages": 1,
            })

        page = max(1, request.args.get("page", 1, type=int))
        per_page = min(100, max(1, request.args.get("per_page", 24, type=int)))
        total = len(filtered)
        total_pages = max(1, -(-total // per_page))  # ceil division
        page = min(page, total_pages)
        start = (page - 1) * per_page
        page_items = filtered[start:start + per_page]

        return jsonify({
            "samples": page_items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
        })

    @app.route("/api/eval/annotate", methods=["POST"])
    def api_eval_annotate():
        """Save a single annotation for an evaluation sample."""
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing image field"}), 400

        image = str(data["image"])
        is_soil = data.get("is_soil")

        # Collect explicitly provided labels (validated against prompt options)
        provided_labels: dict[str, str] = {}
        for cat in LABEL_CATEGORIES:
            val = data.get(cat)
            if isinstance(val, str) and val.strip():
                if val.strip() in label_options_all.get(cat, []):
                    provided_labels[cat] = val.strip()

        # Load sample, update annotation, save
        eval_dir = resolve(cfg["paths"]["dataset"]).parent / "evaluation"
        sample_path = eval_dir / "sample.json"
        if not sample_path.exists():
            return jsonify({"error": "No evaluation sample"}), 404

        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        found = False
        for s in samples:
            if s["image"] == image:
                s["is_soil"] = is_soil

                if is_soil is True:
                    # Start from predicted labels when valid, then override with user input.
                    predicted = s.get("predicted") if isinstance(s.get("predicted"), dict) else {}
                    resolved_ground_truth: dict[str, str] = {}
                    for cat in LABEL_CATEGORIES:
                        pred_val = str(predicted.get(cat, "")).strip()
                        if pred_val in label_options_all.get(cat, []):
                            resolved_ground_truth[cat] = pred_val
                        if cat in provided_labels:
                            resolved_ground_truth[cat] = provided_labels[cat]

                    missing = [cat for cat in LABEL_CATEGORIES if not resolved_ground_truth.get(cat)]
                    if missing:
                        return jsonify({
                            "error": (
                                "Missing class labels for soil image: "
                                + ", ".join(missing)
                            )
                        }), 400
                    s["ground_truth"] = resolved_ground_truth
                else:
                    s.pop("ground_truth", None)
                found = True
                break

        if not found:
            return jsonify({"error": f"Image {image} not in sample"}), 404

        sample_path.write_text(
            json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        annotated = sum(1 for s in samples if s["is_soil"] is not None)
        return jsonify({"status": "saved", "image": image, "progress": f"{annotated}/{len(samples)}"})

    @app.route("/images/eval/<path:filename>")
    def serve_eval_image(filename):
        """Serve images for the evaluation annotation UI."""
        # Try dataset images first, then deduped (for rejected samples)
        dataset_img_dir = resolve(cfg["paths"]["dataset"]) / "images"
        if (dataset_img_dir / filename).is_file():
            return send_from_directory(dataset_img_dir, filename)

        deduped_img_dir = resolve(cfg["paths"]["deduped"]) / "images"
        if (deduped_img_dir / filename).is_file():
            return send_from_directory(deduped_img_dir, filename)

        return "Image not found", 404

    return app

