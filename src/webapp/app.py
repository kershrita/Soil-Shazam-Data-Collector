"""Flask validation website for the soil data collection pipeline."""

from __future__ import annotations

import csv
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
    extract_source,
    find_raw_original,
    get_image_dimensions,
    get_rejection_reason,
    get_step,
    images_dir,
    list_dedup_removed_images,
    list_images,
    list_rejected_images,
    load_corrections,
    load_dedup_groups,
    load_full_labels,
    load_labels,
    load_yaml,
    resolve,
    save_corrections,
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
        return render_template(
            "dashboard.html",
            steps=step_stats,
            categories=LABEL_CATEGORIES,
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

        return render_template(
            "browser.html",
            step=step,
            stats=stats,
            view=view,
            categories=LABEL_CATEGORIES,
            label_options=label_options,
            label_options_all=label_options_all,
            steps=STEPS,
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
        """Serve a downscaled thumbnail (200px max) for grid display."""
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

        # Generate thumbnail in memory
        try:
            with Image.open(src_path) as img:
                img.thumbnail((200, 200))
                buf = io.BytesIO()
                fmt = "JPEG" if src_path.suffix.lower() in (".jpg", ".jpeg") else "PNG"
                img.save(buf, format=fmt, quality=80)
                buf.seek(0)
                mimetype = "image/jpeg" if fmt == "JPEG" else "image/png"
                return Response(buf.getvalue(), mimetype=mimetype,
                                headers={"Cache-Control": "public, max-age=86400"})
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

            source_filter = request.args.get("source", "", type=str).strip().lower()
            if source_filter:
                images = [img for img in images if extract_source(img) == source_filter]

            if reason_filter:
                images = [
                    fn for fn in images
                    if get_rejection_reason(fn, soil_map, overlay_map)["reason"] == reason_filter
                ]

            if search:
                images = [img for img in images if search in img.lower()]

            total = len(images)
            start = (page - 1) * per_page
            page_images = images[start : start + per_page]

            results = []
            for img in page_images:
                reason_info = get_rejection_reason(img, soil_map, overlay_map)
                entry = {
                    "filename": img,
                    "url": f"/images/filter/{img}?view=rejected",
                    "thumb_url": f"/thumbnails/filter/{img}?view=rejected",
                    "rejection": reason_info,
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
            if wm_filter:
                wm_map = build_overlay_score_map(cfg)
                if wm_filter == "flagged":
                    images = [img for img in images if wm_map.get(img, {}).get("flagged", "").lower() == "true"]
                elif wm_filter == "clean":
                    images = [img for img in images if wm_map.get(img, {}).get("flagged", "").lower() != "true"]

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

        corrections_map = {c["image"]: c for c in load_corrections(cfg)}

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
            if labels:
                lbl = labels.get(img, {})
                entry["labels"] = {cat: lbl.get(cat, "") for cat in LABEL_CATEGORIES}
                if img in corrections_map:
                    entry["corrected"] = True
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

            corrections = load_corrections(cfg)
            for c in corrections:
                if c["image"] == filename:
                    result["corrected"] = not c.get("_correct", True)
                    if result["corrected"]:
                        result["corrections"] = {
                            cat: c.get(cat, "") for cat in LABEL_CATEGORIES
                        }
                    break

        if step_id == "filter":
            score_map = build_soil_score_map(cfg)
            overlay_map = build_overlay_score_map(cfg)
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

    # ─── Correction routes ────────────────────────────────────────────────

    @app.route("/api/corrections", methods=["GET"])
    def get_corrections():
        return jsonify(load_corrections(cfg))

    @app.route("/api/corrections", methods=["POST"])
    def save_correction():
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing image field"}), 400

        image = str(data["image"])
        new_labels = {}
        for cat in LABEL_CATEGORIES:
            val = data.get(cat)
            if isinstance(val, str) and val.strip():
                # Validate against allowed options
                if val.strip() in label_options_all.get(cat, []):
                    new_labels[cat] = val.strip()

        if not new_labels:
            return jsonify({"error": "No valid label changes provided"}), 400

        corrections = load_corrections(cfg)

        # Update existing or create new correction entry
        found = False
        for c in corrections:
            if c["image"] == image:
                c.update(new_labels)
                c["_correct"] = False
                found = True
                break

        if not found:
            # Fill unchanged fields from the latest labels
            labels = load_labels("label", cfg)
            original = labels.get(image, {})
            correction = {"image": image, "_correct": False}
            for cat in LABEL_CATEGORIES:
                correction[cat] = new_labels.get(cat, original.get(cat, ""))
            corrections.append(correction)

        save_corrections(cfg, corrections)
        return jsonify({"status": "saved", "image": image})

    @app.route("/api/corrections/<path:image>", methods=["DELETE"])
    def delete_correction(image):
        corrections = load_corrections(cfg)
        corrections = [c for c in corrections if c["image"] != image]
        save_corrections(cfg, corrections)
        return jsonify({"status": "deleted", "image": image})

    # ─── Batch corrections ────────────────────────────────────────────────

    @app.route("/api/corrections/batch", methods=["POST"])
    def batch_corrections():
        """Apply the same label corrections to multiple images at once."""
        data = request.get_json()
        if not data or "images" not in data or not isinstance(data["images"], list):
            return jsonify({"error": "Missing images array"}), 400

        images = [str(img) for img in data["images"] if isinstance(img, str)]
        if not images:
            return jsonify({"error": "No valid image names"}), 400

        new_labels = {}
        for cat in LABEL_CATEGORIES:
            val = data.get(cat)
            if isinstance(val, str) and val.strip():
                if val.strip() in label_options_all.get(cat, []):
                    new_labels[cat] = val.strip()

        if not new_labels:
            return jsonify({"error": "No valid label changes provided"}), 400

        corrections = load_corrections(cfg)
        corrections_map = {c["image"]: c for c in corrections}
        labels = load_labels("label", cfg)

        for image in images:
            if image in corrections_map:
                corrections_map[image].update(new_labels)
                corrections_map[image]["_correct"] = False
            else:
                original = labels.get(image, {})
                correction = {"image": image, "_correct": False}
                for cat in LABEL_CATEGORIES:
                    correction[cat] = new_labels.get(cat, original.get(cat, ""))
                corrections_map[image] = correction

        save_corrections(cfg, list(corrections_map.values()))
        return jsonify({"status": "saved", "count": len(images)})

    # ─── Export corrections as CSV ────────────────────────────────────────

    @app.route("/api/corrections/export")
    def export_corrections_csv():
        """Download all corrections as a CSV file."""
        corrections = load_corrections(cfg)
        output = io.StringIO()
        fieldnames = ["image", "_correct"] + list(LABEL_CATEGORIES)
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for c in corrections:
            writer.writerow(c)
        csv_data = output.getvalue()
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=corrections.csv"},
        )

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
        """Return the evaluation sample for annotation."""
        eval_dir = resolve(cfg["paths"]["dataset"]).parent / "evaluation"
        sample_path = eval_dir / "sample.json"
        if not sample_path.exists():
            return jsonify({"error": "No evaluation sample found. Run: soil-collector eval-sample"}), 404
        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        return jsonify({"samples": samples, "total": len(samples)})

    @app.route("/api/eval/annotate", methods=["POST"])
    def api_eval_annotate():
        """Save a single annotation for an evaluation sample."""
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing image field"}), 400

        image = str(data["image"])
        is_soil = data.get("is_soil")

        # Build ground truth labels (only for accepted soil images)
        ground_truth = {}
        for cat in LABEL_CATEGORIES:
            val = data.get(cat)
            if isinstance(val, str) and val.strip():
                if val.strip() in label_options_all.get(cat, []):
                    ground_truth[cat] = val.strip()

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
                if ground_truth:
                    s["ground_truth"] = ground_truth
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
