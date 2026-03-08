"""Flask validation website for the soil data collection pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

# ─── Project paths ────────────────────────────────────────────────────────────

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent.parent  # src/webapp → project root

STEPS = [
    {"id": "download", "name": "Download", "desc": "Raw images from web sources",   "path_key": "raw",      "has_labels": False, "icon": "1"},
    {"id": "resize",   "name": "Resize",   "desc": "Resolution filtered & resized", "path_key": "resized",  "has_labels": False, "icon": "2"},
    {"id": "dedup",    "name": "Dedup",    "desc": "Perceptual hash deduplication",  "path_key": "deduped",  "has_labels": False, "icon": "3"},
    {"id": "filter",   "name": "Filter",   "desc": "Overlay + soil CLIP filter",    "path_key": "filtered", "has_labels": False, "icon": "4"},
    {"id": "label",    "name": "Label",    "desc": "CLIP multi-category labeling",  "path_key": "labeled",  "has_labels": True,  "icon": "5"},
]

LABEL_CATEGORIES = [
    "soil_color", "soil_texture", "particle_size",
    "crack_presence", "rock_fraction", "surface_structure", "surface_roughness",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve(cfg_path: str) -> Path:
    p = Path(cfg_path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _get_step(step_id: str) -> dict | None:
    return next((s for s in STEPS if s["id"] == step_id), None)


def _step_base_dir(step_id: str, cfg: dict) -> Path:
    step = _get_step(step_id)
    return _resolve(cfg["paths"][step["path_key"]])


def _images_dir(step_id: str, cfg: dict) -> Path:
    base = _step_base_dir(step_id, cfg)
    sub = base / "images"
    return sub if sub.is_dir() else base


def _list_images(step_id: str, cfg: dict) -> list[str]:
    """Return sorted list of image filenames for a step."""
    img_dir = _images_dir(step_id, cfg)
    if not img_dir.exists():
        return []

    if step_id == "download":
        # Recursive walk for nested source/query directories
        return sorted(
            str(p.relative_to(img_dir)).replace("\\", "/")
            for p in img_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
    return sorted(
        p.name for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _load_labels(step_id: str, cfg: dict) -> dict[str, dict]:
    """Load labels.json for a labeled step → {filename: entry}."""
    base = _step_base_dir(step_id, cfg)
    labels_path = base / "labels.json"
    if not labels_path.exists():
        return {}
    entries = json.loads(labels_path.read_text(encoding="utf-8"))
    return {e["image"]: e for e in entries}


def _load_full_labels(step_id: str, cfg: dict) -> dict[str, dict]:
    """Load labels_full.json (with scores) if available."""
    base = _step_base_dir(step_id, cfg)
    path = base / "labels_full.json"
    if not path.exists():
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {e["image"]: e for e in entries}


def _get_image_dimensions(path: Path) -> dict | None:
    """Return {width, height} for an image file."""
    try:
        with Image.open(path) as img:
            w, h = img.size
        return {"width": w, "height": h}
    except Exception:
        return None


def _build_resize_hash_map(cfg: dict) -> dict[str, str]:
    """Build {hash_prefix: relative_parent_path} from raw download dirs."""
    raw_dir = _resolve(cfg["paths"]["raw"])
    result = {}
    if not raw_dir.exists():
        return result
    for source_dir in raw_dir.iterdir():
        if not source_dir.is_dir():
            continue
        for query_dir in source_dir.iterdir():
            if not query_dir.is_dir():
                continue
            rel = str(query_dir.relative_to(raw_dir))
            h = hashlib.md5(rel.encode()).hexdigest()[:8]
            result[h] = rel
    return result


def _find_raw_original(resized_name: str, hash_map: dict, cfg: dict) -> Path | None:
    """Given a resized filename, find the original raw image path."""
    stem = Path(resized_name).stem  # e.g. 0f428dd7_Image_1
    parts = stem.split("_", 1)
    if len(parts) != 2:
        return None
    prefix, orig_stem = parts
    rel_parent = hash_map.get(prefix)
    if not rel_parent:
        return None
    raw_dir = _resolve(cfg["paths"]["raw"]) / rel_parent
    if not raw_dir.exists():
        return None
    for ext in IMAGE_EXTS:
        candidate = raw_dir / f"{orig_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_filter_log(cfg: dict, log_name: str) -> list[dict]:
    logs_dir = _resolve(cfg["paths"]["logs"])
    path = logs_dir / log_name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _corrections_path(cfg: dict) -> Path:
    dataset_dir = _resolve(cfg["paths"]["dataset"])
    return dataset_dir / "verification" / "corrections.json"


def _load_corrections(cfg: dict) -> list[dict]:
    path = _corrections_path(cfg)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _save_corrections(cfg: dict, corrections: list[dict]):
    path = _corrections_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(corrections, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _build_soil_score_map(cfg: dict) -> dict[str, dict]:
    """Build {filename: {positive, negative, kept}} from soil_filter.csv."""
    rows = _load_filter_log(cfg, "soil_filter.csv")
    result = {}
    for r in rows:
        fn = r.get("filename", "").strip()
        if fn:
            result[fn] = {
                "positive": r.get("positive_score", "").strip(),
                "negative": r.get("negative_score", "").strip(),
                "kept": r.get("kept", "").strip(),
            }
    return result


def _build_overlay_score_map(cfg: dict) -> dict[str, dict]:
    """Build {filename: {overlay_score, clean_score, flagged}} from overlay_filter.csv."""
    rows = _load_filter_log(cfg, "overlay_filter.csv")
    result = {}
    for r in rows:
        fn = r.get("filename", "").strip()
        if fn:
            result[fn] = {
                "overlay_score": r.get("overlay_score", "").strip(),
                "clean_score": r.get("clean_score", "").strip(),
                "flagged": r.get("flagged", "").strip(),
            }
    return result


def _list_rejected_images(cfg: dict) -> list[str]:
    """Return filenames from resized/ that were rejected by the filter step."""
    soil_map = _build_soil_score_map(cfg)
    rejected = [
        fn for fn, info in soil_map.items()
        if info["kept"].lower() != "true"
    ]
    return sorted(rejected)


def _list_dedup_removed_images(cfg: dict) -> list[str]:
    """Return filenames removed during dedup (in resized/ but not in deduped/)."""
    resized_dir = _resolve(cfg["paths"]["resized"])
    deduped_dir = _resolve(cfg["paths"]["deduped"])
    resized_imgs = resized_dir / "images" if (resized_dir / "images").is_dir() else resized_dir
    deduped_imgs = deduped_dir / "images" if (deduped_dir / "images").is_dir() else deduped_dir
    if not resized_imgs.exists():
        return []
    resized_set = {
        p.name for p in resized_imgs.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }
    deduped_set = {
        p.name for p in deduped_imgs.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    } if deduped_imgs.exists() else set()
    return sorted(resized_set - deduped_set)


def _load_dedup_groups(cfg: dict) -> dict[str, list[str]]:
    """Load dedup_groups.json → {kept_filename: [removed_filenames]}."""
    deduped_dir = _resolve(cfg["paths"]["deduped"])
    groups_path = deduped_dir / "dedup_groups.json"
    if not groups_path.exists():
        return {}
    try:
        return json.loads(groups_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _extract_source(fn: str) -> str:
    """Extract download source from filename pattern: hash_source_..."""
    parts = fn.split("_")
    if len(parts) >= 3:
        src = parts[1]
        return "bing" if src == "Image" else src
    return "unknown"



def _get_rejection_reason(filename: str, soil_map: dict, overlay_map: dict) -> dict:
    """Get rejection reason and scores for a specific image."""
    result = {"reason": "unknown"}
    soil = soil_map.get(filename, {})
    ov = overlay_map.get(filename, {})

    if soil.get("kept", "").lower() == "overlay":
        result["reason"] = "overlay"
        result["overlay_score"] = ov.get("overlay_score", "")
        result["clean_score"] = ov.get("clean_score", "")
    else:
        result["reason"] = "low_soil_score"
        result["positive_score"] = soil.get("positive", "")
        result["negative_score"] = soil.get("negative", "")

    # Always include all score sets if available
    if ov:
        result["overlay_score"] = ov.get("overlay_score", "")
        result["clean_score"] = ov.get("clean_score", "")
        result["flagged"] = ov.get("flagged", "")
    if soil:
        result["positive_score"] = soil.get("positive", "")
        result["negative_score"] = soil.get("negative", "")
        result["kept"] = soil.get("kept", "")

    return result


def _compute_step_stats(step_id: str, cfg: dict) -> dict:
    """Compute statistics for a pipeline step."""
    images = _list_images(step_id, cfg)
    stats: dict = {"count": len(images)}
    step = _get_step(step_id)

    if step_id == "download":
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
        soil_log = _load_filter_log(cfg, "soil_filter.csv")
        overlay_log = _load_filter_log(cfg, "overlay_filter.csv")
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
        soil_map = _build_soil_score_map(cfg)
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
            _extract_source(fn) for fn, info in soil_map.items()
            if info["kept"].lower() == "true"
        )
        rejected_src: Counter = Counter(
            _extract_source(fn) for fn, info in soil_map.items()
            if info["kept"].lower() != "true"
        )
        stats["distributions"] = {
            "kept": dict(kept_src.most_common()),
            "rejected": dict(rejected_src.most_common()),
        }
        all_filter_sources = sorted(set(kept_src) | set(rejected_src))
        stats["source_list"] = all_filter_sources

    elif step_id == "dedup":
        removed_names = _list_dedup_removed_images(cfg)
        stats["removed_count"] = len(removed_names)

        kept_imgs = _list_images(step_id, cfg)
        source_kept = Counter(_extract_source(fn) for fn in kept_imgs)
        source_removed = Counter(_extract_source(fn) for fn in removed_names)
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
        labels = _load_labels(step_id, cfg)
        if labels:
            distributions = {}
            for cat in LABEL_CATEGORIES:
                dist = Counter(e.get(cat, "unknown") for e in labels.values())
                distributions[cat] = dict(dist.most_common())
            stats["distributions"] = distributions

    return stats


# ─── App factory ──────────────────────────────────────────────────────────────

def create_app(config_dir: Path | None = None) -> Flask:
    config_dir = config_dir or _PROJECT_ROOT / "config"
    cfg = _load_yaml(config_dir / "pipeline.yaml")
    prompts = _load_yaml(config_dir / "prompts.yaml")

    # Extract valid label options from prompts
    label_options_all: dict[str, list[str]] = {}
    for cat in LABEL_CATEGORIES:
        label_options_all[cat] = list(prompts.get("labeling", {}).get(cat, {}).keys())

    app = Flask(
        __name__,
        template_folder=str(_APP_DIR / "templates"),
        static_folder=str(_APP_DIR / "static"),
    )

    # Cache resize hash map
    _resize_hash_map: dict[str, str] | None = None

    def _get_resize_hash_map() -> dict[str, str]:
        nonlocal _resize_hash_map
        if _resize_hash_map is None:
            _resize_hash_map = _build_resize_hash_map(cfg)
        return _resize_hash_map

    # ─── Page routes ──────────────────────────────────────────────────────

    @app.route("/")
    def dashboard():
        step_stats = []
        for step in STEPS:
            stats = _compute_step_stats(step["id"], cfg)
            step_stats.append({**step, "stats": stats})
        return render_template(
            "dashboard.html",
            steps=step_stats,
            categories=LABEL_CATEGORIES,
        )

    @app.route("/step/<step_id>")
    def step_browser(step_id):
        step = _get_step(step_id)
        if not step:
            return "Step not found", 404

        view = request.args.get("view", "kept", type=str)
        stats = _compute_step_stats(step_id, cfg)

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
            deduped_dir = _resolve(cfg["paths"]["deduped"])
            img_dir = deduped_dir / "images" if (deduped_dir / "images").is_dir() else deduped_dir
        elif step_id == "dedup" and view == "removed":
            resized_dir = _resolve(cfg["paths"]["resized"])
            img_dir = resized_dir / "images" if (resized_dir / "images").is_dir() else resized_dir
        else:
            img_dir = _images_dir(step_id, cfg)
        return send_from_directory(img_dir, filename)

    @app.route("/images/raw/<path:filename>")
    def serve_raw_image(filename):
        raw_dir = _resolve(cfg["paths"]["raw"])
        return send_from_directory(raw_dir, filename)

    # ─── API routes ───────────────────────────────────────────────────────

    @app.route("/api/images/<step_id>")
    def api_images(step_id):
        step = _get_step(step_id)
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
            images = _list_rejected_images(cfg)
            soil_map = _build_soil_score_map(cfg)
            overlay_map = _build_overlay_score_map(cfg)

            source_filter = request.args.get("source", "", type=str).strip().lower()
            if source_filter:
                images = [img for img in images if _extract_source(img) == source_filter]

            if reason_filter:
                images = [
                    fn for fn in images
                    if _get_rejection_reason(fn, soil_map, overlay_map)["reason"] == reason_filter
                ]

            if search:
                images = [img for img in images if search in img.lower()]

            total = len(images)
            start = (page - 1) * per_page
            page_images = images[start : start + per_page]

            results = []
            for img in page_images:
                reason_info = _get_rejection_reason(img, soil_map, overlay_map)
                entry = {
                    "filename": img,
                    "url": f"/images/filter/{img}?view=rejected",
                    "rejection": reason_info,
                    "source": _extract_source(img),
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
            images = _list_dedup_removed_images(cfg)

            source_filter = request.args.get("source", "", type=str).strip().lower()
            if source_filter:
                images = [img for img in images if _extract_source(img) == source_filter]

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
                    "removed": True,
                    "source": _extract_source(img),
                }
                results.append(entry)

            return jsonify({
                "images": results,
                "total": total,
                "page": page,
                "per_page": per_page,
                "pages": max(1, (total + per_page - 1) // per_page),
            })

        images = _list_images(step_id, cfg)
        labels = _load_labels(step_id, cfg) if step["has_labels"] else {}

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
                images = [img for img in images if _extract_source(img) == source_filter]

        # Filter kept view: overlay status filter
        if step_id == "filter" and view == "kept":
            wm_filter = request.args.get("watermark", "", type=str).strip().lower()
            if wm_filter:
                wm_map = _build_overlay_score_map(cfg)
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

        corrections_map = {c["image"]: c for c in _load_corrections(cfg)}

        results = []
        for img in page_images:
            entry = {"filename": img, "url": f"/images/{step_id}/{img}"}
            # Add source for filter, dedup, label steps
            if step_id in ("filter", "dedup", "label"):
                entry["source"] = _extract_source(img)
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
        step = _get_step(step_id)
        if not step:
            return jsonify({"error": "Step not found"}), 404

        view = request.args.get("view", "kept", type=str)
        result: dict = {"filename": filename, "url": f"/images/{step_id}/{filename}"}

        # Add source for filter/dedup/label
        if step_id in ("filter", "dedup", "label"):
            result["source"] = _extract_source(filename)

        # Resize step: include before/after dimensions and original URL
        if step_id == "resize":
            resized_dir = _images_dir("resize", cfg)
            dims = _get_image_dimensions(resized_dir / filename)
            if dims:
                result["dimensions"] = dims
            hash_map = _get_resize_hash_map()
            original = _find_raw_original(filename, hash_map, cfg)
            if original:
                raw_dir = _resolve(cfg["paths"]["raw"])
                rel = str(original.relative_to(raw_dir)).replace("\\", "/")
                result["original_url"] = f"/images/raw/{rel}"
                orig_dims = _get_image_dimensions(original)
                if orig_dims:
                    result["original_dimensions"] = orig_dims
            return jsonify(result)

        # For dedup removed view
        if step_id == "dedup" and view == "removed":
            result["url"] = f"/images/dedup/{filename}?view=removed"
            result["removed"] = True
            result["source"] = _extract_source(filename)
            # Find which kept image this was a duplicate of
            groups = _load_dedup_groups(cfg)
            for kept, removed_list in groups.items():
                if filename in removed_list:
                    result["duplicate_of"] = kept
                    result["duplicate_of_url"] = f"/images/dedup/{kept}"
                    siblings = [r for r in removed_list if r != filename]
                    result["duplicate_siblings"] = [
                        {"filename": s, "url": f"/images/dedup/{s}?view=removed", "source": _extract_source(s)}
                        for s in siblings
                    ]
                    break
            return jsonify(result)

        # For dedup kept view, include removed duplicates
        if step_id == "dedup" and view != "removed":
            groups = _load_dedup_groups(cfg)
            removed_list = groups.get(filename, [])
            if removed_list:
                result["duplicates"] = [
                    {"filename": r, "url": f"/images/dedup/{r}?view=removed"}
                    for r in removed_list
                ]

        if step["has_labels"]:
            labels = _load_labels(step_id, cfg)
            full_labels = _load_full_labels(step_id, cfg)

            entry = labels.get(filename, {})
            result["labels"] = {cat: entry.get(cat, "") for cat in LABEL_CATEGORIES}

            full_entry = full_labels.get(filename, {})
            if "scores" in full_entry:
                result["scores"] = full_entry["scores"]

            corrections = _load_corrections(cfg)
            for c in corrections:
                if c["image"] == filename:
                    result["corrected"] = not c.get("_correct", True)
                    if result["corrected"]:
                        result["corrections"] = {
                            cat: c.get(cat, "") for cat in LABEL_CATEGORIES
                        }
                    break

        if step_id == "filter":
            score_map = _build_soil_score_map(cfg)
            overlay_map = _build_overlay_score_map(cfg)
            if filename in score_map:
                result["soil_scores"] = score_map[filename]
            if filename in overlay_map:
                result["overlay_scores"] = overlay_map[filename]
            if view == "rejected":
                result["rejection"] = _get_rejection_reason(
                    filename, score_map, overlay_map
                )
                result["url"] = f"/images/filter/{filename}?view=rejected"

        return jsonify(result)

    @app.route("/api/corrections", methods=["GET"])
    def get_corrections():
        return jsonify(_load_corrections(cfg))

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

        corrections = _load_corrections(cfg)

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
            labels = _load_labels("label", cfg)
            original = labels.get(image, {})
            correction = {"image": image, "_correct": False}
            for cat in LABEL_CATEGORIES:
                correction[cat] = new_labels.get(cat, original.get(cat, ""))
            corrections.append(correction)

        _save_corrections(cfg, corrections)
        return jsonify({"status": "saved", "image": image})

    @app.route("/api/corrections/<path:image>", methods=["DELETE"])
    def delete_correction(image):
        corrections = _load_corrections(cfg)
        corrections = [c for c in corrections if c["image"] != image]
        _save_corrections(cfg, corrections)
        return jsonify({"status": "deleted", "image": image})

    return app
