"""Shared helpers for path resolution, file listing, data loading, and scores."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import yaml
from PIL import Image
from utils import collect_image_paths

# ─── Project paths ────────────────────────────────────────────────────────────

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent.parent  # src/webapp → project root

STEPS = [
    {"id": "download", "name": "Download", "desc": "Raw images from web sources",   "path_key": "raw",      "has_labels": False, "icon": "1"},
    {"id": "resize",   "name": "Resize",   "desc": "Resolution filtered & resized", "path_key": "resized",  "has_labels": False, "icon": "2"},
    {"id": "dedup",    "name": "Dedup",    "desc": "Perceptual hash deduplication",  "path_key": "deduped",  "has_labels": False, "icon": "3"},
    {"id": "filter",   "name": "Filter",   "desc": "Overlay + soil CLIP filter",    "path_key": "filtered", "has_labels": False, "icon": "4"},
    {"id": "label",    "name": "Label",    "desc": "CLIP multi-category labeling",  "path_key": "labeled",  "has_labels": True,  "icon": "5"},
    {"id": "cluster",  "name": "Cluster",  "desc": "Risk-ranked review queues",     "path_key": None,       "has_labels": False, "icon": "6", "url": "/cluster"},
]

LABEL_CATEGORIES = [
    "soil_color", "soil_texture", "particle_size",
    "crack_presence", "rock_fraction", "surface_structure", "surface_roughness",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ─── TTL cache ────────────────────────────────────────────────────────────────

class TTLCache:
    """Simple TTL cache for expensive computations."""

    def __init__(self, ttl: float = 30.0):
        self._ttl = ttl
        self._store: dict[str, tuple[float, object]] = {}

    def get(self, key: str):
        entry = self._store.get(key)
        if entry and (time.monotonic() - entry[0]) < self._ttl:
            return entry[1]
        return None

    def set(self, key: str, value):
        self._store[key] = (time.monotonic(), value)

    def invalidate(self, key: str | None = None):
        if key is None:
            self._store.clear()
        else:
            self._store.pop(key, None)


# Shared cache instance (30s TTL)
cache = TTLCache(ttl=30.0)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve(cfg_path: str) -> Path:
    p = Path(cfg_path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def get_step(step_id: str) -> dict | None:
    return next((s for s in STEPS if s["id"] == step_id), None)


def step_base_dir(step_id: str, cfg: dict) -> Path:
    step = get_step(step_id)
    if not step:
        raise ValueError(f"Unknown step id: {step_id}")
    if not step.get("path_key"):
        raise ValueError(f"Step {step_id} does not map to an image directory")
    return resolve(cfg["paths"][step["path_key"]])


def images_dir(step_id: str, cfg: dict) -> Path:
    base = step_base_dir(step_id, cfg)
    sub = base / "images"
    return sub if sub.is_dir() else base


def list_images(step_id: str, cfg: dict) -> list[str]:
    """Return sorted list of image filenames for a step (cached)."""
    cache_key = f"images:{step_id}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        img_dir = images_dir(step_id, cfg)
    except ValueError:
        return []
    if not img_dir.exists():
        return []

    if step_id == "download":
        result = sorted(
            str(p.relative_to(img_dir)).replace("\\", "/")
            for p in img_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
    else:
        result = sorted(
            p.name for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

    # Single-store fallback: filter kept set is represented in logs even if
    # filtered/images is intentionally empty.
    if step_id == "filter" and not result:
        soil_map = build_soil_score_map(cfg)
        result = sorted(
            fn for fn, info in soil_map.items()
            if str(info.get("kept", "")).strip().lower() == "true"
        )

    # Single-store fallback: label image list can come from labels.json.
    if step_id == "label" and not result:
        try:
            base = step_base_dir(step_id, cfg)
            labels_path = base / "labels.json"
            if labels_path.exists():
                rows = json.loads(labels_path.read_text(encoding="utf-8"))
                result = sorted(
                    str(r.get("image", "")).strip()
                    for r in rows
                    if str(r.get("image", "")).strip()
                )
        except (json.JSONDecodeError, OSError, TypeError):
            result = []

    cache.set(cache_key, result)
    return result


def load_labels(step_id: str, cfg: dict) -> dict[str, dict]:
    """Load labels.json for a labeled step → {filename: entry}."""
    cache_key = f"labels:{step_id}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        base = step_base_dir(step_id, cfg)
    except ValueError:
        return {}
    labels_path = base / "labels.json"
    if not labels_path.exists():
        return {}
    entries = json.loads(labels_path.read_text(encoding="utf-8"))
    result = {e["image"]: e for e in entries}
    cache.set(cache_key, result)
    return result


def load_full_labels(step_id: str, cfg: dict) -> dict[str, dict]:
    """Load labels_full.json (with scores) if available."""
    cache_key = f"full_labels:{step_id}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        base = step_base_dir(step_id, cfg)
    except ValueError:
        return {}
    path = base / "labels_full.json"
    if not path.exists():
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))
    result = {e["image"]: e for e in entries}
    cache.set(cache_key, result)
    return result


def get_image_dimensions(path: Path) -> dict | None:
    """Return {width, height} for an image file."""
    try:
        with Image.open(path) as img:
            w, h = img.size
        return {"width": w, "height": h}
    except Exception:
        return None


def build_resize_hash_map(cfg: dict) -> dict[str, str]:
    """Build {hash_prefix: relative_parent_path} from raw download dirs."""
    cached = cache.get("resize_hash_map")
    if cached is not None:
        return cached

    raw_dir = resolve(cfg["paths"]["raw"])
    result = {}
    if not raw_dir.exists():
        cache.set("resize_hash_map", result)
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
    cache.set("resize_hash_map", result)
    return result


def find_raw_original(resized_name: str, hash_map: dict, cfg: dict) -> Path | None:
    """Given a resized filename, find the original raw image path."""
    stem = Path(resized_name).stem
    parts = stem.split("_", 1)
    if len(parts) != 2:
        return None
    prefix, orig_stem = parts
    rel_parent = hash_map.get(prefix)
    if not rel_parent:
        return None
    raw_dir = resolve(cfg["paths"]["raw"]) / rel_parent
    if not raw_dir.exists():
        return None
    for ext in IMAGE_EXTS:
        candidate = raw_dir / f"{orig_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def clustering_root(cfg: dict) -> Path:
    return resolve(cfg["paths"]["dataset"]).parent / "clustering"


def latest_cluster_run(cfg: dict) -> Path | None:
    root = clustering_root(cfg)
    if not root.exists():
        return None
    runs = [p for p in root.iterdir() if p.is_dir() and p.name != "cache"]
    if not runs:
        return None
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs[0]


def load_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return default


def build_pipeline_precheck(cfg: dict) -> list[dict]:
    """Compute readiness and missing prerequisites for each pipeline command."""
    raw_dir = resolve(cfg["paths"]["raw"])
    resized_dir = resolve(cfg["paths"]["resized"])
    deduped_dir = resolve(cfg["paths"]["deduped"])
    filtered_dir = resolve(cfg["paths"]["filtered"])
    labeled_dir = resolve(cfg["paths"]["labeled"])
    eval_dir = labeled_dir.parent / "evaluation"

    labels_full = labeled_dir / "labels_full.json"
    labels_basic = labeled_dir / "labels.json"
    labels_path = labels_full if labels_full.exists() else labels_basic
    embeddings_path = labeled_dir / "embeddings.npz"
    eval_sample_path = eval_dir / "sample.json"
    eval_metrics_path = eval_dir / "metrics.json"

    filtered_count = len(collect_image_paths(filtered_dir)) if filtered_dir.exists() else 0
    if filtered_count == 0:
        soil_rows = load_filter_log(cfg, "soil_filter.csv")
        filtered_count = sum(
            1 for r in soil_rows
            if str(r.get("kept", "")).strip().lower() == "true"
        )

    counts = {
        "raw": len(collect_image_paths(raw_dir)) if raw_dir.exists() else 0,
        "resized": len(collect_image_paths(resized_dir)) if resized_dir.exists() else 0,
        "deduped": len(collect_image_paths(deduped_dir)) if deduped_dir.exists() else 0,
        "filtered": filtered_count,
    }

    def first_source(candidates: list[tuple[str, str]]) -> tuple[str, int] | None:
        for key, label in candidates:
            count = counts.get(key, 0)
            if count > 0:
                return label, count
        return None

    precheck: list[dict] = [
        {
            "id": "download",
            "title": "download",
            "ready": True,
            "details": "Always ready. This is the root step.",
            "run_first": None,
            "missing": [],
        }
    ]

    if counts["raw"] > 0:
        precheck.append(
            {
                "id": "resize",
                "title": "resize",
                "ready": True,
                "details": f"Ready using raw images ({counts['raw']:,}).",
                "run_first": None,
                "missing": [],
            }
        )
    else:
        precheck.append(
            {
                "id": "resize",
                "title": "resize",
                "ready": False,
                "details": "Blocked: no downloaded images found.",
                "run_first": "soil-shazam-data-collector download",
                "missing": [{"reason": "Missing raw images", "path": str(raw_dir)}],
            }
        )

    dedup_source = first_source([("resized", "resized"), ("raw", "raw")])
    if dedup_source:
        source_label, count = dedup_source
        precheck.append(
            {
                "id": "dedup",
                "title": "dedup",
                "ready": True,
                "details": f"Ready using {source_label} images ({count:,}).",
                "run_first": None,
                "missing": [],
            }
        )
    else:
        precheck.append(
            {
                "id": "dedup",
                "title": "dedup",
                "ready": False,
                "details": "Blocked: no images available for dedup.",
                "run_first": "soil-shazam-data-collector download",
                "missing": [
                    {"reason": "No resized images", "path": str(resized_dir)},
                    {"reason": "No raw images", "path": str(raw_dir)},
                ],
            }
        )

    filter_source = first_source([("deduped", "deduped"), ("resized", "resized"), ("raw", "raw")])
    if filter_source:
        source_label, count = filter_source
        precheck.append(
            {
                "id": "filter",
                "title": "filter",
                "ready": True,
                "details": f"Ready using {source_label} images ({count:,}).",
                "run_first": None,
                "missing": [],
            }
        )
    else:
        precheck.append(
            {
                "id": "filter",
                "title": "filter",
                "ready": False,
                "details": "Blocked: no images available for filter.",
                "run_first": "soil-shazam-data-collector download",
                "missing": [
                    {"reason": "No deduped images", "path": str(deduped_dir)},
                    {"reason": "No resized images", "path": str(resized_dir)},
                    {"reason": "No raw images", "path": str(raw_dir)},
                ],
            }
        )

    label_source = first_source(
        [("filtered", "filtered"), ("deduped", "deduped"), ("resized", "resized"), ("raw", "raw")]
    )
    if label_source:
        source_label, count = label_source
        precheck.append(
            {
                "id": "label",
                "title": "label",
                "ready": True,
                "details": f"Ready using {source_label} images ({count:,}).",
                "run_first": None,
                "missing": [],
            }
        )
    else:
        precheck.append(
            {
                "id": "label",
                "title": "label",
                "ready": False,
                "details": "Blocked: no images available for label.",
                "run_first": "soil-shazam-data-collector download",
                "missing": [
                    {"reason": "No filtered images", "path": str(filtered_dir)},
                    {"reason": "No deduped images", "path": str(deduped_dir)},
                    {"reason": "No resized images", "path": str(resized_dir)},
                    {"reason": "No raw images", "path": str(raw_dir)},
                ],
            }
        )

    has_labels = labels_path.exists()
    precheck.append(
        {
            "id": "eval-sample",
            "title": "eval-sample",
            "ready": has_labels,
            "details": (
                f"Ready using labels file: {labels_path.name}."
                if has_labels
                else "Blocked: missing label outputs."
            ),
            "run_first": None if has_labels else "soil-shazam-data-collector label",
            "missing": [] if has_labels else [{"reason": "Missing labels", "path": str(labeled_dir)}],
        }
    )

    eval_report_missing = []
    if not has_labels:
        eval_report_missing.append({"reason": "Missing labels", "path": str(labeled_dir)})
    if not eval_sample_path.exists():
        eval_report_missing.append({"reason": "Missing evaluation sample", "path": str(eval_sample_path)})

    precheck.append(
        {
            "id": "eval-report",
            "title": "eval-report",
            "ready": len(eval_report_missing) == 0,
            "details": (
                "Ready. Evaluation sample found."
                if len(eval_report_missing) == 0
                else "Blocked: eval prerequisites are incomplete."
            ),
            "run_first": None if len(eval_report_missing) == 0 else "soil-shazam-data-collector eval-sample",
            "missing": eval_report_missing,
        }
    )

    cluster_missing = []
    if not has_labels:
        cluster_missing.append({"reason": "Missing labels", "path": str(labeled_dir)})
    if not embeddings_path.exists():
        cluster_missing.append({"reason": "Missing label embeddings", "path": str(embeddings_path)})
    if not eval_sample_path.exists():
        cluster_missing.append({"reason": "Missing evaluation sample", "path": str(eval_sample_path)})
    if not eval_metrics_path.exists():
        cluster_missing.append({"reason": "Missing evaluation metrics", "path": str(eval_metrics_path)})

    cluster_run_first = None
    if not has_labels or not embeddings_path.exists():
        cluster_run_first = "soil-shazam-data-collector label"
    elif not eval_sample_path.exists():
        cluster_run_first = "soil-shazam-data-collector eval-sample"
    elif not eval_metrics_path.exists():
        cluster_run_first = "soil-shazam-data-collector eval-report"

    precheck.append(
        {
            "id": "cluster-review",
            "title": "cluster-review",
            "ready": len(cluster_missing) == 0,
            "details": (
                "Ready. All cluster prerequisites are available."
                if len(cluster_missing) == 0
                else "Blocked: cluster prerequisites are incomplete."
            ),
            "run_first": cluster_run_first,
            "missing": cluster_missing,
        }
    )

    return precheck


def load_filter_log(cfg: dict, log_name: str) -> list[dict]:
    logs_dir = resolve(cfg["paths"]["logs"])
    path = logs_dir / log_name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _parse_float(value) -> float | None:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s or s.lower() in {"n/a", "nan", "none"}:
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def get_filter_thresholds(cfg: dict) -> dict[str, object]:
    """Return thresholds used by filter logic, preferring runtime metadata when present."""
    cache_key = "filter_thresholds"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    filter_cfg = cfg.get("filter", {})
    soil_threshold = float(filter_cfg.get("soil_threshold", 0.3))
    overlay_margin = float(filter_cfg.get("overlay_margin", 0.07))
    source = "config"
    run_ts = None

    logs_dir = resolve(cfg["paths"]["logs"])
    meta_path = logs_dir / "filter_run_config.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            soil_meta = _parse_float(meta.get("soil_threshold"))
            overlay_meta = _parse_float(meta.get("overlay_margin"))
            if soil_meta is not None:
                soil_threshold = soil_meta
            if overlay_meta is not None:
                overlay_margin = overlay_meta
            source = "run_metadata"
            run_ts = meta.get("generated_at")
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    result = {
        "soil_threshold": soil_threshold,
        "overlay_margin": overlay_margin,
        "source": source,
        "generated_at": run_ts,
    }
    cache.set(cache_key, result)
    return result


def compute_filter_metrics(
    filename: str,
    soil_map: dict[str, dict],
    overlay_map: dict[str, dict],
    thresholds: dict[str, object],
) -> dict[str, object]:
    """Compute rule-based filter margins and decision metadata for one image."""
    soil = soil_map.get(filename, {})
    ov = overlay_map.get(filename, {})

    soil_threshold = float(thresholds.get("soil_threshold", 0.3))
    overlay_margin = float(thresholds.get("overlay_margin", 0.07))

    pos = _parse_float(soil.get("positive"))
    neg = _parse_float(soil.get("negative"))
    ov_score = _parse_float(ov.get("overlay_score"))
    clean_score = _parse_float(ov.get("clean_score"))
    flagged_txt = str(ov.get("flagged", "")).strip().lower()
    flagged_by_log = flagged_txt == "true"

    overlay_delta = None
    if ov_score is not None and clean_score is not None:
        overlay_delta = ov_score - clean_score

    overlay_fail = (overlay_delta is not None and overlay_delta > overlay_margin) or flagged_by_log
    soil_threshold_fail = pos is not None and pos < soil_threshold
    soil_vs_negative_fail = pos is not None and neg is not None and pos <= neg

    kept_by_rule = (
        not overlay_fail
        and pos is not None
        and neg is not None
        and pos >= soil_threshold
        and pos > neg
    )

    rejection_mode = "unknown"
    if overlay_fail:
        rejection_mode = "overlay_margin"
    elif soil_threshold_fail and soil_vs_negative_fail:
        threshold_deficit = soil_threshold - (pos or 0.0)
        gap_deficit = (neg or 0.0) - (pos or 0.0)
        rejection_mode = "soil_vs_negative" if gap_deficit >= threshold_deficit else "soil_threshold"
    elif soil_threshold_fail:
        rejection_mode = "soil_threshold"
    elif soil_vs_negative_fail:
        rejection_mode = "soil_vs_negative"

    decision_distance = None
    nearest_boundary = "unknown"
    if kept_by_rule:
        distances: list[tuple[str, float]] = []
        if pos is not None:
            distances.append(("soil_threshold", pos - soil_threshold))
        if pos is not None and neg is not None:
            distances.append(("soil_vs_negative", pos - neg))
        if overlay_delta is not None:
            distances.append(("overlay_margin", overlay_margin - overlay_delta))
        if distances:
            nearest_boundary, decision_distance = min(distances, key=lambda x: x[1])
    else:
        fail_distances: list[tuple[str, float]] = []
        if overlay_fail and overlay_delta is not None:
            fail_distances.append(("overlay_margin", overlay_delta - overlay_margin))
        if soil_threshold_fail and pos is not None:
            fail_distances.append(("soil_threshold", soil_threshold - pos))
        if soil_vs_negative_fail and pos is not None and neg is not None:
            fail_distances.append(("soil_vs_negative", neg - pos))
        if fail_distances:
            nearest_boundary, decision_distance = max(fail_distances, key=lambda x: x[1])

    decision_band = "unknown"
    if decision_distance is not None:
        if decision_distance < 0.03:
            decision_band = "borderline"
        elif decision_distance < 0.08:
            decision_band = "moderate"
        else:
            decision_band = "clear"

    return {
        "soil_positive": pos,
        "soil_negative": neg,
        "overlay_score": ov_score,
        "clean_score": clean_score,
        "soil_threshold": soil_threshold,
        "overlay_margin": overlay_margin,
        "soil_margin": (pos - soil_threshold) if pos is not None else None,
        "soil_gap": (pos - neg) if (pos is not None and neg is not None) else None,
        "overlay_delta": overlay_delta,
        "overlay_margin_gap": (overlay_margin - overlay_delta) if overlay_delta is not None else None,
        "overlay_fail": overlay_fail,
        "soil_threshold_fail": soil_threshold_fail,
        "soil_vs_negative_fail": soil_vs_negative_fail,
        "kept_by_rule": kept_by_rule,
        "rejection_mode": rejection_mode,
        "nearest_boundary": nearest_boundary,
        "decision_distance": decision_distance,
        "decision_band": decision_band,
    }

def build_soil_score_map(cfg: dict) -> dict[str, dict]:
    """Build {filename: {positive, negative, kept}} from soil_filter.csv."""
    cached = cache.get("soil_score_map")
    if cached is not None:
        return cached

    rows = load_filter_log(cfg, "soil_filter.csv")
    result = {}
    for r in rows:
        fn = r.get("filename", "").strip()
        if fn:
            result[fn] = {
                "positive": r.get("positive_score", "").strip(),
                "negative": r.get("negative_score", "").strip(),
                "kept": r.get("kept", "").strip(),
            }
    cache.set("soil_score_map", result)
    return result


def build_overlay_score_map(cfg: dict) -> dict[str, dict]:
    """Build {filename: {overlay_score, clean_score, flagged}} from overlay_filter.csv."""
    cached = cache.get("overlay_score_map")
    if cached is not None:
        return cached

    rows = load_filter_log(cfg, "overlay_filter.csv")
    result = {}
    for r in rows:
        fn = r.get("filename", "").strip()
        if fn:
            result[fn] = {
                "overlay_score": r.get("overlay_score", "").strip(),
                "clean_score": r.get("clean_score", "").strip(),
                "flagged": r.get("flagged", "").strip(),
            }
    cache.set("overlay_score_map", result)
    return result


def list_rejected_images(cfg: dict) -> list[str]:
    """Return filenames from resized/ that were rejected by the filter step."""
    soil_map = build_soil_score_map(cfg)
    rejected = [
        fn for fn, info in soil_map.items()
        if info["kept"].lower() != "true"
    ]
    return sorted(rejected)


def list_dedup_removed_images(cfg: dict) -> list[str]:
    """Return filenames removed during dedup."""
    cached = cache.get("dedup_removed")
    if cached is not None:
        return cached

    # Primary source: explicit dedup_groups.json mapping.
    groups = load_dedup_groups(cfg)
    if groups:
        removed = sorted(
            {
                name
                for removed_list in groups.values()
                for name in (removed_list or [])
                if str(name).strip()
            }
        )
        cache.set("dedup_removed", removed)
        return removed

    # Fallback: infer by resized - deduped diff (legacy behavior).
    resized_dir = resolve(cfg["paths"]["resized"])
    deduped_dir = resolve(cfg["paths"]["deduped"])
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
    result = sorted(resized_set - deduped_set)
    cache.set("dedup_removed", result)
    return result


def load_dedup_groups(cfg: dict) -> dict[str, list[str]]:
    """Load dedup_groups.json → {kept_filename: [removed_filenames]}."""
    cached = cache.get("dedup_groups")
    if cached is not None:
        return cached

    deduped_dir = resolve(cfg["paths"]["deduped"])
    groups_path = deduped_dir / "dedup_groups.json"
    if not groups_path.exists():
        return {}
    try:
        result = json.loads(groups_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        result = {}
    cache.set("dedup_groups", result)
    return result


def extract_source(fn: str) -> str:
    """Extract download source from filename pattern: hash_source_..."""
    parts = fn.split("_")
    if len(parts) >= 3:
        src = parts[1]
        return "bing" if src == "Image" else src
    return "unknown"


def get_rejection_reason(filename: str, soil_map: dict, overlay_map: dict) -> dict:
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

