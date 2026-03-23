"""Helpers for maintaining a unified per-image pipeline manifest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LABEL_CATEGORIES = [
    "soil_color",
    "soil_texture",
    "particle_size",
    "crack_presence",
    "rock_fraction",
    "surface_structure",
    "surface_roughness",
]

SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _now_iso(),
        "images": {},
    }


def load_pipeline_manifest(path: Path) -> dict[str, Any]:
    """Load manifest JSON or return an empty structure if missing/invalid."""
    if not path.exists():
        return _empty_manifest()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return _empty_manifest()

    if not isinstance(payload, dict):
        return _empty_manifest()

    images = payload.get("images")
    if not isinstance(images, dict):
        payload["images"] = {}

    payload["schema_version"] = SCHEMA_VERSION
    payload["updated_at"] = _now_iso()
    return payload


def save_pipeline_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Persist manifest JSON."""
    payload = manifest if isinstance(manifest, dict) else _empty_manifest()
    payload["schema_version"] = SCHEMA_VERSION
    payload["updated_at"] = _now_iso()
    if not isinstance(payload.get("images"), dict):
        payload["images"] = {}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def ensure_manifest_entry(manifest: dict[str, Any], image_name: str) -> dict[str, Any]:
    """Ensure a valid manifest entry exists for an image."""
    images = manifest.setdefault("images", {})
    entry = images.get(image_name)
    if not isinstance(entry, dict):
        entry = {"image": image_name, "steps": {}}
        images[image_name] = entry

    entry["image"] = image_name
    steps = entry.get("steps")
    if not isinstance(steps, dict):
        entry["steps"] = {}
    return entry


def mark_manifest_step(
    manifest: dict[str, Any],
    image_name: str,
    step: str,
    data: dict[str, Any],
) -> None:
    """Merge step data into one image manifest entry."""
    if not image_name:
        return
    entry = ensure_manifest_entry(manifest, image_name)
    steps = entry.setdefault("steps", {})
    existing = steps.get(step)
    if not isinstance(existing, dict):
        existing = {}
    existing.update(data)
    existing["updated_at"] = _now_iso()
    steps[step] = existing


def update_manifest_after_dedup(
    manifest_path: Path,
    kept_images: list[str],
    dedup_groups: dict[str, list[str]],
    source_paths: dict[str, str] | None = None,
) -> None:
    """Apply dedup step state to the unified manifest."""
    manifest = load_pipeline_manifest(manifest_path)
    source_paths = source_paths or {}

    for image_name in kept_images:
        if not image_name:
            continue
        mark_manifest_step(
            manifest,
            image_name,
            "dedup",
            {
                "kept": True,
                "duplicate_of": None,
                "source_path": source_paths.get(image_name),
            },
        )

    for kept_name, removed_list in (dedup_groups or {}).items():
        for removed_name in removed_list or []:
            if not removed_name:
                continue
            mark_manifest_step(
                manifest,
                removed_name,
                "dedup",
                {
                    "kept": False,
                    "duplicate_of": kept_name,
                    "source_path": source_paths.get(removed_name),
                },
            )

    save_pipeline_manifest(manifest_path, manifest)


def update_manifest_after_label(
    manifest_path: Path,
    labels: list[dict[str, Any]],
) -> None:
    """Apply label step state to the unified manifest."""
    manifest = load_pipeline_manifest(manifest_path)
    for entry in labels:
        image_name = str(entry.get("image", "")).strip()
        if not image_name:
            continue

        categories = {
            cat: str(entry.get(cat, "")).strip()
            for cat in LABEL_CATEGORIES
            if str(entry.get(cat, "")).strip()
        }
        mark_manifest_step(
            manifest,
            image_name,
            "label",
            {
                "labeled": True,
                "categories": categories,
            },
        )

    save_pipeline_manifest(manifest_path, manifest)
