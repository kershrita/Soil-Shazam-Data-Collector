from .image_utils import (
    IMAGE_EXTENSIONS,
    canonical_image_name,
    check_resolution,
    collect_image_paths,
    is_image_file,
    load_image,
    resize_image,
    save_as_jpeg,
)
from .logging import setup_logging
from .pipeline_manifest import (
    ensure_manifest_entry,
    load_pipeline_manifest,
    mark_manifest_step,
    save_pipeline_manifest,
    update_manifest_after_dedup,
    update_manifest_after_label,
)

__all__ = [
    "IMAGE_EXTENSIONS",
    "canonical_image_name",
    "check_resolution",
    "collect_image_paths",
    "is_image_file",
    "load_image",
    "resize_image",
    "save_as_jpeg",
    "setup_logging",
    "ensure_manifest_entry",
    "load_pipeline_manifest",
    "mark_manifest_step",
    "save_pipeline_manifest",
    "update_manifest_after_dedup",
    "update_manifest_after_label",
]


def __getattr__(name: str):
    if name in ("CLIPModel", "get_clip_model"):
        from .clip_model import CLIPModel, get_clip_model

        globals()["CLIPModel"] = CLIPModel
        globals()["get_clip_model"] = get_clip_model
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
