from .image_utils import (
    IMAGE_EXTENSIONS,
    check_resolution,
    collect_image_paths,
    is_image_file,
    load_image,
    resize_image,
    save_as_jpeg,
)
from .logging import setup_logging

__all__ = [
    "CLIPModel",
    "get_clip_model",
    "IMAGE_EXTENSIONS",
    "check_resolution",
    "collect_image_paths",
    "is_image_file",
    "load_image",
    "resize_image",
    "save_as_jpeg",
    "setup_logging",
]


def __getattr__(name: str):
    if name in ("CLIPModel", "get_clip_model"):
        from .clip_model import CLIPModel, get_clip_model

        globals()["CLIPModel"] = CLIPModel
        globals()["get_clip_model"] = get_clip_model
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

