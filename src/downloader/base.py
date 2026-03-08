"""Abstract base class for image downloaders."""

from __future__ import annotations

import abc
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def slugify(text: str) -> str:
    """Convert a query string to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "_", text)
    return text.strip("_")


class ImageDownloader(abc.ABC):
    """Abstract interface for image downloading sources."""

    source_name: str = "base"

    @abc.abstractmethod
    def download(
        self,
        query: str,
        limit: int,
        output_dir: Path,
        timeout: int = 30,
    ) -> list[Path]:
        """Download images for a query.

        Args:
            query: Search query string.
            limit: Maximum number of images to download.
            output_dir: Root output directory (source/query subdirs created inside).
            timeout: HTTP timeout in seconds.

        Returns:
            List of paths to downloaded image files.
        """
        ...

    def _make_output_dir(self, output_dir: Path, query: str) -> Path:
        """Create and return the source/query-specific output directory."""
        target = output_dir / self.source_name / slugify(query)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _count_existing(self, directory: Path) -> int:
        """Count existing image files in a directory."""
        from soil_collector.utils.image_utils import IMAGE_EXTENSIONS
        if not directory.exists():
            return 0
        return sum(1 for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
