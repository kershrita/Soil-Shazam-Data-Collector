"""Bing image downloader — no API key required."""

from __future__ import annotations

import logging
from pathlib import Path

from .base import ImageDownloader
from ..utils import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


class BingDownloader(ImageDownloader):
    """Download images using bing-image-downloader."""

    source_name = "bing"

    def download(
        self,
        query: str,
        limit: int,
        output_dir: Path,
        timeout: int = 30,
    ) -> list[Path]:
        from bing_image_downloader import downloader as bing_dl

        target_dir = self._make_output_dir(output_dir, query)
        existing = self._count_existing(target_dir)
        if existing >= limit:
            logger.info(f"[bing] Skipping '{query}': already have {existing} images")
            return list(target_dir.iterdir())

        logger.info(f"[bing] Downloading up to {limit} images for '{query}'")
        try:
            # bing-image-downloader creates a subdirectory named after the query
            bing_dl.download(
                query,
                limit=limit,
                output_dir=str(target_dir.parent),
                adult_filter_off=False,
                force_replace=False,
                timeout=timeout,
                verbose=False,
            )
            # Move files from the query-named subdir into our target dir
            query_subdir = target_dir.parent / query
            if query_subdir.exists() and query_subdir != target_dir:
                for f in query_subdir.iterdir():
                    if f.is_file():
                        dest = target_dir / f.name
                        if not dest.exists():
                            f.rename(dest)
                # Clean up the extra directory
                try:
                    query_subdir.rmdir()
                except OSError:
                    pass
        except Exception as e:
            logger.error(f"[bing] Error downloading '{query}': {e}")

        downloaded = [
            p for p in target_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        logger.info(f"[bing] Got {len(downloaded)} images for '{query}'")
        return downloaded
