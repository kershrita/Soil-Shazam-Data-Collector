"""DuckDuckGo image downloader — no API key required."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import requests

from soil_collector.downloader.base import ImageDownloader
from soil_collector.utils.image_utils import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

# Delay between search requests to avoid 403 rate-limiting
_QUERY_DELAY = 3.0  # seconds


class DuckDuckGoDownloader(ImageDownloader):
    """Download images from DuckDuckGo using duckduckgo-search."""

    source_name = "ddg"
    _last_search_time: float = 0.0

    def download(
        self,
        query: str,
        limit: int,
        output_dir: Path,
        timeout: int = 30,
    ) -> list[Path]:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            from ddgs import DDGS

        target_dir = self._make_output_dir(output_dir, query)
        existing = self._count_existing(target_dir)
        if existing >= limit:
            logger.info(f"[ddg] Skipping '{query}': already have {existing} images")
            return list(target_dir.iterdir())

        logger.info(f"[ddg] Downloading up to {limit} images for '{query}'")
        downloaded = []

        # Rate-limit: wait between searches
        elapsed = time.time() - DuckDuckGoDownloader._last_search_time
        if elapsed < _QUERY_DELAY:
            time.sleep(_QUERY_DELAY - elapsed)

        results = []
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.images(query, max_results=limit))
                DuckDuckGoDownloader._last_search_time = time.time()
                break
            except Exception as e:
                err_str = str(e)
                if "Ratelimit" in err_str or "403" in err_str:
                    wait = _QUERY_DELAY * (attempt + 2)
                    logger.warning(f"[ddg] Rate-limited on '{query}', waiting {wait:.0f}s (attempt {attempt+1}/3)")
                    time.sleep(wait)
                    DuckDuckGoDownloader._last_search_time = time.time()
                else:
                    logger.error(f"[ddg] Error searching '{query}': {e}")
                    break

        session = requests.Session()
        for i, result in enumerate(results):
            url = result.get("image", "")
            if not url:
                continue
            try:
                resp = session.get(url, timeout=timeout, stream=True)
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")
                if "image" not in content_type:
                    continue

                # Determine extension from content-type
                ext = ".jpg"
                if "png" in content_type:
                    ext = ".png"
                elif "webp" in content_type:
                    ext = ".webp"

                # Use URL hash as filename to avoid collisions
                url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
                filename = f"ddg_{i:05d}_{url_hash}{ext}"
                filepath = target_dir / filename

                if filepath.exists():
                    continue

                content = resp.content
                if len(content) < 1000:  # Skip tiny/broken images
                    continue

                filepath.write_bytes(content)
                downloaded.append(filepath)

            except (requests.RequestException, OSError) as e:
                logger.debug(f"[ddg] Failed to download {url}: {e}")
                continue

        logger.info(f"[ddg] Got {len(downloaded)} images for '{query}'")
        return downloaded
