"""Google image downloader — free, no API key.

Uses direct HTTP scraping since icrawler's Google parser is broken
(Google changed their HTML, causing 'NoneType' is not iterable errors).
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import requests

from .base import ImageDownloader

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Delay between search requests to avoid IP bans
_QUERY_DELAY = 2.0


def _extract_image_urls(html: str, limit: int) -> list[str]:
    """Extract image URLs from Google Images HTML response."""
    # Google embeds full-res image URLs in various JSON-like data attributes.
    # Pattern matches URLs that look like actual images.
    urls: list[str] = []
    seen: set[str] = set()

    # Match URLs in the page that end with common image extensions
    # These appear in JSON data blobs embedded in the page
    for pattern in [
        r'"(https?://[^"]+\.(?:jpg|jpeg|png|webp)(?:\?[^"]*)?)"',
        r'\["(https?://[^"]+\.(?:jpg|jpeg|png|webp)(?:\?[^"]*)?)"',
    ]:
        for match in re.finditer(pattern, html, re.IGNORECASE):
            url = match.group(1)
            # Skip Google's own thumbnails and icons
            if any(skip in url for skip in [
                "gstatic.com", "google.com", "googleapis.com",
                "googleusercontent.com/images", "favicon", "icon",
                "logo", "thumbnail", "/th?", "encrypted-tbn",
            ]):
                continue
            if url not in seen and len(url) < 2000:
                seen.add(url)
                urls.append(url)
                if len(urls) >= limit:
                    return urls
    return urls


class GoogleDownloader(ImageDownloader):
    """Download images from Google via direct HTTP scraping."""

    source_name = "google"
    _last_search_time: float = 0.0
    _rate_lock = threading.Lock()

    def download(
        self,
        query: str,
        limit: int,
        output_dir: Path,
        timeout: int = 30,
        max_retries: int = 0,
    ) -> list[Path]:
        target_dir = self._make_output_dir(output_dir, query)
        existing = self._count_existing(target_dir)
        if existing >= limit:
            logger.info(f"[google] Skipping '{query}': already have {existing} images")
            return list(target_dir.iterdir())

        logger.info(f"[google] Downloading up to {limit} images for '{query}'")

        # Rate-limit: wait between searches (thread-safe)
        with GoogleDownloader._rate_lock:
            elapsed = time.time() - GoogleDownloader._last_search_time
            if elapsed < _QUERY_DELAY:
                time.sleep(_QUERY_DELAY - elapsed)
            GoogleDownloader._last_search_time = time.time()

        downloaded: list[Path] = []
        session = requests.Session()
        session.headers.update(_HEADERS)

        def _get_with_retries(url: str, **kwargs: Any) -> requests.Response:
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return session.get(url, timeout=timeout, **kwargs)
                except requests.RequestException as e:
                    last_error = e
                    if attempt < max_retries:
                        time.sleep(min(1.0 * (attempt + 1), 5.0))
            if last_error is not None:
                raise last_error
            raise RuntimeError("Unreachable: retry loop ended without response or error")

        try:
            # Fetch Google Images search results page
            search_url = (
                f"https://www.google.com/search?q={quote_plus(query)}"
                f"&tbm=isch&ijn=0"
            )
            resp = _get_with_retries(search_url)
            resp.raise_for_status()

            image_urls = _extract_image_urls(resp.text, limit * 3)  # get extras for failures
            logger.debug(f"[google] Found {len(image_urls)} candidate URLs for '{query}'")

            for i, url in enumerate(image_urls):
                if len(downloaded) >= limit:
                    break
                try:
                    img_resp = _get_with_retries(url, stream=True)
                    img_resp.raise_for_status()

                    content_type = img_resp.headers.get("content-type", "")
                    if "image" not in content_type:
                        continue

                    ext = ".jpg"
                    if "png" in content_type:
                        ext = ".png"
                    elif "webp" in content_type:
                        ext = ".webp"

                    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
                    filename = f"google_{i:05d}_{url_hash}{ext}"
                    filepath = target_dir / filename

                    if filepath.exists():
                        continue

                    content = img_resp.content
                    if len(content) < 1000:
                        continue

                    filepath.write_bytes(content)
                    downloaded.append(filepath)
                except (requests.RequestException, OSError):
                    continue

        except Exception as e:
            logger.error(f"[google] Error downloading '{query}': {e}")

        logger.info(f"[google] Got {len(downloaded)} images for '{query}'")
        return downloaded
