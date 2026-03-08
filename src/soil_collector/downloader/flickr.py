"""Flickr image downloader — free, no API key required.

Uses Flickr's public feed API and website search to find images.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from pathlib import Path

import requests

from soil_collector.downloader.base import ImageDownloader
from soil_collector.utils.image_utils import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Delay between search requests to be respectful
_QUERY_DELAY = 2.0


class FlickrDownloader(ImageDownloader):
    """Download images from Flickr via public feed + search."""

    source_name = "flickr"
    _last_search_time: float = 0.0
    _rate_lock = threading.Lock()

    def download(
        self,
        query: str,
        limit: int,
        output_dir: Path,
        timeout: int = 30,
    ) -> list[Path]:
        target_dir = self._make_output_dir(output_dir, query)
        existing = self._count_existing(target_dir)
        if existing >= limit:
            logger.info(f"[flickr] Skipping '{query}': already have {existing} images")
            return list(target_dir.iterdir())

        logger.info(f"[flickr] Downloading up to {limit} images for '{query}'")

        # Rate-limit: wait between searches (thread-safe)
        with FlickrDownloader._rate_lock:
            elapsed = time.time() - FlickrDownloader._last_search_time
            if elapsed < _QUERY_DELAY:
                time.sleep(_QUERY_DELAY - elapsed)
            FlickrDownloader._last_search_time = time.time()

        image_urls = self._search_images(query, limit, timeout)

        downloaded: list[Path] = []
        session = requests.Session()
        session.headers.update(_HEADERS)

        for i, url in enumerate(image_urls):
            if len(downloaded) >= limit:
                break
            try:
                resp = session.get(url, timeout=timeout, stream=True)
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")
                if "image" not in content_type:
                    continue

                ext = ".jpg"
                if "png" in content_type:
                    ext = ".png"
                elif "webp" in content_type:
                    ext = ".webp"

                url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
                filename = f"flickr_{i:05d}_{url_hash}{ext}"
                filepath = target_dir / filename

                if filepath.exists():
                    continue

                content = resp.content
                if len(content) < 1000:
                    continue

                filepath.write_bytes(content)
                downloaded.append(filepath)
            except (requests.RequestException, OSError) as e:
                logger.debug(f"[flickr] Failed to download {url}: {e}")
                continue

        logger.info(f"[flickr] Got {len(downloaded)} images for '{query}'")
        return downloaded

    def _search_images(self, query: str, limit: int, timeout: int) -> list[str]:
        """Gather image URLs from Flickr feed + paginated website search."""
        urls: list[str] = []
        seen: set[str] = set()
        # Extra buffer to account for download failures (~30%)
        target = int(limit * 1.4)

        # Source 1: Public feed API (always returns 20, fast, reliable)
        try:
            tags = query.replace(" ", ",")
            resp = requests.get(
                "https://api.flickr.com/services/feeds/photos_public.gne",
                params={"tags": tags, "tagmode": "all", "format": "json", "nojsoncallback": 1},
                headers=_HEADERS,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("items", []):
                small_url = item.get("media", {}).get("m", "")
                if small_url:
                    # Upgrade _m (240px) to _b (1024px)
                    large_url = small_url.replace("_m.jpg", "_b.jpg")
                    if large_url not in seen:
                        seen.add(large_url)
                        urls.append(large_url)
        except Exception as e:
            logger.debug(f"[flickr] Feed API error for '{query}': {e}")

        # Source 2: Paginated website search (~25 URLs/page)
        page = 1
        consecutive_empty = 0
        while len(urls) < target:
            try:
                resp = requests.get(
                    "https://www.flickr.com/search/",
                    params={"text": query, "media": "photos", "content_type": 1, "page": page},
                    headers=_HEADERS,
                    timeout=timeout,
                )
                resp.raise_for_status()
                found = re.findall(r"(live\.staticflickr\.com/\S+?\.jpg)", resp.text)
                new_count = 0
                for match in found:
                    url = "https://" + match
                    large_url = re.sub(r"_[a-z]\.jpg$", "_b.jpg", url)
                    if large_url not in seen:
                        seen.add(large_url)
                        urls.append(large_url)
                        new_count += 1
                if new_count == 0:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        break  # No more results
                else:
                    consecutive_empty = 0
                page += 1
                # Polite delay between page requests
                time.sleep(0.5)
            except Exception as e:
                logger.debug(f"[flickr] Search page {page} error for '{query}': {e}")
                break

        logger.debug(f"[flickr] Found {len(urls)} candidate URLs for '{query}' ({page-1} pages)")
        return urls[:target]
