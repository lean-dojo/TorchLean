"""Small image-loading helpers for Geometry3D certificate exporters."""

from __future__ import annotations

import io
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image


DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise ValueError(f"remote image redirects must use https://, got {newurl!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def load_local_rgb_image(path: Path) -> Image.Image:
    """Load a local image as RGB."""
    with Image.open(path) as image:
        return image.convert("RGB")


def load_remote_rgb_image(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Image.Image:
    """Load RGB from a bounded HTTPS download, rejecting redirects to another scheme."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"remote image URLs must use https://, got {url!r}")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    request = urllib.request.Request(url, headers={"User-Agent": "TorchLean-Geometry3D/1.0"})
    opener = urllib.request.build_opener(_HttpsOnlyRedirectHandler())
    with opener.open(request, timeout=timeout) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"remote image exceeds {max_bytes} bytes")

    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")
