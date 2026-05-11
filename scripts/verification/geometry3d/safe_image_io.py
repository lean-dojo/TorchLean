#!/usr/bin/env python3
"""Small image-loading helpers for Geometry3D certificate exporters."""

from __future__ import annotations

import io
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image


DEFAULT_TIMEOUT_SECONDS = 20.0


def load_local_rgb_image(path: Path) -> Image.Image:
    """Load a local image as RGB."""
    return Image.open(path).convert("RGB")


def load_remote_rgb_image(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Image.Image:
    """Download an HTTPS image and load it as RGB."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"remote image URLs must use https://, got {url!r}")

    request = urllib.request.Request(url, headers={"User-Agent": "TorchLean-Geometry3D/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()

    return Image.open(io.BytesIO(bytes(data))).convert("RGB")
