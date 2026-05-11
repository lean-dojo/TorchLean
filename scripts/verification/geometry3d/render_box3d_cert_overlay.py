#!/usr/bin/env python3
"""Render visual overlays for TorchLean 3D camera-box certificates.

This script is a human-facing companion to the Lean checker, not part of the checker itself:

1. load one or more exported camera-box JSON artifacts;
2. optionally run `lake exe verify -- camera-box3d-cert <cert>`;
3. reproject the exported 3D corners for display; and
4. save PNG overlays showing the claimed 2D box, projected 3D corners, cuboid edges, and checker
   status.

Why this exists:

The Lean checker gives the accept/reject result for the exported geometry contract, while a picture
makes failures easier to understand.  When a certificate has swapped pixel axes, a wrong principal
point, non-positive depth, or a bad bbox enclosure, the overlay lets a reader see the exact
trust-boundary failure instead of staring at a wall of numbers.
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import math
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from safe_image_io import load_local_rgb_image, load_remote_rgb_image


DEFAULT_CERT = Path("NN/Verification/Geometry3D/smoke_box3d_camera_cert.json")
DEFAULT_OUT_DIR = Path("_external/geometry3d/overlays")
DEFAULT_BAD_GLOB = "_external/geometry3d/bad/*.json"
DEFAULT_REAL_GLOB = "_external/geometry3d/realworld/*.json"


EDGE_INDEX_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def load_json(path: Path) -> dict[str, Any]:
    """Load a certificate JSON file as a dictionary."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def image_source(cert: dict[str, Any]) -> str | None:
    """Return the optional source image path/URL stored by real-model exporters."""
    metadata = cert.get("metadata")
    if isinstance(metadata, dict):
        image = metadata.get("image")
        if isinstance(image, str) and image:
            return image
    return None


def load_background(cert: dict[str, Any]) -> Image.Image:
    """Load the real source image when available; otherwise create a white canvas.

    Synthetic fixtures and malformed bad-cert cases often have no original image.  A blank canvas is
    still useful because it preserves the exported `image_width`/`image_height` coordinate system.
    """
    width = int(round(float(cert.get("image_width", 640.0))))
    height = int(round(float(cert.get("image_height", 480.0))))
    source = image_source(cert)
    if source:
        try:
            if source.startswith(("http://", "https://")):
                image = load_remote_rgb_image(source)
            else:
                image = load_local_rgb_image(Path(source))
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.BILINEAR)
            return image
        except Exception:
            # Visualization must never hide the cert itself just because a remote image is flaky.
            pass
    return Image.new("RGB", (max(width, 1), max(height, 1)), (252, 252, 248))


def verify_cert(path: Path) -> tuple[bool, str]:
    """Ask Lean whether the certificate is accepted."""
    proc = subprocess.run(
        ["lake", "exe", "verify", "--", "camera-box3d-cert", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = proc.stdout.strip()
    last_line = output.splitlines()[-1] if output else ""
    return proc.returncode == 0, last_line


def validate_flat_numbers(name: str, value: Any, expected: int) -> list[float]:
    """Validate a flat numeric list from the certificate and return floats."""
    if not isinstance(value, list):
        raise ValueError(f"{name}: expected list")
    if len(value) != expected:
        raise ValueError(f"{name}: expected {expected} floats, got {len(value)}")
    try:
        return [float(x) for x in value]
    except Exception as exc:
        raise ValueError(f"{name}: expected numeric entries") from exc


def project_corners(cert: dict[str, Any]) -> tuple[list[tuple[float, float, float]], list[str]]:
    """Project 8 exported 3D corners through the exported `3 x 4` camera matrix.

    Returns `(points, warnings)`, where every point is `(u, v, z)`.  Non-positive or non-finite depth
    is retained and marked as a warning so the overlay can show why Lean rejected the artifact.
    """
    P = validate_flat_numbers("camera_P", cert.get("camera_P"), 12)
    corners = validate_flat_numbers("corners3d", cert.get("corners3d"), 24)
    projected: list[tuple[float, float, float]] = []
    warnings: list[str] = []
    for i in range(8):
        x, y, z = corners[3 * i : 3 * i + 3]
        u_num = P[0] * x + P[1] * y + P[2] * z + P[3]
        v_num = P[4] * x + P[5] * y + P[6] * z + P[7]
        depth = P[8] * x + P[9] * y + P[10] * z + P[11]
        if not math.isfinite(depth) or depth == 0.0:
            projected.append((math.nan, math.nan, depth))
            warnings.append(f"corner {i}: zero/non-finite depth {depth:g}")
            continue
        u = u_num / depth
        v = v_num / depth
        projected.append((u, v, depth))
        if depth <= 0.0:
            warnings.append(f"corner {i}: non-positive depth {depth:g}")
    return projected, warnings


def bbox_from_cert(cert: dict[str, Any]) -> tuple[float, float, float, float]:
    """Extract the claimed `[xmin, ymin, xmax, ymax]` bbox."""
    bbox = validate_flat_numbers("bbox2d", cert.get("bbox2d"), 4)
    return bbox[0], bbox[1], bbox[2], bbox[3]


def point_inside_bbox(point: tuple[float, float, float], bbox: tuple[float, float, float, float]) -> bool:
    """Check whether one projected point is finite, positive-depth, and inside the claimed bbox."""
    u, v, z = point
    xmin, ymin, xmax, ymax = bbox
    return math.isfinite(u) and math.isfinite(v) and z > 0 and xmin <= u <= xmax and ymin <= v <= ymax


def draw_text_panel(draw: ImageDraw.ImageDraw, lines: Iterable[str], image_width: int) -> None:
    """Draw a readable translucent-ish text panel using only PIL primitives."""
    font = ImageFont.load_default()
    wrapped: list[str] = []
    max_chars = max(44, image_width // 8)
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=max_chars) or [""])
    panel_height = min(18 + 14 * len(wrapped), 150)
    draw.rectangle((0, 0, image_width, panel_height), fill=(0, 0, 0))
    y = 6
    for line in wrapped[:10]:
        draw.text((8, y), line, fill=(255, 255, 255), font=font)
        y += 14


def draw_overlay(path: Path, out_path: Path, run_verify: bool) -> Path:
    """Render one certificate overlay PNG."""
    cert = load_json(path)
    image = load_background(cert).convert("RGB")
    draw = ImageDraw.Draw(image)
    status_ok: bool | None = None
    status_line = "checker not run"
    if run_verify:
        status_ok, status_line = verify_cert(path)

    errors: list[str] = []
    projected: list[tuple[float, float, float]] = []
    bbox: tuple[float, float, float, float] | None = None
    try:
        bbox = bbox_from_cert(cert)
        projected, warnings = project_corners(cert)
        errors.extend(warnings)
    except Exception as exc:
        errors.append(str(exc))

    accepted_color = (16, 145, 80)
    rejected_color = (218, 65, 50)
    bbox_color = accepted_color if status_ok else rejected_color
    edge_color = (24, 110, 220)
    point_ok = (21, 170, 95)
    point_bad = (235, 73, 60)

    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        draw.rectangle((xmin, ymin, xmax, ymax), outline=bbox_color, width=4)
        draw.text((xmin + 4, max(0, ymin - 16)), "claimed bbox", fill=bbox_color)

    if len(projected) == 8:
        for a, b in EDGE_INDEX_PAIRS:
            pa = projected[a]
            pb = projected[b]
            if math.isfinite(pa[0]) and math.isfinite(pa[1]) and math.isfinite(pb[0]) and math.isfinite(pb[1]):
                draw.line((pa[0], pa[1], pb[0], pb[1]), fill=edge_color, width=2)
        for i, point in enumerate(projected):
            u, v, _ = point
            if not math.isfinite(u) or not math.isfinite(v):
                continue
            good = bbox is not None and point_inside_bbox(point, bbox)
            color = point_ok if good else point_bad
            r = 5
            draw.ellipse((u - r, v - r, u + r, v + r), fill=color, outline=(0, 0, 0), width=1)
            draw.text((u + 7, v + 3), str(i), fill=color)

    status_label = "ACCEPTED by Lean" if status_ok else "REJECTED by Lean" if status_ok is False else "VISUAL ONLY"
    source = str(cert.get("source", path.name))
    visible_errors = errors[:3]
    panel_lines = [
        f"{status_label}: {path.name}",
        source,
        status_line,
        *visible_errors,
    ]
    draw_text_panel(draw, panel_lines, image.width)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def collect_paths(args: argparse.Namespace) -> list[Path]:
    """Collect explicit and globbed certificate paths in stable order."""
    paths: list[Path] = [Path(p) for p in args.cert]
    for pattern in args.glob:
        paths.extend(Path(p) for p in sorted(globlib.glob(pattern)))
    if args.include_default:
        paths.insert(0, DEFAULT_CERT)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path not in seen and path.exists():
            seen.add(path)
            unique.append(path)
    return unique


def make_contact_sheet(images: list[Path], out_path: Path) -> None:
    """Create a compact contact sheet for README/paper screenshots."""
    if not images:
        return
    thumbs: list[Image.Image] = []
    for path in images:
        img = Image.open(path).convert("RGB")
        img.thumbnail((360, 260), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (380, 300), (245, 245, 242))
        canvas.paste(img, ((380 - img.width) // 2, 10))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 274), path.name[:54], fill=(30, 30, 30), font=ImageFont.load_default())
        thumbs.append(canvas)

    cols = min(3, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 380, rows * 300), (235, 235, 232))
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * 380
        y = (i // cols) * 300
        sheet.paste(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cert", action="append", default=[], help="certificate JSON path")
    parser.add_argument("--glob", action="append", default=[], help="glob of certificate JSON paths")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--include-default", action="store_true", help="include bundled synthetic good fixture")
    parser.add_argument("--no-verify", action="store_true", help="do not call the Lean checker")
    parser.add_argument("--contact-sheet", action="store_true", help="also write geometry3d_contact_sheet.png")
    args = parser.parse_args()

    paths = collect_paths(args)
    if not paths:
        raise SystemExit("no certificate paths found")

    outputs: list[Path] = []
    for path in paths:
        out = args.out_dir / f"{path.stem}.png"
        outputs.append(draw_overlay(path, out, run_verify=not args.no_verify))
        print(f"wrote {out}", flush=True)

    if args.contact_sheet:
        sheet = args.out_dir / "geometry3d_contact_sheet.png"
        make_contact_sheet(outputs, sheet)
        print(f"wrote {sheet}", flush=True)


if __name__ == "__main__":
    main()
