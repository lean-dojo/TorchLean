#!/usr/bin/env python3
"""Plot bbox-vs-projected-footprint diagnostics for a Geometry3D certificate.

The overlay image answers "what did this look like on the input image?".  This plot answers the
more audit-oriented question "which numeric interval failed?"  It reads the metadata written by the
WildDet3D exporter and draws the model's 2D bbox next to the bbox induced by projecting the exported
3D corners.

This script is not trusted by Lean.  It is only a visual explanation for humans; the
accept/reject decision comes from `lake exe verify -- camera-box3d-cert ...`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_CERT = Path("_external/geometry3d/wilddet3d/wilddet3d_cat_box3d_cert.json")
DEFAULT_OUT = Path("_external/geometry3d/wilddet3d/wilddet3d_bbox_diagnostic.png")


def _bbox_from_metadata(metadata: dict[str, Any], key: str) -> list[float]:
    """Extract a four-float bbox from certificate metadata."""
    value = metadata.get(key)
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"metadata.{key}: expected four floats")
    return [float(x) for x in value]


def _draw_interval(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    lo: float,
    hi: float,
    domain_hi: float,
    y: int,
    color: tuple[int, int, int],
    width: int,
) -> None:
    """Draw one interval on a shared `[0, domain_hi]` axis."""
    axis_x0 = 180
    axis_x1 = width - 60
    axis_y = y + 16
    scale = (axis_x1 - axis_x0) / max(domain_hi, 1.0)
    x0 = axis_x0 + lo * scale
    x1 = axis_x0 + hi * scale
    font = ImageFont.load_default()
    draw.text((24, y + 4), label, fill=(35, 35, 35), font=font)
    draw.line((axis_x0, axis_y, axis_x1, axis_y), fill=(190, 190, 190), width=2)
    draw.rectangle((x0, axis_y - 8, x1, axis_y + 8), fill=color, outline=(30, 30, 30), width=1)
    draw.text((axis_x0, axis_y + 14), "0", fill=(90, 90, 90), font=font)
    draw.text((axis_x1 - 36, axis_y + 14), f"{domain_hi:.0f}", fill=(90, 90, 90), font=font)
    draw.text((x0, axis_y - 25), f"{lo:.1f}", fill=color, font=font)
    draw.text((x1 - 36, axis_y - 25), f"{hi:.1f}", fill=color, font=font)


def plot(cert_path: Path, out_path: Path) -> Path:
    """Render the diagnostic plot and return the output path."""
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    metadata = cert.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{cert_path}: missing metadata object")

    model = _bbox_from_metadata(metadata, "model_bbox2d")
    projected = _bbox_from_metadata(metadata, "projected_envelope_bbox2d")
    exported = _bbox_from_metadata(metadata, "exported_bbox2d")
    image_w = float(cert["image_width"])
    image_h = float(cert["image_height"])
    encloses = bool(metadata.get("model_bbox_encloses_projected_corners"))

    width, height = 960, 520
    image = Image.new("RGB", (width, height), (250, 249, 245))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    title = "WildDet3D 2D Box vs Projected 3D Footprint"
    draw.text((24, 24), title, fill=(0, 0, 0), font=font)
    status = (
        "model 2D bbox encloses projected 3D corners"
        if encloses
        else "model 2D bbox does NOT enclose projected 3D corners"
    )
    status_color = (18, 135, 80) if encloses else (205, 55, 45)
    draw.text((24, 52), status, fill=status_color, font=font)

    model_color = (235, 82, 62)
    projected_color = (37, 126, 213)
    exported_color = (20, 150, 95)

    draw.text((24, 92), "X interval in pixels", fill=(0, 0, 0), font=font)
    _draw_interval(draw, label="model bbox x", lo=model[0], hi=model[2],
                   domain_hi=image_w, y=125, color=model_color, width=width)
    _draw_interval(draw, label="projected x", lo=projected[0], hi=projected[2],
                   domain_hi=image_w, y=180, color=projected_color, width=width)
    _draw_interval(draw, label="exported cert x", lo=exported[0], hi=exported[2],
                   domain_hi=image_w, y=235, color=exported_color, width=width)

    draw.text((24, 305), "Y interval in pixels", fill=(0, 0, 0), font=font)
    _draw_interval(draw, label="model bbox y", lo=model[1], hi=model[3],
                   domain_hi=image_h, y=338, color=model_color, width=width)
    _draw_interval(draw, label="projected y", lo=projected[1], hi=projected[3],
                   domain_hi=image_h, y=393, color=projected_color, width=width)
    _draw_interval(draw, label="exported cert y", lo=exported[1], hi=exported[3],
                   domain_hi=image_h, y=448, color=exported_color, width=width)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def main() -> None:
    """Parse CLI args and write the interval diagnostic image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cert", type=Path, default=DEFAULT_CERT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(f"wrote {plot(args.cert, args.out)}")


if __name__ == "__main__":
    main()
