#!/usr/bin/env python3
"""Export a TorchLean 3D camera-box certificate from Cube R-CNN / Omni3D predictions.

This exporter treats Cube R-CNN, SAM-3D-style systems, or any other 3D detector as untrusted
producers; TorchLean accepts the result only after Lean recomputes the
projection contract with:

    lake exe verify -- camera-box3d-cert <out.json>

Expected input shape
--------------------
The exporter consumes the public Omni3D/Cube R-CNN image-level prediction shape documented by the
Omni3D repository:

    {
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "width": 640,
      "height": 480,
      "instances": [
        {
          "bbox": [x1, y1, x2, y2],
          "score": 0.91,
          "bbox3D": [[X,Y,Z], ... eight corners ...]
        }
      ]
    }

The top-level JSON may be one such object, a list of image objects, or an object containing a
`predictions`/`images` list.  The 3D corners are assumed to be in the camera coordinate frame, so
the exported 3x4 projection matrix is `[K | 0]`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("_external/geometry3d/omni3d_box3d_cert.json")
FORMAT = "torchlean.camera.box3d.v1"


def _as_float_list(value: Any, *, name: str) -> list[float]:
    """Flatten a nested numeric JSON list into row-major floats.

    Omni3D-style outputs often store matrices/corners as nested lists, while the TorchLean JSON
    schema stores flat row-major arrays.  This helper performs only shape/number sanitation; Lean
    later checks the geometry contract.
    """
    if not isinstance(value, list):
        raise ValueError(f"{name}: expected a JSON list")
    out: list[float] = []
    for item in value:
        if isinstance(item, list):
            out.extend(_as_float_list(item, name=name))
        elif isinstance(item, (int, float)):
            out.append(float(item))
        else:
            raise ValueError(f"{name}: expected only numeric entries")
    return out


def _prediction_records(payload: Any) -> list[dict[str, Any]]:
    """Find image-level prediction records in common exporter layouts."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("predictions", "images", "outputs", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [payload]
    raise ValueError("prediction JSON must be an object or a list of objects")


def _get_intrinsics(record: dict[str, Any], args: argparse.Namespace) -> list[float]:
    """Return a flat `3 x 3` camera intrinsic matrix.

    Prefer the model/exporter's image-level `K`.  If a prediction file lacks `K`, users may pass a
    fallback focal length and image dimensions.  Those fallback values remain untrusted candidate
    metadata until Lean checks the projected corners.
    """
    if "K" in record:
        flat = _as_float_list(record["K"], name="K")
        if len(flat) != 9:
            raise ValueError(f"K: expected 9 floats, got {len(flat)}")
        return flat

    if args.focal_length is None:
        raise ValueError("missing K; pass --focal-length or provide image-level K")
    width = float(record.get("width", args.width if args.width is not None else 0.0))
    height = float(record.get("height", args.height if args.height is not None else 0.0))
    if width <= 0 or height <= 0:
        raise ValueError("missing image width/height; pass --width and --height")
    cx = args.principal_x if args.principal_x is not None else width / 2.0
    cy = args.principal_y if args.principal_y is not None else height / 2.0
    fx = args.focal_length
    fy = args.focal_y if args.focal_y is not None else fx
    return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


def _choose_instance(instances: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Select one predicted 3D instance from an image record."""
    if not instances:
        raise ValueError("selected image has no instances")
    if args.instance_index is not None:
        if args.instance_index < 0 or args.instance_index >= len(instances):
            raise ValueError(f"--instance-index out of range for {len(instances)} instances")
        return instances[args.instance_index]
    return max(instances, key=lambda inst: float(inst.get("score", 0.0)))


def _field(obj: dict[str, Any], names: tuple[str, ...], *, ctx: str) -> Any:
    """Read one field while accepting a small set of common spelling variants."""
    for name in names:
        if name in obj:
            return obj[name]
    raise ValueError(f"{ctx}: missing one of {names}")


def export_cert(payload: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Convert one Omni3D-style prediction into the TorchLean certificate schema.

    This path assumes the producer already exported true 3D corners (`bbox3D`/`corners3d`).  The
    script only packs them with `[K | 0]`, image dimensions, and the claimed 2D bbox.  Lean then
    independently recomputes all projections and rejects incoherent artifacts.
    """
    records = _prediction_records(payload)
    if args.image_index < 0 or args.image_index >= len(records):
        raise ValueError(f"--image-index out of range for {len(records)} image records")
    record = records[args.image_index]
    instances_raw = _field(record, ("instances", "detections", "predictions"), ctx="image record")
    if not isinstance(instances_raw, list):
        raise ValueError("image record instances/detections/predictions must be a list")
    instances = [x for x in instances_raw if isinstance(x, dict)]
    instance = _choose_instance(instances, args)

    width = float(record.get("width", args.width if args.width is not None else 0.0))
    height = float(record.get("height", args.height if args.height is not None else 0.0))
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")

    K = _get_intrinsics(record, args)
    camera_p = [
        K[0], K[1], K[2], 0.0,
        K[3], K[4], K[5], 0.0,
        K[6], K[7], K[8], 0.0,
    ]

    bbox = _as_float_list(_field(instance, ("bbox", "bbox2D", "bbox2d"), ctx="instance"), name="bbox")
    corners = _as_float_list(
        _field(instance, ("bbox3D", "bbox3d", "corners3d", "corners_3d"), ctx="instance"),
        name="bbox3D",
    )
    if len(bbox) != 4:
        raise ValueError(f"bbox: expected 4 floats, got {len(bbox)}")
    if len(corners) != 24:
        raise ValueError(f"bbox3D/corners3d: expected 24 floats, got {len(corners)}")

    return {
        "format": FORMAT,
        "source": args.source,
        "image_width": width,
        "image_height": height,
        "tol": float(args.tol),
        "camera_P": camera_p,
        "corners3d": corners,
        "bbox2d": bbox,
        "metadata": {
            "producer": "scripts/verification/geometry3d/export_omni3d_box3d_cert.py",
            "input": str(args.prediction_json),
            "image_index": args.image_index,
            "instance_index": args.instance_index,
            "selection": "highest_score" if args.instance_index is None else "explicit_index",
            "score": instance.get("score"),
            "category_id": instance.get("category_id"),
        },
    }


def main() -> None:
    """Command-line entrypoint for converting one 3D-detector prediction JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-json", type=Path, required=True, help="Cube R-CNN/Omni3D prediction JSON")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="TorchLean cert output path")
    parser.add_argument("--image-index", type=int, default=0, help="image record to export")
    parser.add_argument("--instance-index", type=int, default=None, help="instance to export; default: highest score")
    parser.add_argument("--tol", type=float, default=2.0, help="pixel tolerance for bbox enclosure")
    parser.add_argument("--source", default="Cube R-CNN / Omni3D prediction JSON", help="source string stored in cert")
    parser.add_argument("--width", type=float, default=None, help="fallback image width if prediction JSON omits it")
    parser.add_argument("--height", type=float, default=None, help="fallback image height if prediction JSON omits it")
    parser.add_argument("--focal-length", type=float, default=None, help="fallback fx if prediction JSON omits K")
    parser.add_argument("--focal-y", type=float, default=None, help="fallback fy; defaults to fx")
    parser.add_argument("--principal-x", type=float, default=None, help="fallback cx; defaults to width/2")
    parser.add_argument("--principal-y", type=float, default=None, help="fallback cy; defaults to height/2")
    parser.add_argument("--verify", action="store_true", help="run `lake exe verify -- camera-box3d-cert` after export")
    args = parser.parse_args()

    with args.prediction_json.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cert = export_cert(payload, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(cert, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.out}", flush=True)

    if args.verify:
        subprocess.run(["lake", "exe", "verify", "--", "camera-box3d-cert", str(args.out)], check=True)


if __name__ == "__main__":
    main()
