#!/usr/bin/env python3
"""Run Hugging Face vision pipelines and export a TorchLean 3D-box certificate.

This is the PyTorch-to-Lean Geometry3D path:

1. run an object detector on a normal image;
2. run Depth Anything V2 on the same image;
3. lift one detected 2D box into a conservative 3D frustum/cuboid using the depth crop; and
4. ask Lean to check that the exported 3D corners project back inside the claimed 2D box.

The learned models are not trusted.  The certificate says only:

    Given the exported camera matrix, 3D corners, image size, and 2D box, the geometry is coherent.

The trust boundary is explicit: PyTorch proposes geometry, and TorchLean checks the contract before
downstream code relies on it.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from safe_image_io import load_local_rgb_image, load_remote_rgb_image


FORMAT = "torchlean.camera.box3d.v1"
DEFAULT_IMAGE_URL = (
    "https://images.cocodataset.org/val2017/000000039769.jpg"
)
DEFAULT_OUT = Path("_external/geometry3d/hf_depth_box3d_cert.json")
DEFAULT_BATCH_MANIFEST = Path("scripts/verification/geometry3d/realworld_manifest.json")


def load_image(path: Path | None, url: str | None) -> Image.Image:
    """Load a local or remote RGB image.

    This is producer-side convenience only.  Lean never trusts pixels or model inference; it only
    receives the exported numeric certificate after this script finishes.
    """
    if path is not None:
        return load_local_rgb_image(path)
    if url is None:
        url = DEFAULT_IMAGE_URL
    return load_remote_rgb_image(url)


def pipeline_device(device: str) -> int | str:
    """Translate a friendly device string into the Hugging Face pipeline convention.

    HF pipelines use `-1` for CPU and integer GPU IDs for CUDA.  Keeping this logic in one place
    makes the command-line interface stable across CPU-only laptops and GPU workstations.
    """
    if device == "auto":
        return 0 if torch.cuda.is_available() else -1
    if device == "cpu":
        return -1
    if device.startswith("cuda"):
        return 0
    return device


def detection_box(det: dict[str, Any]) -> tuple[float, float, float, float]:
    """Extract `[xmin, ymin, xmax, ymax]` from one HF object-detection result.

    The detector output is untrusted.  This function validates only the minimal JSON shape needed to produce a
    candidate certificate; Lean later checks whether the candidate box encloses the projected 3D
    corners.
    """
    box = det.get("box")
    if not isinstance(box, dict):
        raise ValueError(f"detection has no box dict: {det}")
    return (
        float(box["xmin"]),
        float(box["ymin"]),
        float(box["xmax"]),
        float(box["ymax"]),
    )


def choose_detection(detections: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Select one detection from the model output.

    The default policy takes the highest-scoring detection after threshold and optional label filtering.
    Users can pass `--detection-index` when they want a specific filtered detection instead.
    """
    filtered = [d for d in detections if float(d.get("score", 0.0)) >= args.det_threshold]
    if args.label:
        want = args.label.lower()
        filtered = [d for d in filtered if str(d.get("label", "")).lower() == want]
    if not filtered:
        raise ValueError("no detection survived threshold/label filters")
    if args.detection_index is not None:
        if args.detection_index < 0 or args.detection_index >= len(filtered):
            raise ValueError(f"--detection-index out of range for {len(filtered)} filtered detections")
        return filtered[args.detection_index]
    return max(filtered, key=lambda d: float(d.get("score", 0.0)))


def depth_to_array(depth_output: Any, size: tuple[int, int]) -> np.ndarray:
    """Normalize HF depth-pipeline output to an `H x W` NumPy array.

    Different Transformers versions may return `predicted_depth` tensors or PIL-like `depth`
    images.  This function deliberately handles both.  If the model output is resized by the
    pipeline, this helper resizes it back to the input image size so bbox crop coordinates line up.
    """
    if isinstance(depth_output, dict):
        if "predicted_depth" in depth_output:
            arr = depth_output["predicted_depth"]
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            arr = np.asarray(arr, dtype=np.float64)
            arr = np.squeeze(arr)
        elif "depth" in depth_output:
            arr = np.asarray(depth_output["depth"], dtype=np.float64)
        else:
            raise ValueError(f"depth pipeline output has no predicted_depth/depth keys: {depth_output.keys()}")
    else:
        arr = np.asarray(depth_output, dtype=np.float64)

    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != (size[1], size[0]):
        pil = Image.fromarray(arr.astype(np.float32), mode="F")
        arr = np.asarray(pil.resize(size, Image.Resampling.BILINEAR), dtype=np.float64)
    return arr


def robust_depth_range(depth: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return a stable positive depth interval for the detected crop.

    Depth Anything V2's default small HF checkpoint is a relative-depth model.  The exporter avoids
    interpreting the values as meters.  For this certificate, any positive projective depth scale is
    enough: Lean checks that the exported 3D corners are internally coherent under the exported
    camera matrix, not that the monocular depth estimate is metric ground truth.
    """
    h, w = depth.shape
    xmin, ymin, xmax, ymax = bbox
    x0 = max(0, min(w - 1, int(math.floor(xmin))))
    y0 = max(0, min(h - 1, int(math.floor(ymin))))
    x1 = max(x0 + 1, min(w, int(math.ceil(xmax))))
    y1 = max(y0 + 1, min(h, int(math.ceil(ymax))))
    crop = depth[y0:y1, x0:x1]
    finite = crop[np.isfinite(crop)]
    if finite.size == 0:
        raise ValueError("depth crop has no finite values")
    lo = float(np.quantile(finite, 0.20))
    hi = float(np.quantile(finite, 0.80))
    # Depth Anything V2 may be relative, but the geometric checker only needs positive scale.
    lo = max(0.1, lo)
    hi = max(lo + 0.1, hi)
    return lo, hi


def backproject(u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float) -> list[float]:
    """Backproject one pixel/depth pair through a pinhole camera.

    Formula:

    `X = (u - cx) * z / fx`, `Y = (v - cy) * z / fy`, `Z = z`.

    This is ordinary camera geometry, but it is still only Python-side production.  The Lean checker
    reprojects the resulting `[X,Y,Z]` values and checks the projection contract.
    """
    return [(u - cx) * z / fx, (v - cy) * z / fy, z]


def corners_from_bbox_depth(
    bbox: tuple[float, float, float, float],
    z_near: float,
    z_far: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> list[float]:
    """Backproject the 2D detection box into a conservative 8-corner camera-space frustum.

    The four 2D box corners are backprojected at two depths.  By construction, projecting these
    eight points through the same pinhole camera should recover the 2D box corners.  The Lean
    checker catches exporter bugs such as wrong intrinsics ordering, negative/zero depth, swapped
    coordinates, malformed tensor lengths, or a claimed bbox that does not enclose the projection.
    """
    xmin, ymin, xmax, ymax = bbox
    pixels = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]
    corners: list[float] = []
    for z in (z_near, z_far):
        for u, v in pixels:
            corners.extend(backproject(u, v, z, fx, fy, cx, cy))
    return corners


def export_cert_with_pipelines(
    args: argparse.Namespace,
    detector: Any | None = None,
    depth_estimator: Any | None = None,
) -> dict[str, Any]:
    """Run models, build one certificate dictionary, and return it without writing to disk.

    The returned dictionary has the exact schema consumed by `NN.Verification.Geometry3D.Box3D`.
    All model-dependent choices live here: detection selection, relative-depth interval selection,
    fallback camera intrinsics, and frustum corner construction.  None of those choices are trusted;
    they are merely the candidate artifact submitted to Lean.
    """
    image = load_image(args.image, args.image_url)
    width, height = image.size
    device = pipeline_device(args.device)

    if detector is None:
        detector = pipeline("object-detection", model=args.detector_model, device=device)
    if depth_estimator is None:
        depth_estimator = pipeline("depth-estimation", model=args.depth_model, device=device)

    detections = detector(image)
    if not isinstance(detections, list):
        raise ValueError("object detector did not return a detection list")
    det = choose_detection(detections, args)
    bbox = detection_box(det)

    depth_output = depth_estimator(image)
    depth = depth_to_array(depth_output, image.size)
    z_near, z_far = robust_depth_range(depth, bbox)

    fx = args.focal_length if args.focal_length is not None else float(max(width, height))
    fy = args.focal_y if args.focal_y is not None else fx
    cx = args.principal_x if args.principal_x is not None else width / 2.0
    cy = args.principal_y if args.principal_y is not None else height / 2.0

    camera_p = [
        fx, 0.0, cx, 0.0,
        0.0, fy, cy, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]
    corners = corners_from_bbox_depth(bbox, z_near, z_far, fx, fy, cx, cy)

    return {
        "format": FORMAT,
        "source": "HF object detection + Depth Anything V2 real-image exporter",
        "image_width": float(width),
        "image_height": float(height),
        "tol": float(args.tol),
        "camera_P": camera_p,
        "corners3d": corners,
        "bbox2d": [float(x) for x in bbox],
        "metadata": {
            "producer": "scripts/verification/geometry3d/export_hf_depth_box3d_cert.py",
            "detector_model": args.detector_model,
            "depth_model": args.depth_model,
            "label": det.get("label"),
            "score": float(det.get("score", 0.0)),
            "z_near": z_near,
            "z_far": z_far,
            "camera_intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "image": str(args.image) if args.image is not None else args.image_url or DEFAULT_IMAGE_URL,
        },
    }


def export_cert(args: argparse.Namespace) -> dict[str, Any]:
    """Single-image wrapper that creates model pipelines internally."""
    return export_cert_with_pipelines(args)


def args_with_case(args: argparse.Namespace, case: dict[str, Any], index: int) -> argparse.Namespace:
    """Overlay one manifest case onto CLI defaults.

    Manifest files are deliberately small and boring JSON so users can add their own real-image
    examples without touching Python.  Each case can set `image`, `image_url`, `label`,
    `det_threshold`, `focal_length`, `tol`, and `out`.
    """
    case_args = argparse.Namespace(**vars(args))
    if "image" in case and case["image"] is not None:
        case_args.image = Path(case["image"])
        case_args.image_url = None
    if "image_url" in case:
        case_args.image = None
        case_args.image_url = case["image_url"]
    if "label" in case:
        case_args.label = case["label"]
    if "det_threshold" in case:
        case_args.det_threshold = float(case["det_threshold"])
    if "tol" in case:
        case_args.tol = float(case["tol"])
    if "focal_length" in case:
        case_args.focal_length = float(case["focal_length"])
    if "focal_y" in case:
        case_args.focal_y = float(case["focal_y"])
    if "principal_x" in case:
        case_args.principal_x = float(case["principal_x"])
    if "principal_y" in case:
        case_args.principal_y = float(case["principal_y"])
    if "detection_index" in case:
        value = case["detection_index"]
        case_args.detection_index = None if value is None else int(value)
    if "out" in case:
        case_args.out = Path(case["out"])
    else:
        stem = str(case.get("name", f"case_{index}")).replace("/", "_").replace(" ", "_")
        case_args.out = args.batch_out_dir / f"{stem}.json"
    return case_args


def run_batch(args: argparse.Namespace) -> None:
    """Run one detector/depth-model load over many real-image cases.

    This avoids reloading model weights for every manifest entry.  If `--verify` is set, every
    generated certificate is immediately checked by the Lean CLI, so failures surface at the exact
    producer/checker boundary.
    """
    with args.batch_manifest.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    cases = manifest.get("cases") if isinstance(manifest, dict) else manifest
    if not isinstance(cases, list):
        raise ValueError("batch manifest must be a list or an object with a 'cases' list")

    device = pipeline_device(args.device)
    detector = pipeline("object-detection", model=args.detector_model, device=device)
    depth_estimator = pipeline("depth-estimation", model=args.depth_model, device=device)

    written: list[Path] = []
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case {i}: expected object")
        case_args = args_with_case(args, case, i)
        cert = export_cert_with_pipelines(case_args, detector, depth_estimator)
        case_args.out.parent.mkdir(parents=True, exist_ok=True)
        with case_args.out.open("w", encoding="utf-8") as fh:
            json.dump(cert, fh, indent=2)
            fh.write("\n")
        written.append(case_args.out)
        label = cert.get("metadata", {}).get("label")
        score = cert.get("metadata", {}).get("score")
        print(f"wrote {case_args.out}  label={label} score={score}", flush=True)

    if args.verify:
        for path in written:
            subprocess.run(["lake", "exe", "verify", "--", "camera-box3d-cert", str(path)], check=True)


def main() -> None:
    """Command-line entrypoint for single-image and batch Geometry3D exports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None, help="local image path")
    parser.add_argument("--image-url", default=None, help="image URL; default: COCO val image")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="TorchLean cert output path")
    parser.add_argument("--batch-manifest", type=Path, default=None, help="JSON manifest of real-image cases")
    parser.add_argument("--batch-out-dir", type=Path, default=Path("_external/geometry3d/realworld"))
    parser.add_argument("--detector-model", default="facebook/detr-resnet-50")
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a pipeline device")
    parser.add_argument("--det-threshold", type=float, default=0.7)
    parser.add_argument("--label", default=None, help="optional detector label filter, e.g. cat")
    parser.add_argument("--detection-index", type=int, default=None)
    parser.add_argument("--tol", type=float, default=2.0, help="pixel tolerance for projection enclosure")
    parser.add_argument("--focal-length", type=float, default=None, help="fx fallback; default max(width,height)")
    parser.add_argument("--focal-y", type=float, default=None, help="fy fallback; default fx")
    parser.add_argument("--principal-x", type=float, default=None, help="cx fallback; default width/2")
    parser.add_argument("--principal-y", type=float, default=None, help="cy fallback; default height/2")
    parser.add_argument("--verify", action="store_true", help="run Lean checker after export")
    args = parser.parse_args()

    if args.batch_manifest is not None:
        run_batch(args)
        return

    cert = export_cert(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(cert, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.out}", flush=True)

    if args.verify:
        subprocess.run(["lake", "exe", "verify", "--", "camera-box3d-cert", str(args.out)], check=True)


if __name__ == "__main__":
    main()
