#!/usr/bin/env python3
"""Run Hugging Face vision pipelines and export a TorchLean 3D-box certificate.

The PyTorch-to-Lean Geometry3D path has four steps:

1. run an object detector on an ordinary image,
2. run Depth Anything V2 on the same image,
3. lift one detected 2D box into a conservative 3D frustum/cuboid using the depth crop, and
4. with `--verify`, ask Lean to check that the exported 3D corners project back inside the
   claimed 2D box.

The learned models are untrusted. The certificate says only:

    Given the exported camera matrix, 3D corners, image size, and 2D box, the geometry is coherent.

At the trust boundary, PyTorch proposes the geometry and TorchLean checks the contract before
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


FORMAT = "torchlean.camera.box3d.v2"
DEFAULT_IMAGE_URL = (
    "https://images.cocodataset.org/val2017/000000039769.jpg"
)
DEFAULT_OUT = Path("_external/geometry3d/hf_depth_box3d_cert.json")
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_image(path: Path | None, url: str | None) -> Image.Image:
    """Load a local or remote RGB image (the default COCO image when neither is given).

    Pixels and model inference stay on the producer side. Lean sees only the exported numeric
    certificate.
    """
    if path is not None:
        return load_local_rgb_image(path)
    if url is None:
        url = DEFAULT_IMAGE_URL
    return load_remote_rgb_image(url)


def pipeline_device(device: str) -> int | str:
    """Translate a friendly device string into the Hugging Face pipeline convention.

    HF pipelines use `-1` for CPU and integer GPU IDs for CUDA. `cpu`, `cuda` and `cuda:N` map
    to those IDs, `auto` picks GPU 0 when CUDA is available, and any other string passes through.
    """
    if device == "auto":
        return 0 if torch.cuda.is_available() else -1
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        index = device.removeprefix("cuda:")
        if not index.isascii() or not index.isdecimal():
            raise ValueError(f"invalid CUDA device: {device!r}")
        return int(index)
    return device


def detection_box(det: dict[str, Any]) -> tuple[float, float, float, float]:
    """Extract `[xmin, ymin, xmax, ymax]` from one HF object-detection result.

    The detector output is untrusted. The function checks only that the box holds four finite,
    ordered coordinates, and Lean later checks whether it encloses the projected 3D corners.
    """
    box = det.get("box")
    if not isinstance(box, dict):
        raise ValueError(f"detection has no box dict: {det}")
    bbox = (
        float(box["xmin"]),
        float(box["ymin"]),
        float(box["xmax"]),
        float(box["ymax"]),
    )
    if not all(math.isfinite(x) for x in bbox) or bbox[0] > bbox[2] or bbox[1] > bbox[3]:
        raise ValueError("detection box must be finite and ordered")
    return bbox


def choose_detection(detections: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Select one detection from the model output.

    Detections below `--det-threshold` (or with a non-finite score) are dropped, and so are those
    with another label when `--label` is given. The highest-scoring survivor wins unless
    `--detection-index` picks one of the survivors.
    """
    if not math.isfinite(args.det_threshold):
        raise ValueError("detection threshold must be finite")
    filtered = [d for d in detections if math.isfinite(float(d.get("score", 0.0)))
                and float(d.get("score", 0.0)) >= args.det_threshold]
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

    Transformers versions differ: some return `predicted_depth` tensors, others PIL-like `depth`
    images, and both are handled. A depth map whose size differs from the input image is resized
    back to it so bbox crop coordinates line up.
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
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError("depth output must be a nonempty image")
    if arr.shape != (size[1], size[0]):
        pil = Image.fromarray(arr.astype(np.float32), mode="F")
        arr = np.asarray(pil.resize(size, Image.Resampling.BILINEAR), dtype=np.float64)
    return arr


def robust_depth_range(depth: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return a positive depth interval for the detected crop.

    The interval runs from the 20th to the 80th percentile of the crop's finite depths, with
    `lo >= 0.1` and `hi >= lo + 0.1`. Depth Anything V2's default small HF checkpoint predicts
    relative depth, so the values are not meters. Any positive projective scale is enough, because
    Lean checks that the exported corners are coherent under the exported camera matrix.
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

    The Lean checker reprojects the resulting `[X,Y,Z]` values and checks the projection contract.
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


def export_cert(
    args: argparse.Namespace,
    detector: Any | None = None,
    depth_estimator: Any | None = None,
) -> dict[str, Any]:
    """Run the models and return one certificate dictionary without writing it to disk.

    The dictionary has the schema that `NN.Verification.Geometry3D.Box3D` reads. Every
    model-dependent choice lives here: detection selection, the relative-depth interval, default
    camera intrinsics and frustum corners. None of them is trusted, since together they form the
    candidate artifact that Lean checks.
    """
    image = load_image(args.image, args.image_url)
    width, height = image.size
    device = pipeline_device(args.device)
    if not math.isfinite(args.tol) or args.tol < 0:
        raise ValueError("tolerance must be finite and nonnegative")

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
    if not all(math.isfinite(x) for x in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ValueError("intrinsics must be finite with positive focal lengths")

    camera_p = [
        fx, 0.0, cx, 0.0,
        0.0, fy, cy, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]
    corners = corners_from_bbox_depth(bbox, z_near, z_far, fx, fy, cx, cy)

    return {
        "format": FORMAT,
        "point_count": 8,
        "source": "HF object detection + Depth Anything V2 real-image exporter",
        "image_width": float(width),
        "image_height": float(height),
        "tol": float(args.tol),
        "camera_P": camera_p,
        "corners3d": corners,
        "bbox2d": list(bbox),
        "metadata": {
            "producer": "TorchLeanExamples/Verification/Geometry3D/export_hf_depth_box3d_cert.py",
            "corner_order": "binary",
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


def args_with_case(args: argparse.Namespace, case: dict[str, Any], index: int) -> argparse.Namespace:
    """Overlay one manifest case onto the command-line arguments.

    Manifests are small, plain JSON, so users can add their own real-image examples without
    touching Python. Each case can set `image`, `image_url`, `label`, `det_threshold`, `tol`,
    `focal_length`, `focal_y`, `principal_x`, `principal_y`, `detection_index` and `out`. Without
    `out`, the certificate goes to `<name>.json` (or `case_<index>.json`) in `--batch-out-dir`.
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
    for name in ("det_threshold", "tol", "focal_length", "focal_y", "principal_x", "principal_y"):
        if name in case:
            setattr(case_args, name, float(case[name]))
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

    The models are loaded once. With `--verify`, the Lean CLI checks each certificate after
    all cases have been exported.
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
        cert = export_cert(case_args, detector, depth_estimator)
        case_args.out.parent.mkdir(parents=True, exist_ok=True)
        with case_args.out.open("w", encoding="utf-8") as fh:
            json.dump(cert, fh, indent=2, allow_nan=False)
            fh.write("\n")
        written.append(case_args.out)
        label = cert.get("metadata", {}).get("label")
        score = cert.get("metadata", {}).get("score")
        print(f"wrote {case_args.out}  label={label} score={score}", flush=True)

    if args.verify:
        for path in written:
            subprocess.run([str(REPO_ROOT / "scripts/lake.sh"), "exe", "verify", "--",
                            "camera-box3d-cert", str(path.resolve())], cwd=REPO_ROOT, check=True)


def main() -> None:
    """Command-line entrypoint for single-image and batch Geometry3D exports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None, help="local image path")
    parser.add_argument("--image-url", default=None, help="image URL (default: COCO val image)")
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
    parser.add_argument("--focal-length", type=float, default=None,
                        help="focal length fx (default: max(width, height))")
    parser.add_argument("--focal-y", type=float, default=None, help="focal length fy (default: fx)")
    parser.add_argument("--principal-x", type=float, default=None,
                        help="principal point cx (default: width/2)")
    parser.add_argument("--principal-y", type=float, default=None,
                        help="principal point cy (default: height/2)")
    parser.add_argument("--verify", action="store_true", help="run Lean checker after export")
    args = parser.parse_args()

    if args.batch_manifest is not None:
        run_batch(args)
        return

    cert = export_cert(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(cert, fh, indent=2, allow_nan=False)
        fh.write("\n")
    print(f"wrote {args.out}", flush=True)

    if args.verify:
        subprocess.run([str(REPO_ROOT / "scripts/lake.sh"), "exe", "verify", "--",
                        "camera-box3d-cert", str(args.out.resolve())], cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
