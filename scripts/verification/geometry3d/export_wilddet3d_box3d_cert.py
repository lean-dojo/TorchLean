#!/usr/bin/env python3
"""Run WildDet3D and export a TorchLean 3D camera-box certificate.

WildDet3D is a promptable monocular 3D detection model hosted by Ai2 on Hugging Face.  Unlike the
DETR+Depth Anything bridge, WildDet3D directly predicts 3D boxes.  This script treats WildDet3D as
an untrusted external producer:

1. download the WildDet3D Hugging Face Space source and model checkpoint;
2. run one text-prompt monocular 3D detection pass;
3. convert the selected predicted 3D box to eight camera-frame corners;
4. export a `torchlean.camera.box3d.v1` JSON artifact; and
5. optionally ask Lean to verify the exported projection contract.

Lean does not trust WildDet3D, PyTorch, vis4d, or this exporter.  Lean only checks the final JSON:
positive depth, projection into the image, and 2D bbox enclosure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

from safe_image_io import load_local_rgb_image, load_remote_rgb_image


FORMAT = "torchlean.camera.box3d.v1"
HF_MODEL_REPO = "allenai/WildDet3D"
HF_SPACE_REPO = "allenai/WildDet3D"
HF_CKPT_NAME = "wilddet3d_alldata_all_prompt_v1.0.pt"
DEFAULT_IMAGE_URL = "https://images.cocodataset.org/val2017/000000039769.jpg"
DEFAULT_OUT = Path("_external/geometry3d/wilddet3d/wilddet3d_box3d_cert.json")


def load_image(path: Path | None, url: str | None) -> Image.Image:
    """Load an RGB image from disk or URL."""
    if path is not None:
        return load_local_rgb_image(path)
    return load_remote_rgb_image(url or DEFAULT_IMAGE_URL)


def prepare_wilddet3d_source() -> Path:
    """Download the WildDet3D Space source and make its packages importable."""
    snapshot = Path(
        snapshot_download(
            HF_SPACE_REPO,
            repo_type="space",
            ignore_patterns=[
                "**/__pycache__/**",
                "*.pyc",
                ".git/**",
            ],
        )
    )
    sys.path.insert(0, str(snapshot))
    sys.path.insert(0, str(snapshot / "third_party" / "lingbot_depth"))
    sys.path.insert(0, str(snapshot / "third_party" / "sam3"))
    return snapshot


def download_checkpoint() -> str:
    """Download the WildDet3D checkpoint from Hugging Face."""
    return hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=HF_CKPT_NAME,
        token=os.environ.get("HF_TOKEN"),
    )


def scale_intrinsics_to_original(K: torch.Tensor, input_hw: tuple[int, int], original_hw: tuple[int, int]) -> np.ndarray:
    """Scale model-input intrinsics back to original image resolution."""
    K_np = K.detach().cpu().numpy()
    if K_np.ndim == 3:
        K_np = K_np[0]
    in_h, in_w = input_hw
    orig_h, orig_w = original_hw
    scaled = K_np.astype(np.float64).copy()
    scaled[0, :] *= float(orig_w) / float(in_w)
    scaled[1, :] *= float(orig_h) / float(in_h)
    return scaled


def model_intrinsics_for_export(data: dict[str, Any], predicted_K: torch.Tensor | None) -> np.ndarray:
    """Choose the camera intrinsics used for exporting/checking predicted boxes.

    WildDet3D can predict intrinsics when no real camera calibration is provided.  If predicted
    intrinsics are returned, this exporter scales them to the original image resolution. Otherwise it
    use the preprocessed original intrinsics recorded by WildDet3D's data pipeline.
    """
    if predicted_K is not None:
        return scale_intrinsics_to_original(
            predicted_K,
            input_hw=data["input_hw"],
            original_hw=data["original_hw"],
        )
    K = data["original_intrinsics"].detach().cpu().numpy()
    if K.ndim == 3:
        K = K[0]
    return K.astype(np.float64)


def choose_prediction(
    boxes: torch.Tensor,
    boxes3d: torch.Tensor,
    scores: torch.Tensor,
    scores_2d: torch.Tensor,
    scores_3d: torch.Tensor,
    class_ids: torch.Tensor,
    min_score: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float, int]:
    """Select the highest-scoring predicted 3D detection above threshold."""
    if len(boxes) == 0:
        raise ValueError("WildDet3D returned no detections")
    mask = scores >= min_score
    if not bool(mask.any()):
        best = int(scores.argmax().item())
    else:
        masked_indices = torch.nonzero(mask, as_tuple=False).flatten()
        best = int(masked_indices[scores[masked_indices].argmax()].item())
    return (
        boxes[best].detach().cpu(),
        boxes3d[best].detach().cpu(),
        float(scores[best].detach().cpu()),
        float(scores_2d[best].detach().cpu()),
        float(scores_3d[best].detach().cpu()),
        int(class_ids[best].detach().cpu()),
    )


def corners_from_wilddet3d_box(box3d: torch.Tensor) -> list[float]:
    """Convert one WildDet3D `(10,)` box to flat row-major eight camera-frame corners.

    WildDet3D decodes boxes as `[center_x, center_y, center_z, W, L, H, qw, qx, qy, qz]`
    in OpenCV camera coordinates.  The model's own evaluation code feeds this format to
    `vis4d.op.box.box3d.boxes3d_to_corners(..., AxisMode.OPENCV)`.  This exporter implements the same
    mathematical conversion locally instead of importing that helper, because the PyPI `vis4d`
    package changes APIs across releases and is not part of the checked trust boundary.

    The corner order matches the Vis4D drawing convention:
    bottom/top rectangles in local `(x, y, z) = (±L/2, ±H/2, ±W/2)` coordinates, rotated by the
    quaternion and translated by the camera-frame center.  Lean's checker is order-insensitive for
    the projection contract, but keeping a stable order makes the overlay edges understandable.
    """
    b = box3d.detach().cpu().double().numpy()
    if b.shape[0] < 10:
        raise ValueError(f"expected a WildDet3D 10D box, got shape {b.shape}")

    center = b[:3]
    width, length, height = b[3:6]
    qw, qx, qy, qz = b[6:10]
    norm2 = qw * qw + qx * qx + qy * qy + qz * qz
    if norm2 <= 0.0:
        raise ValueError("WildDet3D returned a zero-norm orientation quaternion")
    s = 2.0 / norm2
    rot = np.array(
        [
            [1.0 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw), s * (qx * qz + qy * qw)],
            [s * (qx * qy + qz * qw), 1.0 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
            [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw), 1.0 - s * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )

    local = np.array(
        [
            [ length / 2.0,  height / 2.0,  width / 2.0],
            [ length / 2.0,  height / 2.0, -width / 2.0],
            [-length / 2.0,  height / 2.0, -width / 2.0],
            [-length / 2.0,  height / 2.0,  width / 2.0],
            [ length / 2.0, -height / 2.0,  width / 2.0],
            [ length / 2.0, -height / 2.0, -width / 2.0],
            [-length / 2.0, -height / 2.0, -width / 2.0],
            [-length / 2.0, -height / 2.0,  width / 2.0],
        ],
        dtype=np.float64,
    )
    corners = local @ rot.T + center
    return [float(x) for x in corners.reshape(-1).tolist()]


def project_corners(camera_p: list[float], corners3d: list[float]) -> list[tuple[float, float, float]]:
    """Project flat eight-corner camera-frame coordinates with the exported `3x4` camera matrix.

    The Lean checker repeats this computation on the exported artifact.  The Python version is only
    used to build metadata and, when requested, a projected-footprint bbox; Lean's accept/reject
    result does not rely on trusting this helper.
    """
    projected: list[tuple[float, float, float]] = []
    for i in range(0, len(corners3d), 3):
        x, y, z = corners3d[i:i + 3]
        u_num = camera_p[0] * x + camera_p[1] * y + camera_p[2] * z + camera_p[3]
        v_num = camera_p[4] * x + camera_p[5] * y + camera_p[6] * z + camera_p[7]
        w = camera_p[8] * x + camera_p[9] * y + camera_p[10] * z + camera_p[11]
        projected.append((float(u_num / w), float(v_num / w), float(w)))
    return projected


def projected_envelope_bbox(
    projected: list[tuple[float, float, float]],
    width: float,
    height: float,
    pad: float,
) -> list[float]:
    """Return an image-clipped bbox enclosing the projected 3D corners."""
    xmin = max(0.0, min(u for u, _v, _z in projected) - pad)
    ymin = max(0.0, min(v for _u, v, _z in projected) - pad)
    xmax = min(width, max(u for u, _v, _z in projected) + pad)
    ymax = min(height, max(v for _u, v, _z in projected) + pad)
    return [float(xmin), float(ymin), float(xmax), float(ymax)]


def bbox_encloses(projected: list[tuple[float, float, float]], bbox: list[float], tol: float) -> bool:
    """Check whether a candidate bbox encloses all projected corners up to tolerance."""
    xmin, ymin, xmax, ymax = bbox
    return all(
        xmin - tol <= u <= xmax + tol and ymin - tol <= v <= ymax + tol
        for u, v, _z in projected
    )


def export_cert(args: argparse.Namespace) -> dict[str, Any]:
    """Run WildDet3D and return one TorchLean certificate dictionary."""
    prepare_wilddet3d_source()
    from wilddet3d.inference import build_model
    from wilddet3d.preprocessing import preprocess

    image = load_image(args.image, args.image_url)
    image_np = np.array(image.convert("RGB"))
    data = preprocess(image_np.astype(np.float32), intrinsics=None)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    checkpoint = download_checkpoint()
    detector = build_model(
        checkpoint=checkpoint,
        score_threshold=0.0,
        canonical_rotation=True,
        skip_pretrained=True,
    ).to(device)
    detector.eval()

    prompts = [p.strip() for p in args.text_prompt.split(".") if p.strip()]
    if not prompts:
        prompts = ["object"]

    with torch.no_grad():
        results = detector(
            images=data["images"].to(device),
            intrinsics=data["intrinsics"].to(device)[None],
            input_hw=[data["input_hw"]],
            original_hw=[data["original_hw"]],
            padding=[data["padding"]],
            input_texts=prompts,
            return_predicted_intrinsics=True,
        )

    (
        boxes,
        boxes3d,
        scores,
        scores_2d,
        scores_3d,
        class_ids,
        _depth_maps,
        predicted_K,
        _confidence_maps,
    ) = results

    box2d, box3d, score, score_2d, score_3d, class_id = choose_prediction(
        boxes[0], boxes3d[0], scores[0], scores_2d[0], scores_3d[0], class_ids[0], args.min_score
    )
    K = model_intrinsics_for_export(data, predicted_K)
    camera_p = [
        float(K[0, 0]), float(K[0, 1]), float(K[0, 2]), 0.0,
        float(K[1, 0]), float(K[1, 1]), float(K[1, 2]), 0.0,
        float(K[2, 0]), float(K[2, 1]), float(K[2, 2]), 0.0,
    ]

    width, height = image.size
    label = prompts[class_id] if 0 <= class_id < len(prompts) else str(class_id)
    corners3d = corners_from_wilddet3d_box(box3d)
    projected = project_corners(camera_p, corners3d)
    model_bbox2d = [float(x) for x in box2d.tolist()]
    footprint_bbox2d = projected_envelope_bbox(
        projected,
        width=float(width),
        height=float(height),
        pad=float(args.envelope_pad),
    )
    if args.bbox_source == "model2d":
        bbox2d = model_bbox2d
    elif args.bbox_source == "projected-envelope":
        bbox2d = footprint_bbox2d
    else:
        bbox2d = model_bbox2d if bbox_encloses(projected, model_bbox2d, args.tol) else footprint_bbox2d

    return {
        "format": FORMAT,
        "source": "WildDet3D monocular 3D detector exporter",
        "image_width": float(width),
        "image_height": float(height),
        "tol": float(args.tol),
        "camera_P": camera_p,
        "corners3d": corners3d,
        "bbox2d": bbox2d,
        "metadata": {
            "producer": "scripts/verification/geometry3d/export_wilddet3d_box3d_cert.py",
            "model": HF_MODEL_REPO,
            "space_source": HF_SPACE_REPO,
            "text_prompt": args.text_prompt,
            "selected_label": label,
            "score": score,
            "score_2d": score_2d,
            "score_3d": score_3d,
            "box3d_10d": [float(x) for x in box3d.tolist()],
            "bbox_source": args.bbox_source,
            "exported_bbox2d": bbox2d,
            "model_bbox2d": model_bbox2d,
            "projected_envelope_bbox2d": footprint_bbox2d,
            "model_bbox_encloses_projected_corners": bbox_encloses(projected, model_bbox2d, args.tol),
            "projected_corners2d": [[u, v] for u, v, _z in projected],
            "camera_intrinsics": {
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
            },
            "image": str(args.image) if args.image is not None else args.image_url or DEFAULT_IMAGE_URL,
        },
    }


def main() -> None:
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None, help="local image path")
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--text-prompt", default="cat", help='text prompt, e.g. "car.person.chair"')
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--tol", type=float, default=4.0)
    parser.add_argument(
        "--bbox-source",
        choices=("auto", "model2d", "projected-envelope"),
        default="auto",
        help=(
            "`model2d` checks WildDet3D's 2D detection box against the projected 3D box. "
            "`projected-envelope` exports a bbox derived from the 3D box projection. "
            "`auto` uses the model bbox if it already encloses the projection, otherwise the "
            "projected envelope and records the mismatch in metadata."
        ),
    )
    parser.add_argument("--envelope-pad", type=float, default=2.0)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--overlay", action="store_true", help="render a PNG overlay after export")
    args = parser.parse_args()

    cert = export_cert(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(cert, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.out}", flush=True)

    if args.verify:
        subprocess.run(["lake", "exe", "verify", "--", "camera-box3d-cert", str(args.out)], check=True)

    if args.overlay:
        overlay = args.out.with_suffix(".png")
        subprocess.run(
            [
                "python3",
                "scripts/verification/geometry3d/render_box3d_cert_overlay.py",
                "--cert",
                str(args.out),
                "--out-dir",
                str(overlay.parent),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
