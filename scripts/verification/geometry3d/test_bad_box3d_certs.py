#!/usr/bin/env python3
"""Generate deliberately invalid 3D-box certificates and require Lean to reject them.

This is the regression test for the "glue bugs" workflow. The good fixture is copied, mutated in
ways that resemble real projection-pipeline mistakes, and then sent to:

    lake exe verify -- camera-box3d-cert <bad.json>

The script succeeds only when every bad artifact is rejected.  Python is not the verifier here; it
is merely a mischievous artifact generator.  Lean remains the checker.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any, Callable


DEFAULT_GOOD = Path("NN/Verification/Geometry3D/smoke_box3d_camera_cert.json")
DEFAULT_OUT_DIR = Path("_external/geometry3d/bad")


Mutation = Callable[[dict[str, Any]], None]


# Real bug reports / docs motivating each mutation. Keeping the links next to the
# tests makes the purpose of each invalid artifact visible during maintenance.
BUG_REFERENCES: dict[str, list[str]] = {
    "negative_depth": [
        # Negative-Z / camera-convention mismatch can make otherwise valid-looking geometry vanish.
        "https://github.com/facebookresearch/pytorch3d/issues/1427",
        # Blender/PyTorch3D camera conversions commonly trip over view direction and axis signs.
        "https://github.com/facebookresearch/pytorch3d/issues/1105",
    ],
    "wrong_principal_point": [
        # OpenCV/KITTI/PyTorch3D camera conversion and screen/NDC mismatches.
        "https://github.com/facebookresearch/pytorch3d/issues/596",
        # OpenCV `K,R,t` to PyTorch3D conversion trouble.
        "https://github.com/facebookresearch/pytorch3d/issues/522",
    ],
    "bad_bbox_enclosure": [
        # 3D bbox vertices projected into the wrong image location.
        "https://github.com/DLR-RM/BlenderProc/issues/1150",
        # Omni3D/Cube R-CNN conversion to KITTI involved bbox/dimension/center consistency gaps.
        "https://github.com/facebookresearch/omni3d/issues/60",
    ],
    "swapped_bbox_axes": [
        # PyTorch3D transform composition/layout confusion.
        "https://github.com/facebookresearch/pytorch3d/issues/1183",
        # Detectron2 bbox mode conventions are explicit because xyxy/xywh mistakes are easy.
        "https://detectron2.readthedocs.io/en/v0.4.1/_modules/detectron2/structures/boxes.html",
    ],
    "malformed_corners": [
        # Detectron2 rotated boxes triggered Nx5-vs-Nx4 shape/layout mismatch.
        "https://github.com/facebookresearch/detectron2/issues/2402",
    ],
    "malformed_camera": [
        # Matrix layout/transpose mistakes in PyTorch3D projection composition.
        "https://github.com/facebookresearch/pytorch3d/issues/1183",
    ],
}


def load_cert(path: Path) -> dict[str, Any]:
    """Load a JSON certificate as a mutable Python dictionary."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def write_cert(path: Path, cert: dict[str, Any]) -> None:
    """Write one mutated certificate with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(cert, fh, indent=2)
        fh.write("\n")


def mutate_negative_depth(cert: dict[str, Any]) -> None:
    """Put one 3D corner behind the camera.

    This simulates camera-convention bugs such as using a renderer convention where visible points
    have negative `Z`, then feeding that artifact to a checker expecting positive camera depth.
    Lean should reject through `PositiveDepths`.

    Motivating reports:
    - https://github.com/facebookresearch/pytorch3d/issues/1427
    - https://github.com/facebookresearch/pytorch3d/issues/1105
    """
    cert["source"] = "bad fixture: negative depth"
    cert["corners3d"][2] = -abs(float(cert["corners3d"][2]))


def mutate_wrong_principal_point(cert: dict[str, Any]) -> None:
    """Move `cx` far outside the image in the projection matrix.

    This resembles wrong intrinsics, wrong image scaling, or OpenCV/PyTorch3D camera conversion
    bugs.  The projected corners should leave the image and/or claimed bbox.

    Motivating reports:
    - https://github.com/facebookresearch/pytorch3d/issues/596
    - https://github.com/facebookresearch/pytorch3d/issues/522
    """
    cert["source"] = "bad fixture: wrong camera principal point"
    cert["camera_P"][2] = 10000.0


def mutate_bad_bbox_enclosure(cert: dict[str, Any]) -> None:
    """Shrink the 2D box so it no longer encloses the projected 3D corners.

    Motivating reports:
    - https://github.com/DLR-RM/BlenderProc/issues/1150
    - https://github.com/facebookresearch/omni3d/issues/60
    """
    cert["source"] = "bad fixture: bbox does not enclose projection"
    cert["bbox2d"] = [320.0, 240.0, 321.0, 241.0]


def mutate_swapped_bbox_axes(cert: dict[str, Any]) -> None:
    """Swap x/y-style coordinates while keeping the box ordered and in frame.

    This catches the common `xyxy`, `yxyx`, row/column, or image-axis confusion class of bugs.

    Motivating reports:
    - https://github.com/facebookresearch/pytorch3d/issues/1183
    - https://detectron2.readthedocs.io/en/v0.4.1/_modules/detectron2/structures/boxes.html
    """
    cert["source"] = "bad fixture: swapped bbox axes"
    cert["bbox2d"] = [229.0, 309.0, 251.0, 331.0]


def mutate_malformed_corners(cert: dict[str, Any]) -> None:
    """Drop one scalar from the `8 x 3` corner tensor.

    Motivating report:
    - https://github.com/facebookresearch/detectron2/issues/2402
    """
    cert["source"] = "bad fixture: malformed corners tensor"
    cert["corners3d"] = cert["corners3d"][:-1]


def mutate_malformed_camera(cert: dict[str, Any]) -> None:
    """Drop one scalar from the `3 x 4` projection matrix.

    Motivating report:
    - https://github.com/facebookresearch/pytorch3d/issues/1183
    """
    cert["source"] = "bad fixture: malformed camera tensor"
    cert["camera_P"] = cert["camera_P"][:-1]


MUTATIONS: dict[str, Mutation] = {
    "negative_depth": mutate_negative_depth,
    "wrong_principal_point": mutate_wrong_principal_point,
    "bad_bbox_enclosure": mutate_bad_bbox_enclosure,
    "swapped_bbox_axes": mutate_swapped_bbox_axes,
    "malformed_corners": mutate_malformed_corners,
    "malformed_camera": mutate_malformed_camera,
}


def lean_accepts(path: Path) -> tuple[bool, str]:
    """Run the Lean camera-box checker and report whether it accepted."""
    proc = subprocess.run(
        ["lake", "exe", "verify", "--", "camera-box3d-cert", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode == 0, proc.stdout


def run_bad_suite(good_path: Path, out_dir: Path) -> None:
    """Generate every mutation and assert that Lean rejects it."""
    good = load_cert(good_path)
    failures: list[str] = []
    for name, mutate in MUTATIONS.items():
        refs = BUG_REFERENCES.get(name, [])
        if refs:
            print(f"[bad-cert] testing {name}; refs: {', '.join(refs)}")
        cert = copy.deepcopy(good)
        mutate(cert)
        out = out_dir / f"{name}.json"
        write_cert(out, cert)
        accepted, output = lean_accepts(out)
        if accepted:
            failures.append(name)
            print(f"[bad-cert] UNEXPECTED ACCEPT: {name} -> {out}")
        else:
            last_line = output.strip().splitlines()[-1] if output.strip() else "rejected"
            print(f"[bad-cert] rejected {name}: {last_line}")
    if failures:
        joined = ", ".join(failures)
        raise SystemExit(f"Lean accepted deliberately invalid cert(s): {joined}")


def main() -> None:
    """Command-line entrypoint for bad-certificate regression tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--good", type=Path, default=DEFAULT_GOOD, help="known-good cert to mutate")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="where bad certs are written")
    args = parser.parse_args()
    run_bad_suite(args.good, args.out_dir)


if __name__ == "__main__":
    main()
