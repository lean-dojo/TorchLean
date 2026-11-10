# Camera-box certificates

A camera-box certificate claims that 3D points, projected through a 3×4 camera matrix, land in the
image and inside a stated 2D bounding box. Lean recomputes each projection and compares it with the
claim. Our exporters build such certificates from real images with learned detectors and depth
models, and the checker trusts none of them.

- `Box3D.lean`: `BoxCameraCert`, the Boolean checker `checkCert`, the property `Verified`,
  `parseJsonCert` for the format `torchlean.camera.box3d.v2`, and the theorems below.
- `CLI.lean`: `camera-box3d-cert`, which takes a certificate path or `--cert=PATH`.
- `check_box3d_camera_cert.json`: eight cuboid corners at `x, y = ±1` and depths 10 and 12, seen by
  a camera with focal length 100 and principal point (320, 240) in a 640×480 image. They project
  into [310, 330] × [230, 250], and the claimed box is [309, 229, 331, 251] with `tol` 0.
- `export_hf_depth_box3d_cert.py`: DETR (`facebook/detr-resnet-50`) finds an object, Depth
  Anything V2 (`depth-anything/Depth-Anything-V2-Small-hf`) estimates depth, and the script lifts
  the 2D box into a 3D frustum. Its default image is COCO val2017 image 39769, with `--tol 2.0`.
- `export_omni3d_box3d_cert.py`: converts a Cube R-CNN or Omni3D prediction JSON
  (`--prediction-json PATH`).
- `export_wilddet3d_box3d_cert.py`: runs the WildDet3D monocular detector on a text prompt
  (`--text-prompt`, default `cat`, with `--tol 4.0`).
- `render_box3d_cert_overlay.py`: draws certificates over their images or as interval plots.
- `safe_image_io.py`: local and HTTPS image loading for the scripts, with a 20 s timeout and a
  64 MiB cap on downloads.
- `realworld_manifest.json`: three COCO val2017 images for the DETR exporter's `--batch-manifest`.
- `requirements-realworld.txt`, `requirements-wilddet3d.txt`: the Python packages of the exporters.

## Commands

```bash
scripts/lake.sh exe verify -- camera-box3d-cert
python3 TorchLeanExamples/Verification/Geometry3D/export_hf_depth_box3d_cert.py --verify
```

`camera-box3d-cert` verifies the bundled certificate and prints `3D camera-box certificate verified`
with its path. `export_hf_depth_box3d_cert.py` downloads the image and both models, writes
`_external/geometry3d/hf_depth_box3d_cert.json`, and with `--verify` runs `camera-box3d-cert` on
it. All three exporters take `--out PATH` and `--verify`. Rendered overlays are PNGs under
`_external/geometry3d/overlays`. Pass `--view intervals --cert PATH --out PATH` to plot the model,
projected and exported boxes from a WildDet3D certificate's metadata, without running the Lean
checker. In its default overlay view the renderer does run it, unless `--no-verify` is given.

## What acceptance means

`checkCert_sound` proves that an artifact the Boolean checker accepts satisfies `Verified`:
the image size is positive, `tol` is nonnegative, and the box is ordered and inside the image. Every
point has positive depth and projects into the image, and every projection lies in the box widened
by `tol`. These are conditions on each supplied point. `point_count` must be positive and match the
number of triples in `corners3d` (8 for a cuboid), but acceptance doesn't establish that the points
form a cuboid or belong to an object in the image.

In the command, `checkCert` runs on `BoxCameraCert Float`: the projection and every comparison use
host binary64 with round-to-nearest. Because the soundness theorems are generic in the scalar type,
for `Float` they say that the accepted Boolean tests held for the rounded values. Nothing is proved
about the exact real projection, and the artifact's `tol` is its sole slack for rounding.

`ofHomogeneous_inside_bbox` works in an ordered field. Given
nonnegative intervals for the two projection numerators and a depth interval with a positive lower
end, whose quotient intervals lie in the box, every projection drawn from those intervals lies in
the box. `pinholeInterval_inside_bbox` does the same for nonnegative
intervals on the focal lengths and normalized coordinates, plus any interval on the principal point.
Both take their intervals from the caller, and neither says anything about a `Float` run.
