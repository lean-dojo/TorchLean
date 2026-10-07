---
title: 3D Vision Projection Certificates
---

In this example, we'll take a detector's 3D points and check whether their projections fit inside
its claimed 2D image box. We'll export the camera matrix, points, image dimensions, and box to
JSON, then check their geometry in Lean.

The checker projects the points and tests whether the box encloses them, allowing a stated
nonnegative tolerance. A cuboid has eight corners, but the certificate also supports other point
counts.

<div class="media-slab">
  <img src="{{ '/assets/media/examples/showcase/geometry3d-vision-certificates.png' | relative_url }}" alt="3D vision projection certificate workflow"/>
</div>

The illustration sketches the workflow. Here we check projection and enclosure for the supplied
points—not the pose, dimensions, or perpendicular edges of a detected cuboid. The sections below
show the exact certificate fields and checks.

## Checked Geometry

Let's start with the data we need for the geometry check:

- `camera_P` is a $3 \times 4$ projection matrix;
- `corners3d` is a $\mathtt{pointCount} \times 3$ matrix of supplied points;
- `bbox2d` stores $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$.

For each corner $(x,y,z)$, the checker forms the homogeneous point $[x,y,z,1]$, multiplies by
the $3 \times 4$ camera matrix, divides image coordinates by projected depth, and compares the
resulting pixel $(u,v)$ with both the image bounds and the claimed 2D box. The depth check matters:
a point behind the camera is rejected before its divided coordinates can be treated as an image
point.

Lean checks that image dimensions are positive, the box is ordered and inside the image, all
supplied points have positive projected depth, every projected point is inside the image, and
every projected point is enclosed by the claimed 2D box expanded by `tol` in each direction.
The tolerance must be nonnegative. It does not relax the positive-depth or image-bound checks. These checks do not establish that the
points form a cuboid or that a detector found the right object. For an empty point set, the
pointwise conditions are vacuous; the image and box checks still apply.

We'll collect these tensors in a camera certificate. The excerpts below use the
`NN.Verification.Geometry3D.Box3D` namespace and omit the surrounding scalar-instance parameters:

```lean
structure BoxCameraCert (α : Type)
    [TorchLean.Storage α] where
  pointCount : Nat := 8
  width : α
  height : α
  tol : α
  camera : CameraP α
  corners : Tensor α [pointCount, 3]
  bbox : Box2D α
```

The executable checker is a Boolean function:

```lean
def checkCert (cert : BoxCameraCert α) : Bool :=
  checkPositiveImageSize cert &&
    checkBBoxOrdered cert &&
    checkBBoxInsideImage cert &&
    checkPositiveDepths cert &&
    checkProjectedInImage cert &&
    checkBBoxEnclosesProjection cert
```

If the checker returns `true`, we can use this theorem to obtain `Verified3DBox cert`.
The excerpt omits the
standard arithmetic/typeclass parameters; the full theorem is in the
[3D geometry verification source](https://github.com/lean-dojo/TorchLean/tree/main/NN/Verification/Geometry3D):

```lean
theorem checkCert_sound
    {cert : BoxCameraCert α} (h : checkCert cert = true) :
    Verified3DBox cert
```

## Run The Real Model Path

To try this with a detector, we'll download and run WildDet3D from Hugging Face. This optional
step requires the detector's dependencies as well as TorchLean:

```bash
python3 -m pip install -r scripts/verification/geometry3d/requirements-wilddet3d.txt
python3 -m pip install --no-deps utils3d
python3 scripts/verification/geometry3d/export_wilddet3d_box3d_cert.py \
  --text-prompt cat \
  --out _external/geometry3d/wilddet3d/wilddet3d_cat_box3d_cert.json \
  --verify --overlay
```

The command exports and checks:

```bash
scripts/lake.sh exe verify -- camera-box3d-cert \
  _external/geometry3d/wilddet3d/wilddet3d_cat_box3d_cert.json
```

It also renders PNG overlays under:

```text
_external/geometry3d/wilddet3d/
```

The accepted overlay uses the projected 3D footprint as the claimed box. The strict diagnostic
overlay uses WildDet3D's own 2D detection box. On the default example image, Lean rejects the strict
claim because projected 3D corners fall outside that box.

WildDet3D downloads its Hugging Face Space source and checkpoint, including the SAM and
LingBot-depth dependencies. We install `utils3d` separately with `--no-deps` because its full
dependency chain includes Open3D/GLTF visualization packages that the exporter does not use.
The exporter rejects detections below `--min-score`.

Choose the claimed box with `--bbox-source`:

- `auto`, the default, uses the model's 2D box when it encloses the projected corners. Otherwise,
  it records the mismatch in metadata and exports the projected footprint.
- `model2d` keeps the model's box, so the checker can reject an inconsistent prediction.
- `projected-envelope` always uses the image-clipped projected footprint with the requested padding.

<div class="media-slab">
  <img src="{{ '/assets/media/examples/bug-zoo/geometry3d-wilddet3d-bbox-diagnostic.png' | relative_url }}" alt="WildDet3D model 2D box compared with projected 3D footprint"/>
</div>

### Combining A Detector With Depth

We can also combine `facebook/detr-resnet-50` detections with
`depth-anything/Depth-Anything-V2-Small-hf` depth estimates. The producer backprojects each selected
image box and depth crop into eight camera-frame corners. Lean checks the exported projection
contract; it does not establish that the label or estimated depth matches the scene.

```bash
python3 -m pip install -r scripts/verification/geometry3d/requirements-realworld.txt
python3 scripts/verification/geometry3d/export_hf_depth_box3d_cert.py \
  --batch-manifest scripts/verification/geometry3d/realworld_manifest.json \
  --batch-out-dir _external/geometry3d/realworld --verify
python3 scripts/verification/geometry3d/render_box3d_cert_overlay.py \
  --glob '_external/geometry3d/realworld/*.json' \
  --out-dir _external/geometry3d/overlays/realworld --contact-sheet
```

The JSON artifacts go under `_external/geometry3d/realworld/`; the overlays and
`geometry3d_contact_sheet.png` go under `_external/geometry3d/overlays/realworld/`.

## What A JSON Artifact Looks Like

Here's the JSON we'll pass to the checker. We can use the same format with WildDet3D, Omni3D,
or another detector that exports the camera and box fields. A `torchlean.camera.box3d.v1`
artifact carries exactly eight corners, with `point_count` omitted or set to `8`. A
`torchlean.camera.box3d.v2` artifact must declare a positive `point_count` that matches the number
of triples in `corners3d`.

```json
{
  "format": "torchlean.camera.box3d.v1",
  "image_width": 640.0,
  "image_height": 480.0,
  "tol": 1.0,
  "point_count": 8,
  "camera_P": [1.0, 0.0, 320.0, 0.0, 0.0, 1.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0],
  "corners3d": [0.0, 0.0, 8.0, 1.0, 0.0, 8.0, 1.0, 1.0, 8.0, 0.0, 1.0, 8.0,
                0.0, 0.0, 10.0, 1.0, 0.0, 10.0, 1.0, 1.0, 10.0, 0.0, 1.0, 10.0],
  "bbox2d": [319.0, 239.0, 321.0, 241.0]
}
```

For Cube R-CNN, Omni3D, or another detector, the export path is the same: export $K$ or $P$, image
size, corners, and a claimed box, then run the Lean checker.

```bash
python3 scripts/verification/geometry3d/export_omni3d_box3d_cert.py \
  --prediction-json output/evaluation/predictions.json \
  --out _external/geometry3d/omni3d_box3d_cert.json \
  --verify
```

## Negative Cases

Let's also check a case that can fail. Here we'll use the detector's own 2D box, rather than
constructing a box from the projected points:

```bash
python3 scripts/verification/geometry3d/export_wilddet3d_box3d_cert.py \
  --text-prompt cat --bbox-source model2d \
  --out _external/geometry3d/wilddet3d/wilddet3d_cat_model2d_strict_box3d_cert.json \
  --overlay
```

To compare the claimed box with the projected corners:

```bash
python3 scripts/verification/geometry3d/render_box3d_cert_overlay.py --view intervals \
  --cert _external/geometry3d/wilddet3d/wilddet3d_cat_box3d_cert.json \
  --out _external/geometry3d/wilddet3d/wilddet3d_bbox_diagnostic.png
```

The default overlay runs the Lean checker and marks accepted and rejected artifacts. The
interval view above only compares the exported metadata; it does not run the checker. A model-box
mismatch remains visible in this diagnostic; it is not repaired by changing the claimed box.

The Bug Zoo wrapper re-exports the theorem under a tutorial-facing name:

```lean
theorem accepted_camera_box_certificate_is_verified
    {cert : BoxCameraCert α}
    (h : checkCert cert = true) :
    Verified3DBox cert :=
  checkCert_sound h
```

We can also reason about uncertainty in the projection using intervals. If homogeneous projection
numerator/depth intervals divide into a pixel interval contained in the bbox, every concrete camera
choice represented by those intervals stays inside the same bbox. The displayed theorem keeps the
readable shape of the statement; the full theorem includes the interval hypotheses.

```lean
theorem homogeneous_projection_uncertainty_stays_inside_bbox :
    xmin cert ≤ uNum / z ∧
      uNum / z ≤ xmax cert ∧
      ymin cert ≤ vNum / z ∧
      vNum / z ≤ ymax cert
```

Next, read the [Bug Zoo walkthrough]({{ '/examples/bug-zoo/' | relative_url }}) for the surrounding
failure-mode catalog, or the
[Verification Bounds walkthrough]({{ '/examples/verification/' | relative_url }}) for IBP and
CROWN-style graph verification.
