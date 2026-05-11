---
title: 3D Vision Projection Certificates
usemathjax: true
---

This example treats a 3D detector as an artifact producer. The detector/exporter emits a camera
matrix, eight 3D box corners, image dimensions, and a claimed 2D box. TorchLean reloads that JSON,
recomputes the projection in Lean, and checks whether the claimed box really encloses the projected
corners.

The result is a small but useful certificate: not a proof that the detector is always correct, but
a check that this exported geometry claim follows from the tensors in the artifact.

<div class="media-slab">
  <img src="{{ '/assets/media/examples/showcase/geometry3d-vision-certificates.png' | relative_url }}" alt="3D vision projection certificate workflow"/>
</div>

## What This Checks

The checked claim is deliberately small and geometric:

```text
camera_P : 3 x 4 projection matrix
corners3d : 8 x 3 cuboid corners
bbox2d : [xmin, ymin, xmax, ymax]
```

For each corner `(x, y, z)`, the checker forms the homogeneous point `[x, y, z, 1]`, multiplies by
the `3 x 4` camera matrix, divides image coordinates by projected depth, and compares the resulting
pixel `(u, v)` with both the image bounds and the claimed 2D box. The depth check matters: a point
behind the camera should not be accepted just because its divided coordinates look plausible.

Lean verifies that image dimensions are positive, the box is ordered and inside the image, all eight
corners have positive projected depth, every projected corner is inside the image, and every
projected corner is enclosed by the claimed 2D box.

The core artifact type is a tensor-shaped camera certificate:

```lean
structure BoxCameraCert (α : Type) where
  width : α
  height : α
  tol : α
  camera : CameraP α
  corners : BoxCorners α
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

And the theorem connects the executable checker to the theorem-facing contract. The page omits the
standard arithmetic/typeclass parameters; the full theorem is in the
[3D box verification API]({{ '/docs/NN/Verification/Geometry3D/Box3D.html' | relative_url }}):

```lean
theorem checkCert_sound
    {cert : BoxCameraCert α} (h : checkCert cert = true) :
    Verified3DBox cert
```

This is the key pattern: the detector and exporter produce data; the Lean checker recomputes the
geometric claim from that data.

## Run The Real Model Path

The direct 3D detector route uses WildDet3D from Hugging Face. This path installs a real 3D
detector stack, so it is best treated as an optional end-to-end demo rather than a first smoke test.

```bash
python3 -m pip install -r scripts/verification/geometry3d/requirements-wilddet3d.txt
python3 -m pip install --no-deps utils3d
python3 scripts/verification/regenerate_assets.py --group geometry3d-wilddet3d --run
```

The command exports and checks:

```bash
lake exe verify -- camera-box3d-cert \
  _external/geometry3d/wilddet3d/wilddet3d_cat_box3d_cert.json
```

It also renders PNG overlays under:

```text
_external/geometry3d/wilddet3d/
```

The accepted overlay uses the projected 3D footprint as the claimed box. The strict diagnostic
overlay uses WildDet3D's own 2D detection box. On the default demo image, Lean rejects the strict
claim because projected 3D corners fall outside that box.

<div class="media-slab">
  <img src="{{ '/assets/media/examples/bug-zoo/geometry3d-wilddet3d-bbox-diagnostic.png' | relative_url }}" alt="WildDet3D model 2D box compared with projected 3D footprint"/>
</div>

## What A JSON Artifact Looks Like

The concrete JSON is deliberately plain so it is easy to produce from any detector, not just
WildDet3D.

```json
{
  "format": "torchlean.camera.box3d.v1",
  "width": 640.0,
  "height": 480.0,
  "tol": 1.0,
  "camera_P": [1.0, 0.0, 320.0, 0.0, 0.0, 1.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0],
  "corners3d": [0.0, 0.0, 8.0, 1.0, 0.0, 8.0, 1.0, 1.0, 8.0, 0.0, 1.0, 8.0,
                0.0, 0.0, 10.0, 1.0, 0.0, 10.0, 1.0, 1.0, 10.0, 0.0, 1.0, 10.0],
  "bbox2d": [319.0, 239.0, 321.0, 241.0]
}
```

For Cube R-CNN, Omni3D, or another detector, the workflow is the same: export `K` or `P`, image
size, corners, and a claimed box, then run the Lean checker.

```bash
python3 scripts/verification/geometry3d/export_omni3d_box3d_cert.py \
  --prediction-json output/evaluation/predictions.json \
  --out _external/geometry3d/omni3d_box3d_cert.json \
  --verify
```

## Negative Cases

The demo also includes bad certificates motivated by real glue failures: swapped box layouts,
negative depth, wrong projection matrix layout, and 2D boxes that do not enclose projected 3D
corners.

```bash
python3 scripts/verification/regenerate_assets.py --group geometry3d-visual --run
```

Green overlays correspond to certificates accepted by Lean. Red overlays correspond to rejected
certificates. The checker reads the exported camera and box tensors directly; the visualization is
there to help a human see what happened.

The Bug Zoo wrapper re-exports the theorem under a tutorial-facing name:

```lean
theorem accepted_camera_box_certificate_is_verified
    {cert : BoxCameraCert α}
    (h : checkCert cert = true) :
    Verified3DBox cert :=
  checkCert_sound h
```

There is also an interval robustness theorem for perspective division. If homogeneous projection
numerator/depth intervals divide into a pixel interval contained in the bbox, every concrete camera
choice represented by those intervals stays inside the same bbox. Again, this is the readable
shape of the statement; the full theorem includes the interval hypotheses.

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
