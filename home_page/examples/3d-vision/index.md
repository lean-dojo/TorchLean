---
title: 3D Vision Projection Certificates
---

A 3D detector can act as an artifact producer. The detector/exporter emits a camera matrix, 3D
points, image dimensions, and a claimed 2D box. TorchLean reloads that JSON, recomputes the
projection in Lean, and checks whether the claimed box encloses the projected points. A cuboid
uses eight corners; the certificate also accepts other point counts.

The result is a small certificate for this exported scene: the checker establishes that
the claimed 2D box follows from the camera and corner tensors in the artifact. The detector remains
the producer; TorchLean checks the geometric claim it exported.

<div class="media-slab">
  <img src="{{ '/assets/media/examples/showcase/geometry3d-vision-certificates.png' | relative_url }}" alt="3D vision projection certificate workflow"/>
</div>

## Checked Geometry

The checked claim is geometric:

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
every projected point is enclosed by the claimed 2D box. These checks do not establish that the
points form a cuboid or that a detector found the right object. For an empty point set, the
pointwise conditions are vacuous; the image and box checks still apply.

The core artifact type is a tensor-shaped camera certificate:

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

And the theorem connects the executable checker to the formal contract. The excerpt omits the
standard arithmetic/typeclass parameters; the full theorem is in the
[3D geometry verification source](https://github.com/lean-dojo/TorchLean/tree/main/NN/Verification/Geometry3D):

```lean
theorem checkCert_sound
    {cert : BoxCameraCert α} (h : checkCert cert = true) :
    Verified3DBox cert
```

The key pattern is simple: the detector and exporter produce data; the Lean checker recomputes the
geometric claim from that data.

## Run The Real Model Path

The direct 3D detector route uses WildDet3D from Hugging Face. The optional path installs a real 3D
detector pipeline, so it is better as an end-to-end example than as a first runtime check.

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
lake exe verify -- camera-box3d-cert \
  _external/geometry3d/wilddet3d/wilddet3d_cat_box3d_cert.json
```

It also renders PNG overlays under:

```text
_external/geometry3d/wilddet3d/
```

The accepted overlay uses the projected 3D footprint as the claimed box. The strict diagnostic
overlay uses WildDet3D's own 2D detection box. On the default example image, Lean rejects the strict
claim because projected 3D corners fall outside that box.

<div class="media-slab">
  <img src="{{ '/assets/media/examples/bug-zoo/geometry3d-wilddet3d-bbox-diagnostic.png' | relative_url }}" alt="WildDet3D model 2D box compared with projected 3D footprint"/>
</div>

## What A JSON Artifact Looks Like

The concrete JSON is kept plain so the same checker can read artifacts from WildDet3D, Omni3D,
or another detector/exporter that emits the camera and box fields. A `torchlean.camera.box3d.v1`
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

Use the detector's own 2D box to inspect a strict projection claim:

```bash
python3 scripts/verification/geometry3d/export_wilddet3d_box3d_cert.py \
  --text-prompt cat --bbox-source model2d \
  --out _external/geometry3d/wilddet3d/wilddet3d_cat_model2d_strict_box3d_cert.json \
  --overlay
```

The renderer runs the Lean checker and marks accepted and rejected artifacts. A model-box
mismatch remains visible in this diagnostic; it is not repaired by changing the claimed box.

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
