# Geometry3D Real Demo

This is the Geometry3D demo for users. The Lean checker and theorem statements are under:

```text
NN/Verification/Geometry3D
```

This directory contains untrusted producers and visualizers. They run external model code, export
JSON, ask Lean to check the exported tensor artifact, and render overlays for humans.

## Real Model Path

Run the real image smoke suite:

```bash
python3 scripts/verification/regenerate_assets.py --group geometry3d-real --run
```

This loads:

- `facebook/detr-resnet-50` for real 2D detections;
- `depth-anything/Depth-Anything-V2-Small-hf` for real monocular depth;
- a conservative backprojection producer that converts the detected 2D box and depth crop into
  eight 3D frustum/cuboid corners.

It writes real exported certs to:

```text
_external/geometry3d/realworld/*.json
```

and visual overlays to:

```text
_external/geometry3d/overlays/realworld/*.png
_external/geometry3d/overlays/realworld/geometry3d_contact_sheet.png
```

Important honesty point: this path is a real model pipeline, but it is not a specialized 3D
detector. It uses a detector plus monocular depth to produce a conservative 3D camera-box artifact.
Lean verifies the exported geometry contract, not the semantic truth of the object label or metric
depth.

## Direct 3D Detector Path: WildDet3D

Run the heavier direct 3D detector path:

```bash
python3 -m pip install -r scripts/verification/geometry3d/requirements-wilddet3d.txt
python3 -m pip install --no-deps utils3d
python3 scripts/verification/regenerate_assets.py --group geometry3d-wilddet3d --run
```

This downloads the [`allenai/WildDet3D`](https://huggingface.co/allenai/WildDet3D) Hugging Face
Space source plus the [`allenai/WildDet3D`](https://huggingface.co/allenai/WildDet3D) checkpoint,
runs a text prompt monocular 3D detection pass, converts the selected predicted 3D box to eight
camera frame corners, exports:

```text
_external/geometry3d/wilddet3d/wilddet3d_cat_box3d_cert.json
```

and checks it with:

```bash
lake exe verify -- camera-box3d-cert _external/geometry3d/wilddet3d/wilddet3d_cat_box3d_cert.json
```

It also renders:

```text
_external/geometry3d/wilddet3d/wilddet3d_cat_box3d_cert.png
_external/geometry3d/wilddet3d/wilddet3d_cat_model2d_strict_box3d_cert.png
_external/geometry3d/wilddet3d/wilddet3d_bbox_diagnostic.png
_external/geometry3d/wilddet3d/geometry3d_contact_sheet.png
```

This is the direct 3D detector route. It is heavier because WildDet3D is about a 1.2B
parameter model and depends on the Space's SAM/LingBot-depth support code and checkpoint.  The
separate `utils3d --no-deps` install is deliberate: WildDet3D imports `utils3d` through its depth
backend, but the full dependency chain pulls Open3D/GLTF visualization packages that are not needed
for the certificate exporter and can fail on newer Python versions.

The exporter has two bbox modes:

- `--bbox-source auto` is the default used above. It checks whether the model's own 2D box encloses
  the projected 3D corners. If not, it records the mismatch in metadata and exports the exact
  projected footprint bbox, which Lean can verify.
- `--bbox-source model2d` exports the strict model 2D box claim. On the default cat image this is
  rejected by Lean, which is a useful real diagnostic: the 3D prediction is projectable, but its
  projected footprint is slightly outside the detector's 2D box.

## Direct 3D Detector Path: Cube R-CNN / Omni3D

For Cube R-CNN / Omni3D style predictions, use:

```bash
python3 scripts/verification/geometry3d/export_omni3d_box3d_cert.py \
  --prediction-json output/evaluation/predictions.json \
  --out _external/geometry3d/omni3d_box3d_cert.json \
  --verify
```

That path expects an external detector to have already produced `K`, image size, a 2D bbox, and
`bbox3D` corners. The model and exporter are untrusted; Lean checks the final tensor artifact.

## Bad Case Visual Demo

Run:

```bash
python3 scripts/verification/regenerate_assets.py --group geometry3d-visual --run
```

This generates bad certificates motivated by real projection and camera layout issues, requires Lean to
reject them, and renders a contact sheet:

```text
_external/geometry3d/overlays/bugzoo/geometry3d_contact_sheet.png
```

Green overlays are accepted by Lean. Red overlays are rejected by Lean. The images draw the claimed
2D box, projected 3D corners, cuboid edges, and checker status so the geometry failure is visible.

## Theorem Checks

The main theorem objects live in:

```text
NN/Verification/Geometry3D/Box3D.lean
```

Key statements:

- `checkCert_sound`: if the Boolean checker accepts a certificate, the mathematical
  `Verified3DBox` predicate holds.
- `homogeneous_projection_interval_inside_bbox_sound`: if intervals for `u_num`, `v_num`, and
  positive depth `z` perspective-divide into pixel intervals inside the bbox, every represented
  concrete projection stays inside the bbox.
- `bbox_encloses_perturbed_of_margin`: if nominal projected corners have enough slack, bounded
  pixel perturbations remain inside the same bbox.

Build/check:

```bash
lake build NN.Verification.Geometry3D NN.Verification.CLI NN.Examples.BugZoo.All
lake exe verify -- camera-box3d-cert _external/geometry3d/realworld/coco_cats_depth_box.json
```
