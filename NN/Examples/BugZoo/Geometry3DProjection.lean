/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Geometry3D.Box3D

/-!
# BugZoo: 3D camera projection glue

3D vision systems often fail at the boundary between a learned model and the geometry code that
consumes its tensors.  The model may output plausible 3D corners, a camera matrix, and a 2D box, but
small glue mistakes can silently invalidate the claim:

- OpenCV / PyTorch3D / KITTI camera convention mismatches;
- negative-depth or opposite-facing camera coordinate systems;
- row/column or `xyxy`/`yxyx` box-layout swaps;
- malformed `8 x 3` corner tensors or `3 x 4` camera matrices; and
- projected 3D corners that are not actually enclosed by the claimed 2D detector box.

Real reports motivating this card include PyTorch3D camera-conversion/projection issues
`#522`, `#596`, `#1105`, `#1183`, `#1427`; Detectron2 rotated-box shape issue `#2402`;
Omni3D/Cube R-CNN conversion issue `#60`; and BlenderProc projected-3D-bbox issue `#1150`.

TorchLean's checked boundary is not "the neural detector is correct."  The detector is an
untrusted producer.  The checked contract is:

> Given exported tensors `P : Tensor [3,4]`, `corners : Tensor [8,3]`, and a claimed 2D box, Lean
> recomputes the projection, checks positive depth, image bounds, bbox enclosure, and optional
> interval robustness.

The visual companion script
`scripts/verification/geometry3d/render_box3d_cert_overlay.py` renders the same contract for humans:
green accepted overlays, red rejected overlays, and a contact sheet for the bug zoo.
-/

@[expose] public section

namespace NN.Examples.BugZoo.Geometry3DProjection

open NN.Verification.Geometry3D.Box3D

/--
The core 3D glue contract used by the BugZoo card.

If a certificate passes `checkCert`, then it satisfies the theorem-facing `Verified3DBox` predicate:
positive image dimensions, ordered in-frame bbox, positive depth for all eight tensor corners,
projected corners inside the image, and projected corners enclosed by the claimed 2D box.
-/
theorem accepted_camera_box_certificate_is_verified
    {α : Type} [OfNat α 0] [OfNat α 1] [Add α] [Sub α] [Mul α] [Div α]
    [LE α] [LT α] [DecidableRel (α := α) (· ≤ ·)] [DecidableRel (α := α) (· < ·)]
    {cert : BoxCameraCert α}
    (h : checkCert cert = true) :
    Verified3DBox cert :=
  checkCert_sound h

/--
The interval-robustness contract for perspective division.

This re-exports the stronger theorem under the BugZoo namespace: if intervals for homogeneous
projection numerators and positive depth divide into pixel intervals inside the claimed bbox, then
every concrete projection represented by those intervals is inside the bbox.
-/
theorem homogeneous_projection_uncertainty_stays_inside_bbox
    {α : Type} [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {cert : BoxCameraCert α}
    {uNumI vNumI zI : ScalarInterval α}
    {uNum vNum z : α}
    (huNumNonneg : NonnegativeInterval uNumI)
    (hvNumNonneg : NonnegativeInterval vNumI)
    (hzPos : 0 < zI.lo)
    (hinside :
      PixelIntervalInsideBBox cert
        (homogeneousProjectionInterval uNumI vNumI zI))
    (huNum : InInterval uNumI uNum)
    (hvNum : InInterval vNumI vNum)
    (hz : InInterval zI z) :
    xmin cert ≤ uNum / z ∧
      uNum / z ≤ xmax cert ∧
      ymin cert ≤ vNum / z ∧
      vNum / z ≤ ymax cert :=
  homogeneous_projection_interval_inside_bbox_sound
    huNumNonneg hvNumNonneg hzPos hinside huNum hvNum hz

end NN.Examples.BugZoo.Geometry3DProjection
