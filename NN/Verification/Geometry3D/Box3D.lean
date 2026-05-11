/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public
public import NN.Verification.Util.FloatApprox
public import NN.Verification.Util.Json

/-!
# Tensor-native 3D box camera certificates

This module is TorchLean's tensor-native camera-box certificate implementation.

Cube R-CNN, SAM 3D, and related systems can be treated as untrusted producers of tensors. This
checker verifies the geometric contract of one exported artifact:

* a camera projection matrix `P : Tensor α (shape![3,4])`;
* eight 3D cuboid corners `corners : Tensor α (shape![8,3])`;
* image dimensions; and
* a claimed 2D enclosing box.

The checker recomputes the pinhole projection of every 3D corner and checks:

1. every projected corner has positive camera depth;
2. every projected pixel lies inside the image bounds; and
3. the claimed 2D box itself is inside the image;
4. the claimed 2D box encloses all projected corners, up to an explicit tolerance.

Why this shape?

Real 3D perception systems already speak tensors: Cube R-CNN / Omni3D predictions, PyTorch3D
utilities, and SAM-3D-style post-hoc metadata all move around matrices, points, boxes, and camera
parameters. So this module uses `Spec.Tensor` throughout instead of introducing a detached `Vec3`
island. The few abbrevs below are only names for tensor shapes.
-/

@[expose] public section

namespace NN.Verification.Geometry3D.Box3D

open scoped BigOperators
open NN.Verification.Json

/-! ## Small interval arithmetic layer -/

/--
A closed scalar interval `[lo, hi]`.

This intentionally stays compact: it is just enough to state and prove camera-parameter uncertainty
theorems without pulling the 3D demo into a much larger interval-analysis framework.
-/
structure ScalarInterval (α : Type) where
  /-- Lower endpoint. -/
  lo : α
  /-- Upper endpoint. -/
  hi : α

/-- Membership in a closed scalar interval. -/
def InInterval {α : Type} [LE α] (I : ScalarInterval α) (x : α) : Prop :=
  I.lo ≤ x ∧ x ≤ I.hi

/-- The interval has nonnegative lower endpoint, hence all members are nonnegative. -/
def NonnegativeInterval {α : Type} [OfNat α 0] [LE α] (I : ScalarInterval α) : Prop :=
  0 ≤ I.lo

/-- Interval addition: `[a,b] + [c,d] = [a+c, b+d]`. -/
def addInterval {α : Type} [Add α] (I J : ScalarInterval α) : ScalarInterval α where
  lo := I.lo + J.lo
  hi := I.hi + J.hi

/--
Interval multiplication for nonnegative intervals:
`[a,b] * [c,d] = [a*c, b*d]` when both intervals are nonnegative.
-/
def mulNonnegInterval {α : Type} [Mul α] (I J : ScalarInterval α) : ScalarInterval α where
  lo := I.lo * J.lo
  hi := I.hi * J.hi

/--
Interval division for a nonnegative numerator interval and a strictly positive denominator
interval:

`[a,b] / [c,d] = [a/d, b/c]` when `0 ≤ a` and `0 < c`.

This is the perspective-division primitive.  A pinhole projection is not just affine camera
arithmetic; it also divides homogeneous pixel numerators by positive depth.  Keeping this operation
separate makes the trust boundary explicit: an exporter may provide interval enclosures for
`u_num`, `v_num`, and `z`, and Lean proves the quotient enclosure is sound.
-/
def divNonnegByPosInterval {α : Type} [Div α] (num den : ScalarInterval α) :
    ScalarInterval α where
  lo := num.lo / den.hi
  hi := num.hi / den.lo

/--
Soundness of interval addition.

If `x ∈ I` and `y ∈ J`, then `x + y ∈ I + J`.
-/
theorem addInterval_sound
    {α : Type} [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {I J : ScalarInterval α} {x y : α}
    (hx : InInterval I x) (hy : InInterval J y) :
    InInterval (addInterval I J) (x + y) := by
  rcases hx with ⟨hxlo, hxhi⟩
  rcases hy with ⟨hylo, hyhi⟩
  dsimp [InInterval, addInterval]
  constructor <;> linarith

/--
Soundness of nonnegative interval multiplication.

If `I` and `J` have nonnegative lower endpoints, then multiplication is monotone over both
intervals: `x ∈ I`, `y ∈ J` implies `x*y ∈ [I.lo*J.lo, I.hi*J.hi]`.
-/
theorem mulNonnegInterval_sound
    {α : Type} [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {I J : ScalarInterval α} {x y : α}
    (hI : NonnegativeInterval I) (hJ : NonnegativeInterval J)
    (hx : InInterval I x) (hy : InInterval J y) :
    InInterval (mulNonnegInterval I J) (x * y) := by
  rcases hx with ⟨hxlo, hxhi⟩
  rcases hy with ⟨hylo, hyhi⟩
  have hx_nonneg : 0 ≤ x := le_trans hI hxlo
  have hy_nonneg : 0 ≤ y := le_trans hJ hylo
  have hIhi_nonneg : 0 ≤ I.hi := le_trans hx_nonneg hxhi
  dsimp [InInterval, mulNonnegInterval]
  constructor
  ·
    have h1 : I.lo * J.lo ≤ x * J.lo :=
      mul_le_mul_of_nonneg_right hxlo hJ
    have h2 : x * J.lo ≤ x * y :=
      mul_le_mul_of_nonneg_left hylo hx_nonneg
    exact le_trans h1 h2
  ·
    have h1 : x * y ≤ I.hi * y :=
      mul_le_mul_of_nonneg_right hxhi hy_nonneg
    have h2 : I.hi * y ≤ I.hi * J.hi :=
      mul_le_mul_of_nonneg_left hyhi hIhi_nonneg
    exact le_trans h1 h2

/--
Soundness of perspective-style interval division.

If `x ∈ num`, `z ∈ den`, `num` is nonnegative, and the denominator interval is bounded away from
zero, then `x / z` lies in `[num.lo / den.hi, num.hi / den.lo]`.

This theorem is the mathematical core behind the "full uncertainty envelope" for 3D boxes: depth
uncertainty is handled by a certified quotient, not by an informal post-processing check.
-/
theorem divNonnegByPosInterval_sound
    {α : Type} [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {num den : ScalarInterval α} {x z : α}
    (hnum : NonnegativeInterval num)
    (hdenPos : 0 < den.lo)
    (hx : InInterval num x)
    (hz : InInterval den z) :
    InInterval (divNonnegByPosInterval num den) (x / z) := by
  rcases hx with ⟨hxlo, hxhi⟩
  rcases hz with ⟨hzlo, hzhi⟩
  have hx_nonneg : 0 ≤ x := le_trans hnum hxlo
  have hnum_hi_nonneg : 0 ≤ num.hi := le_trans hx_nonneg hxhi
  have hz_pos : 0 < z := lt_of_lt_of_le hdenPos hzlo
  dsimp [InInterval, divNonnegByPosInterval]
  constructor
  ·
    exact div_le_div₀ hx_nonneg hxlo hz_pos hzhi
  ·
    exact div_le_div₀ hnum_hi_nonneg hxhi hdenPos hzlo

/--
Pinhole x-coordinate interval from uncertain intrinsics and normalized coordinate.

For `u = fx * xn + cx`, with `fx ≥ 0` and `xn ≥ 0`, interval arithmetic gives:
`u ∈ fxI*xnI + cxI`.
-/
def pinholePixelInterval {α : Type} [Add α] [Mul α]
    (fI coordI cI : ScalarInterval α) : ScalarInterval α :=
  addInterval (mulNonnegInterval fI coordI) cI

/--
Soundness of the one-axis pinhole interval formula.

This is the camera-parameter uncertainty theorem used by the 3D geometry certificate workflow: if focal length,
principal point, and normalized coordinate are each inside certified intervals, then the resulting
pixel coordinate is inside the interval computed by `pinholePixelInterval`.
-/
theorem pinholePixelInterval_sound
    {α : Type} [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {fI coordI cI : ScalarInterval α} {f coord c : α}
    (hfNonneg : NonnegativeInterval fI)
    (hcoordNonneg : NonnegativeInterval coordI)
    (hf : InInterval fI f)
    (hcoord : InInterval coordI coord)
    (hc : InInterval cI c) :
    InInterval (pinholePixelInterval fI coordI cI) (f * coord + c) :=
  addInterval_sound
    (mulNonnegInterval_sound hfNonneg hcoordNonneg hf hcoord)
    hc

/-- A pair of x/y pixel intervals enclosing a projected point under camera uncertainty. -/
structure PixelInterval2D (α : Type) where
  /-- Pixel-x interval. -/
  x : ScalarInterval α
  /-- Pixel-y interval. -/
  y : ScalarInterval α

/-- A concrete pixel lies inside a 2D pixel interval. -/
def PixelInInterval2D {α : Type} [LE α] (pix : PixelInterval2D α) (px py : α) : Prop :=
  InInterval pix.x px ∧ InInterval pix.y py

/--
Pixel interval obtained from homogeneous camera-coordinate intervals.

Here `uNumI` and `vNumI` enclose the first two rows of `P * [X,Y,Z,1]`, while `zI` encloses the
positive depth row.  The actual projected pixel is `(u_num / z, v_num / z)`.

This formulation matches real exported tensors better than a detached `Vec3` API: camera matrices,
3D corners, and interval bounds can all be produced by the same tensor pipeline, and Lean proves the
final quotient/bbox claim.
-/
def homogeneousProjectionInterval {α : Type} [Div α]
    (uNumI vNumI zI : ScalarInterval α) : PixelInterval2D α where
  x := divNonnegByPosInterval uNumI zI
  y := divNonnegByPosInterval vNumI zI

/-- If `List.all p xs` succeeds, then `p` succeeds on every member of `xs`. -/
theorem list_all_true_of_mem {α : Type} {p : α → Bool} {xs : List α}
    (h : xs.all p = true) {x : α} (hx : x ∈ xs) : p x = true := by
  induction xs with
  | nil =>
      simp at hx
  | cons a xs ih =>
      simp [List.all] at h hx
      rcases h with ⟨ha, ht⟩
      rcases hx with rfl | hx
      · exact ha
      · exact ht x hx

/-- A 3D point as a tensor of shape `[3]`. -/
abbrev Point3 (α : Type) := Spec.Tensor α (NN.Tensor.Shape.Vec 3)

/-- A 2D point as a tensor of shape `[2]`. -/
abbrev Point2 (α : Type) := Spec.Tensor α (NN.Tensor.Shape.Vec 2)

/-- Eight cuboid corners, stored as a tensor of shape `[8, 3]`. -/
abbrev BoxCorners (α : Type) := Spec.Tensor α (NN.Tensor.Shape.Mat 8 3)

/--
A 3×4 camera projection matrix.

For a homogeneous point `[X,Y,Z,1]`, the raw camera coordinates are:

`u_num = P₀·[X,Y,Z,1]`, `v_num = P₁·[X,Y,Z,1]`, `z = P₂·[X,Y,Z,1]`.

The projected pixel is `(u_num / z, v_num / z)`, checked only when `z > 0`.
-/
abbrev CameraP (α : Type) := Spec.Tensor α (NN.Tensor.Shape.Mat 3 4)

/-- Projected eight-corner tensor of shape `[8, 2]`. -/
abbrev ProjectedCorners (α : Type) := Spec.Tensor α (NN.Tensor.Shape.Mat 8 2)

/-- A 2D box `[xmin, ymin, xmax, ymax]`. -/
abbrev Box2D (α : Type) := Spec.Tensor α (NN.Tensor.Shape.Vec 4)

/-- Matrix scalar accessor for tensor-shaped matrices. -/
def matGet {α : Type} {rows cols : Nat}
    (x : Spec.Tensor α (NN.Tensor.Shape.Mat rows cols)) (i : Fin rows) (j : Fin cols) : α :=
  Spec.Tensor.toScalar (Spec.get (Spec.get x i) j)

/-- Vector scalar accessor for tensor-shaped vectors. -/
def vecGet {α : Type} {n : Nat}
    (x : Spec.Tensor α (NN.Tensor.Shape.Vec n)) (i : Fin n) : α :=
  Spec.Tensor.toScalar (Spec.get x i)

/-- Extract the `i`-th cuboid corner as a `[3]` tensor. -/
def corner {α : Type} (corners : BoxCorners α) (i : Fin 8) : Point3 α :=
  Spec.Tensor.dim (fun j => Spec.Tensor.scalar (matGet corners i j))

/-- Build a matrix tensor from a coordinate function. Useful for small fixtures. -/
def matOfFn {α : Type} (rows cols : Nat) (f : Fin rows → Fin cols → α) :
    Spec.Tensor α (NN.Tensor.Shape.Mat rows cols) :=
  Spec.Tensor.dim (fun i => Spec.Tensor.dim (fun j => Spec.Tensor.scalar (f i j)))

/-- Build a vector tensor from a coordinate function. Useful for small fixtures. -/
def vecOfFn {α : Type} (n : Nat) (f : Fin n → α) :
    Spec.Tensor α (NN.Tensor.Shape.Vec n) :=
  Spec.Tensor.dim (fun i => Spec.Tensor.scalar (f i))

/-- Raw homogeneous camera coordinate `P[row] · [X,Y,Z,1]`. -/
def cameraCoord {α : Type} [OfNat α 1] [Add α] [Mul α]
    (P : CameraP α) (x : Point3 α) (row : Fin 3) : α :=
  matGet P row ⟨0, by decide⟩ * vecGet x ⟨0, by decide⟩ +
  matGet P row ⟨1, by decide⟩ * vecGet x ⟨1, by decide⟩ +
  matGet P row ⟨2, by decide⟩ * vecGet x ⟨2, by decide⟩ +
  matGet P row ⟨3, by decide⟩ * (1 : α)

/-- Positive-depth denominator used by pinhole projection. -/
def projectZ {α : Type} [OfNat α 1] [Add α] [Mul α]
    (P : CameraP α) (x : Point3 α) : α :=
  cameraCoord P x ⟨2, by decide⟩

/-- Projected x/pixel coordinate. Meaningful when `projectZ P x ≠ 0`. -/
def projectX {α : Type} [OfNat α 1] [Add α] [Mul α] [Div α]
    (P : CameraP α) (x : Point3 α) : α :=
  cameraCoord P x ⟨0, by decide⟩ / projectZ P x

/-- Projected y/pixel coordinate. Meaningful when `projectZ P x ≠ 0`. -/
def projectY {α : Type} [OfNat α 1] [Add α] [Mul α] [Div α]
    (P : CameraP α) (x : Point3 α) : α :=
  cameraCoord P x ⟨1, by decide⟩ / projectZ P x

/-- Project one 3D point to a 2D tensor. -/
def projectPoint {α : Type} [OfNat α 1] [Add α] [Mul α] [Div α]
    (P : CameraP α) (x : Point3 α) : Point2 α :=
  vecOfFn 2 (fun j =>
    if j.val = 0 then
      projectX P x
    else
      projectY P x)

/-- Project all eight cuboid corners. -/
def projectBox {α : Type} [OfNat α 1] [Add α] [Mul α] [Div α]
    (P : CameraP α) (corners : BoxCorners α) : ProjectedCorners α :=
  matOfFn 8 2 (fun i j => vecGet (projectPoint P (corner corners i)) j)

/-- A compact exported 3D-box/camera certificate. -/
structure BoxCameraCert (α : Type) where
  /-- Image width in pixels. -/
  width : α
  /-- Image height in pixels. -/
  height : α
  /-- Tolerance used when checking the claimed enclosing box. -/
  tol : α
  /-- 3×4 camera projection matrix. -/
  camera : CameraP α
  /-- Eight exported 3D cuboid corners. -/
  corners : BoxCorners α
  /-- Claimed 2D box `[xmin,ymin,xmax,ymax]`. -/
  bbox : Box2D α

/-- Claimed 2D box left edge. -/
def xmin {α : Type} (cert : BoxCameraCert α) : α := vecGet cert.bbox ⟨0, by decide⟩
/-- Claimed 2D box top edge. -/
def ymin {α : Type} (cert : BoxCameraCert α) : α := vecGet cert.bbox ⟨1, by decide⟩
/-- Claimed 2D box right edge. -/
def xmax {α : Type} (cert : BoxCameraCert α) : α := vecGet cert.bbox ⟨2, by decide⟩
/-- Claimed 2D box bottom edge. -/
def ymax {α : Type} (cert : BoxCameraCert α) : α := vecGet cert.bbox ⟨3, by decide⟩

/-- The pixel interval is contained in the claimed 2D box. -/
def PixelIntervalInsideBBox {α : Type} [LE α]
    (cert : BoxCameraCert α) (pix : PixelInterval2D α) : Prop :=
  xmin cert ≤ pix.x.lo ∧ pix.x.hi ≤ xmax cert ∧
    ymin cert ≤ pix.y.lo ∧ pix.y.hi ≤ ymax cert

/--
If a pixel interval is contained in the claimed bbox, every concrete pixel inside that interval is
also contained in the bbox.
-/
theorem pixel_inside_bbox_of_interval_inside
    {α : Type} [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {cert : BoxCameraCert α} {pix : PixelInterval2D α} {px py : α}
    (hinside : PixelIntervalInsideBBox cert pix)
    (hpix : PixelInInterval2D pix px py) :
    xmin cert ≤ px ∧ px ≤ xmax cert ∧ ymin cert ≤ py ∧ py ≤ ymax cert := by
  rcases hinside with ⟨hxlo, hxhi, hylo, hyhi⟩
  rcases hpix with ⟨⟨hpxlo, hpxhi⟩, ⟨hpylo, hpyhi⟩⟩
  constructor
  · exact le_trans hxlo hpxlo
  constructor
  · exact le_trans hpxhi hxhi
  constructor
  · exact le_trans hylo hpylo
  · exact le_trans hpyhi hyhi

/--
Full pinhole-intrinsics interval-to-bbox theorem.

Assume independent intervals for focal lengths (`fx`, `fy`), normalized coordinates
(`xn = X/Z`, `yn = Y/Z`), and principal point (`cx`, `cy`).  If the interval arithmetic result

`u ∈ fxI*xnI + cxI`, `v ∈ fyI*ynI + cyI`

is contained in the claimed bbox, then every concrete camera/pixel choice satisfying those input
intervals projects inside the bbox.

This is the full uncertainty-envelope bridge for the camera-uncertainty workflow: uncertainty begins at camera
intrinsics and normalized camera coordinates, propagates through the pinhole equations, and ends as
a certified bbox-enclosure statement.  The theorem is phrased over normalized coordinates so callers
can plug in whichever depth-interval or model-uncertainty method they use to bound `X/Z` and `Y/Z`.
-/
theorem pinhole_intrinsics_interval_inside_bbox_sound
    {α : Type} [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {cert : BoxCameraCert α}
    {fxI fyI xnI ynI cxI cyI : ScalarInterval α}
    {fx fy xn yn cx cy : α}
    (hfxNonneg : NonnegativeInterval fxI)
    (hfyNonneg : NonnegativeInterval fyI)
    (hxnNonneg : NonnegativeInterval xnI)
    (hynNonneg : NonnegativeInterval ynI)
    (hinside :
      PixelIntervalInsideBBox cert
        { x := pinholePixelInterval fxI xnI cxI
          y := pinholePixelInterval fyI ynI cyI })
    (hfx : InInterval fxI fx)
    (hfy : InInterval fyI fy)
    (hxn : InInterval xnI xn)
    (hyn : InInterval ynI yn)
    (hcx : InInterval cxI cx)
    (hcy : InInterval cyI cy) :
    xmin cert ≤ fx * xn + cx ∧
      fx * xn + cx ≤ xmax cert ∧
      ymin cert ≤ fy * yn + cy ∧
      fy * yn + cy ≤ ymax cert := by
  have hx :
      InInterval (pinholePixelInterval fxI xnI cxI) (fx * xn + cx) :=
    pinholePixelInterval_sound hfxNonneg hxnNonneg hfx hxn hcx
  have hy :
      InInterval (pinholePixelInterval fyI ynI cyI) (fy * yn + cy) :=
    pinholePixelInterval_sound hfyNonneg hynNonneg hfy hyn hcy
  exact pixel_inside_bbox_of_interval_inside
    (cert := cert)
    (pix := { x := pinholePixelInterval fxI xnI cxI
              y := pinholePixelInterval fyI ynI cyI })
    hinside
    ⟨hx, hy⟩

/--
Full homogeneous projection interval-to-bbox theorem.

This is the stronger robustness theorem. Instead of assuming the projected pixel has
already been bounded, it starts from intervals over the homogeneous camera outputs:

* `uNumI` encloses the x numerator `P₀ · [X,Y,Z,1]`;
* `vNumI` encloses the y numerator `P₁ · [X,Y,Z,1]`; and
* `zI` encloses the positive depth denominator `P₂ · [X,Y,Z,1]`.

If the quotient intervals `[uNumI.lo / zI.hi, uNumI.hi / zI.lo]` and
`[vNumI.lo / zI.hi, vNumI.hi / zI.lo]` are inside the claimed 2D box, then every concrete camera
projection represented by those intervals is inside the box.  This is exactly the guard we want for
uncertain camera intrinsics/extrinsics, bounded corner perturbations, or mixed numeric backends:
they may produce ranges for homogeneous coordinates, but Lean checks the perspective divide and
the final bbox containment.
-/
theorem homogeneous_projection_interval_inside_bbox_sound
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
      vNum / z ≤ ymax cert := by
  have hx :
      InInterval (divNonnegByPosInterval uNumI zI) (uNum / z) :=
    divNonnegByPosInterval_sound huNumNonneg hzPos huNum hz
  have hy :
      InInterval (divNonnegByPosInterval vNumI zI) (vNum / z) :=
    divNonnegByPosInterval_sound hvNumNonneg hzPos hvNum hz
  exact pixel_inside_bbox_of_interval_inside
    (cert := cert)
    (pix := homogeneousProjectionInterval uNumI vNumI zI)
    hinside
    ⟨hx, hy⟩

/-- Projected x-coordinate of corner `i`. -/
def certProjectX {α : Type} [OfNat α 1] [Add α] [Mul α] [Div α]
    (cert : BoxCameraCert α) (i : Fin 8) : α :=
  projectX cert.camera (corner cert.corners i)

/-- Projected y-coordinate of corner `i`. -/
def certProjectY {α : Type} [OfNat α 1] [Add α] [Mul α] [Div α]
    (cert : BoxCameraCert α) (i : Fin 8) : α :=
  projectY cert.camera (corner cert.corners i)

/-- Camera depth of corner `i`. -/
def certProjectZ {α : Type} [OfNat α 1] [Add α] [Mul α]
    (cert : BoxCameraCert α) (i : Fin 8) : α :=
  projectZ cert.camera (corner cert.corners i)

/-! ## Mathematical certificate predicates -/

/-- Image dimensions must be positive. -/
def PositiveImageSize {α : Type} [OfNat α 0] [LT α] (cert : BoxCameraCert α) : Prop :=
  0 < cert.width ∧ 0 < cert.height

/-- Basic sanity for the reported 2D box and tolerance. -/
def BBoxOrdered {α : Type} [OfNat α 0] [LE α] (cert : BoxCameraCert α) : Prop :=
  0 ≤ cert.tol ∧ xmin cert ≤ xmax cert ∧ ymin cert ≤ ymax cert

/-- The reported 2D box itself should lie inside the image frame. -/
def BBoxInsideImage {α : Type} [OfNat α 0] [LE α] (cert : BoxCameraCert α) : Prop :=
  0 ≤ xmin cert ∧ xmax cert ≤ cert.width ∧
  0 ≤ ymin cert ∧ ymax cert ≤ cert.height

/-- Every 3D corner lies in front of the camera. -/
def PositiveDepths {α : Type} [OfNat α 0] [OfNat α 1] [Add α] [Mul α] [LT α]
    (cert : BoxCameraCert α) : Prop :=
  ∀ i : Fin 8, 0 < certProjectZ cert i

/-- Every projected corner lies inside the image rectangle. -/
def ProjectedInImage {α : Type} [OfNat α 0] [OfNat α 1] [Add α] [Mul α] [Div α] [LE α]
    (cert : BoxCameraCert α) : Prop :=
  ∀ i : Fin 8,
    0 ≤ certProjectX cert i ∧ certProjectX cert i ≤ cert.width ∧
    0 ≤ certProjectY cert i ∧ certProjectY cert i ≤ cert.height

/-- The reported 2D box encloses all projected 3D corners, up to tolerance. -/
def BBoxEnclosesProjection {α : Type} [OfNat α 1] [Add α] [Sub α] [Mul α] [Div α] [LE α]
    (cert : BoxCameraCert α) : Prop :=
  ∀ i : Fin 8,
    xmin cert - cert.tol ≤ certProjectX cert i ∧
      certProjectX cert i ≤ xmax cert + cert.tol ∧
    ymin cert - cert.tol ≤ certProjectY cert i ∧
      certProjectY cert i ≤ ymax cert + cert.tol

/-! ## Robust projection predicates

The single-artifact checker above says the exported projection is coherent at one numeric point.
For real model outputs we also want robustness statements around that point.

There are two complementary robustness layers in this file:

* `homogeneous_projection_interval_inside_bbox_sound` starts from intervals over homogeneous camera
  outputs `(u_num, v_num, z)` and proves the perspective-divided pixels stay inside the claimed box.
* `bbox_encloses_perturbed_of_margin` starts from a nominal projected pixel and proves that any
  bounded pixel-space perturbation stays inside the claimed box if there is enough slack.

Together they cover the two most common exporter contracts: interval-enclosed camera arithmetic and
bounded downstream pixel perturbation.
-/

/--
The projected corners are inside the claimed 2D box with an explicit positive slack `margin`.

Compared with `BBoxEnclosesProjection`, this predicate is stricter: corners must land inside
`[xmin + margin, xmax - margin] × [ymin + margin, ymax - margin]`.  The margin is what pays for
future uncertainty.
-/
def BBoxEnclosesProjectionWithMargin {α : Type} [OfNat α 1] [Add α] [Sub α] [Mul α] [Div α]
    [LE α] (cert : BoxCameraCert α) (margin : α) : Prop :=
  ∀ i : Fin 8,
    xmin cert + margin ≤ certProjectX cert i ∧
      certProjectX cert i ≤ xmax cert - margin ∧
    ymin cert + margin ≤ certProjectY cert i ∧
      certProjectY cert i ≤ ymax cert - margin

/--
`px, py` describe an interval-perturbed projection of every corner.

For each corner `i`, `px i` and `py i` are some possible projected coordinates after exporter,
renderer, rounding, or uncertainty perturbation.  This predicate says those coordinates are within
`eps` pixels of the nominal projection recomputed from the certificate.
-/
def ProjectionPerturbationWithin {α : Type} [OfNat α 1] [Add α] [Sub α] [Mul α] [Div α]
    [LE α] (cert : BoxCameraCert α) (eps : α) (px py : Fin 8 → α) : Prop :=
  ∀ i : Fin 8,
    certProjectX cert i - eps ≤ px i ∧
      px i ≤ certProjectX cert i + eps ∧
    certProjectY cert i - eps ≤ py i ∧
      py i ≤ certProjectY cert i + eps

/--
The claimed 2D box encloses every perturbed projected corner.

This is the robust postcondition downstream consumers want: even after a bounded perturbation, the
candidate 3D artifact still projects inside the detector's claimed 2D box.
-/
def BBoxEnclosesPerturbedProjection {α : Type} [LE α]
    (cert : BoxCameraCert α) (px py : Fin 8 → α) : Prop :=
  ∀ i : Fin 8,
    xmin cert ≤ px i ∧ px i ≤ xmax cert ∧
      ymin cert ≤ py i ∧ py i ≤ ymax cert

/--
Projection-interval robustness theorem.

If every nominal projected corner has at least `margin` pixels of slack inside the box, and every
perturbed projection is within `eps ≤ margin` pixels of that nominal projection, then all perturbed
projections remain inside the original claimed 2D box.

This is deliberately independent of how the perturbation was produced: it can come from Float
rounding, a renderer approximation, a model uncertainty interval, or an external exporter.  The
theorem is the certified geometry guard.
-/
theorem bbox_encloses_perturbed_of_margin
    {α : Type} [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {cert : BoxCameraCert α} {eps margin : α} {px py : Fin 8 → α}
    (hle : eps ≤ margin)
    (hmargin : BBoxEnclosesProjectionWithMargin cert margin)
    (hperturb : ProjectionPerturbationWithin cert eps px py) :
    BBoxEnclosesPerturbedProjection cert px py := by
  intro i
  rcases hmargin i with ⟨hxlo, hxhi, hylo, hyhi⟩
  rcases hperturb i with ⟨hpxlo, hpxhi, hpylo, hpyhi⟩
  constructor
  · linarith
  constructor
  · linarith
  constructor
  · linarith
  · linarith

/--
The end-user theorem-facing statement.

This says the exported detector artifact is internally coherent under its stated camera model:
positive image dimensions, ordered in-frame 2D box, positive camera depth, projected corners inside
the image, and claimed 2D box enclosure.
-/
def Verified3DBox {α : Type} [OfNat α 0] [OfNat α 1] [Add α] [Sub α] [Mul α] [Div α]
    [LE α] [LT α] (cert : BoxCameraCert α) : Prop :=
  PositiveImageSize cert ∧
    BBoxOrdered cert ∧
    BBoxInsideImage cert ∧
    PositiveDepths cert ∧
    ProjectedInImage cert ∧
    BBoxEnclosesProjection cert

/-! ## Boolean checker and soundness theorems -/

/-- Boolean check for positive image dimensions. -/
def checkPositiveImageSize {α : Type} [OfNat α 0] [LT α]
    [DecidableRel (α := α) (· < ·)] (cert : BoxCameraCert α) : Bool :=
  decide (0 < cert.width) && decide (0 < cert.height)

/-- Boolean check for `BBoxOrdered`. -/
def checkBBoxOrdered {α : Type} [OfNat α 0] [LE α] [DecidableRel (α := α) (· ≤ ·)]
    (cert : BoxCameraCert α) : Bool :=
  decide (0 ≤ cert.tol) && decide (xmin cert ≤ xmax cert) && decide (ymin cert ≤ ymax cert)

/-- Boolean check for `BBoxInsideImage`. -/
def checkBBoxInsideImage {α : Type} [OfNat α 0] [LE α]
    [DecidableRel (α := α) (· ≤ ·)] (cert : BoxCameraCert α) : Bool :=
  decide (0 ≤ xmin cert) &&
    decide (xmax cert ≤ cert.width) &&
    decide (0 ≤ ymin cert) &&
    decide (ymax cert ≤ cert.height)

/-- Boolean check for `PositiveDepths`. -/
def checkPositiveDepths {α : Type} [OfNat α 0] [OfNat α 1] [Add α] [Mul α] [LT α]
    [DecidableRel (α := α) (· < ·)] (cert : BoxCameraCert α) : Bool :=
  (List.finRange 8).all (fun i => decide (0 < certProjectZ cert i))

/-- Boolean check for `ProjectedInImage`. -/
def checkProjectedInImage {α : Type} [OfNat α 0] [OfNat α 1] [Add α] [Mul α] [Div α] [LE α]
    [DecidableRel (α := α) (· ≤ ·)] (cert : BoxCameraCert α) : Bool :=
  (List.finRange 8).all (fun i =>
    decide
      (0 ≤ certProjectX cert i ∧ certProjectX cert i ≤ cert.width ∧
       0 ≤ certProjectY cert i ∧ certProjectY cert i ≤ cert.height))

/-- Boolean check for `BBoxEnclosesProjection`. -/
def checkBBoxEnclosesProjection {α : Type} [OfNat α 1] [Add α] [Sub α] [Mul α] [Div α] [LE α]
    [DecidableRel (α := α) (· ≤ ·)] (cert : BoxCameraCert α) : Bool :=
  (List.finRange 8).all (fun i =>
    decide
      (xmin cert - cert.tol ≤ certProjectX cert i ∧
       certProjectX cert i ≤ xmax cert + cert.tol ∧
       ymin cert - cert.tol ≤ certProjectY cert i ∧
       certProjectY cert i ≤ ymax cert + cert.tol))

/-- Full Boolean checker for one exported 3D-box/camera artifact. -/
def checkCert {α : Type} [OfNat α 0] [OfNat α 1] [Add α] [Sub α] [Mul α] [Div α]
    [LE α] [LT α] [DecidableRel (α := α) (· ≤ ·)] [DecidableRel (α := α) (· < ·)]
    (cert : BoxCameraCert α) : Bool :=
  checkPositiveImageSize cert &&
    checkBBoxOrdered cert &&
    checkBBoxInsideImage cert &&
    checkPositiveDepths cert &&
    checkProjectedInImage cert &&
    checkBBoxEnclosesProjection cert

theorem checkPositiveImageSize_sound {α : Type} [OfNat α 0] [LT α]
    [DecidableRel (α := α) (· < ·)] {cert : BoxCameraCert α}
    (h : checkPositiveImageSize cert = true) :
    PositiveImageSize cert := by
  simp [checkPositiveImageSize] at h
  exact h

theorem checkBBoxOrdered_sound {α : Type} [OfNat α 0] [LE α]
    [DecidableRel (α := α) (· ≤ ·)] {cert : BoxCameraCert α}
    (h : checkBBoxOrdered cert = true) :
    BBoxOrdered cert := by
  simp [checkBBoxOrdered] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

theorem checkBBoxInsideImage_sound {α : Type} [OfNat α 0] [LE α]
    [DecidableRel (α := α) (· ≤ ·)] {cert : BoxCameraCert α}
    (h : checkBBoxInsideImage cert = true) :
    BBoxInsideImage cert := by
  simp [checkBBoxInsideImage] at h
  exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩

theorem checkPositiveDepths_sound {α : Type} [OfNat α 0] [OfNat α 1] [Add α] [Mul α] [LT α]
    [DecidableRel (α := α) (· < ·)] {cert : BoxCameraCert α}
    (h : checkPositiveDepths cert = true) :
    PositiveDepths cert := by
  intro i
  have hi : decide (0 < certProjectZ cert i) = true :=
    list_all_true_of_mem h (by simp)
  exact of_decide_eq_true hi

theorem checkProjectedInImage_sound {α : Type} [OfNat α 0] [OfNat α 1]
    [Add α] [Mul α] [Div α] [LE α] [DecidableRel (α := α) (· ≤ ·)]
    {cert : BoxCameraCert α} (h : checkProjectedInImage cert = true) :
    ProjectedInImage cert := by
  intro i
  have hi :
      decide
         (0 ≤ certProjectX cert i ∧ certProjectX cert i ≤ cert.width ∧
         0 ≤ certProjectY cert i ∧ certProjectY cert i ≤ cert.height) = true :=
    list_all_true_of_mem h (by simp)
  exact of_decide_eq_true hi

theorem checkBBoxEnclosesProjection_sound {α : Type} [OfNat α 1]
    [Add α] [Sub α] [Mul α] [Div α] [LE α] [DecidableRel (α := α) (· ≤ ·)]
    {cert : BoxCameraCert α} (h : checkBBoxEnclosesProjection cert = true) :
    BBoxEnclosesProjection cert := by
  intro i
  have hi :
      decide
        (xmin cert - cert.tol ≤ certProjectX cert i ∧
         certProjectX cert i ≤ xmax cert + cert.tol ∧
         ymin cert - cert.tol ≤ certProjectY cert i ∧
         certProjectY cert i ≤ ymax cert + cert.tol) = true :=
    list_all_true_of_mem h (by simp)
  exact of_decide_eq_true hi

/--
Soundness of the executable checker.

This is the main theorem for the implementation: if the Boolean checker accepts an artifact, the
artifact satisfies the mathematical `Verified3DBox` predicate.
-/
theorem checkCert_sound {α : Type} [OfNat α 0] [OfNat α 1] [Add α] [Sub α] [Mul α] [Div α]
    [LE α] [LT α] [DecidableRel (α := α) (· ≤ ·)] [DecidableRel (α := α) (· < ·)]
    {cert : BoxCameraCert α} (h : checkCert cert = true) :
    Verified3DBox cert := by
  simp [checkCert] at h
  rcases h with ⟨⟨⟨⟨⟨hSize, hBox⟩, hBoxIn⟩, hDepth⟩, hImage⟩, hEnclose⟩
  exact ⟨
    checkPositiveImageSize_sound hSize,
    checkBBoxOrdered_sound hBox,
    checkBBoxInsideImage_sound hBoxIn,
    checkPositiveDepths_sound hDepth,
    checkProjectedInImage_sound hImage,
    checkBBoxEnclosesProjection_sound hEnclose
  ⟩

/-! ## Consequence theorems for verified artifacts -/

theorem Verified3DBox.positive_image_size {α : Type} [OfNat α 0] [OfNat α 1]
    [Add α] [Sub α] [Mul α] [Div α] [LE α] [LT α]
    {cert : BoxCameraCert α} (h : Verified3DBox cert) :
    0 < cert.width ∧ 0 < cert.height :=
  h.1

theorem Verified3DBox.bbox_inside_image {α : Type} [OfNat α 0] [OfNat α 1]
    [Add α] [Sub α] [Mul α] [Div α] [LE α] [LT α]
    {cert : BoxCameraCert α} (h : Verified3DBox cert) :
    0 ≤ xmin cert ∧ xmax cert ≤ cert.width ∧
      0 ≤ ymin cert ∧ ymax cert ≤ cert.height :=
  h.2.2.1

theorem Verified3DBox.corner_positive_depth {α : Type} [OfNat α 0] [OfNat α 1]
    [Add α] [Sub α] [Mul α] [Div α] [LE α] [LT α]
    {cert : BoxCameraCert α} (h : Verified3DBox cert) (i : Fin 8) :
    0 < certProjectZ cert i :=
  h.2.2.2.1 i

theorem Verified3DBox.projected_corner_in_image {α : Type} [OfNat α 0] [OfNat α 1]
    [Add α] [Sub α] [Mul α] [Div α] [LE α] [LT α]
    {cert : BoxCameraCert α} (h : Verified3DBox cert) (i : Fin 8) :
    0 ≤ certProjectX cert i ∧ certProjectX cert i ≤ cert.width ∧
      0 ≤ certProjectY cert i ∧ certProjectY cert i ≤ cert.height :=
  h.2.2.2.2.1 i

theorem Verified3DBox.projected_corner_in_claimed_bbox {α : Type} [OfNat α 0] [OfNat α 1]
    [Add α] [Sub α] [Mul α] [Div α] [LE α] [LT α]
    {cert : BoxCameraCert α} (h : Verified3DBox cert) (i : Fin 8) :
    xmin cert - cert.tol ≤ certProjectX cert i ∧
      certProjectX cert i ≤ xmax cert + cert.tol ∧
      ymin cert - cert.tol ≤ certProjectY cert i ∧
      certProjectY cert i ≤ ymax cert + cert.tol :=
  h.2.2.2.2.2 i

/-! ## Float JSON checker for exported artifacts -/

/-- Expected schema string for JSON artifacts accepted by this checker. -/
def formatString : String := "torchlean.camera.box3d.v1"

/-- Tensor from a flat array. The caller is responsible for checking the length. -/
def matFloatOfFlat (rows cols : Nat) (xs : Array Float) :
    Spec.Tensor Float (NN.Tensor.Shape.Mat rows cols) :=
  matOfFn rows cols (fun i j => xs.getD (i.val * cols + j.val) 0.0)

/-- Vector tensor from a flat array. The caller is responsible for checking the length. -/
def vecFloatOfFlat (n : Nat) (xs : Array Float) :
    Spec.Tensor Float (NN.Tensor.Shape.Vec n) :=
  vecOfFn n (fun i => xs.getD i.val 0.0)

def requireSize (ctx : String) (xs : Array Float) (expected : Nat) : IO Unit := do
  if xs.size != expected then
    throw <| IO.userError s!"{ctx}: expected {expected} floats, got {xs.size}"

/-- Parse a JSON artifact into the Float checker representation. -/
def parseJsonCert (j : Lean.Json) : IO (BoxCameraCert Float) := do
  expectFormat j formatString
  let width ← expectFloat (← expectField j "image_width" "top-level") "top-level.image_width"
  let height ← expectFloat (← expectField j "image_height" "top-level") "top-level.image_height"
  let tol ← expectFloat (← expectField j "tol" "top-level") "top-level.tol"
  let cameraFlat ← expectFieldFloatArray j "camera_P" "top-level"
  let cornersFlat ← expectFieldFloatArray j "corners3d" "top-level"
  let bboxFlat ← expectFieldFloatArray j "bbox2d" "top-level"
  requireSize "top-level.camera_P" cameraFlat 12
  requireSize "top-level.corners3d" cornersFlat 24
  requireSize "top-level.bbox2d" bboxFlat 4
  pure {
    width := width
    height := height
    tol := tol
    camera := matFloatOfFlat 3 4 cameraFlat
    corners := matFloatOfFlat 8 3 cornersFlat
    bbox := vecFloatOfFlat 4 bboxFlat
  }

/--
Load and check a JSON 3D-box/camera certificate.

This is the bridge to Cube R-CNN/SAM-3D-style outputs. The JSON is untrusted; Lean recomputes the
projection and accepts only if the tensor artifact passes the checker.
-/
def checkFile (path : String) : IO Bool := do
  let j ← readJsonObjectFile path
  let cert ← parseJsonCert j
  let ok := checkCert cert
  if ok then
    IO.println s!"3D camera-box certificate verified: {path}"
  else
    IO.println s!"3D camera-box certificate rejected: {path}"
  pure ok

/-- Convenience wrapper for CLI fixtures. -/
def checkOrThrow (path : String) : IO Unit := do
  let ok ← checkFile path
  if !ok then
    throw <| IO.userError s!"camera-box3d certificate check failed: {path}"

end NN.Verification.Geometry3D.Box3D
