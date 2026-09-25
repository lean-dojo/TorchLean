/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Util.Json
public import NN.Verification.Util.Tensor

/-!
# Tensor-native 3D box camera certificates

This module is TorchLean's tensor-native camera-box certificate implementation.

Cube R-CNN, SAM 3D, and related systems can be treated as untrusted producers of tensors. This
checker verifies the geometric contract of one exported artifact:

* a camera projection matrix `P : Tensor α [3,4]`;
* a configurable number of 3D points `corners : Tensor α [pointCount,3]`;
* image dimensions; and
* a claimed 2D enclosing box.

The checker recomputes the pinhole projection of every 3D corner and checks:

1. every projected corner has positive camera depth;
2. every projected pixel lies inside the image bounds; and
3. the claimed 2D box itself is inside the image;
4. the claimed 2D box encloses all projected corners, up to an explicit tolerance.

Eight points are the default for a cuboid. The contract covers every supplied point; it does not
assert that those points form a cuboid. With no points, the pointwise conditions are vacuous,
while the image-size and bounding-box conditions still apply.

Why this shape?

Real 3D perception systems already speak tensors: Cube R-CNN / Omni3D predictions, PyTorch3D
utilities, and SAM-3D-style post-hoc metadata all move around matrices, points, boxes, and camera
parameters. So this module uses `TorchLean.Tensor` throughout instead of introducing a detached
`Vec3` island. The few abbrevs below are only names for tensor shapes.
-/

@[expose] public section

namespace NN.Verification.Geometry3D.Box3D

open NN.Verification.Json

/-! ## Small interval arithmetic layer -/

/--
A closed scalar interval `[lo, hi]`.

This module states and proves the camera-parameter uncertainty
theorems without pulling the 3D example into a much larger interval-analysis framework.
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

/-- Interval addition: $[a,b]+[c,d]=[a+c,b+d]$. -/
def addInterval {α : Type} [Add α] (I J : ScalarInterval α) : ScalarInterval α where
  lo := I.lo + J.lo
  hi := I.hi + J.hi

/--
Interval multiplication for nonnegative intervals:
$[a,b][c,d]=[ac,bd]$ when both intervals are nonnegative.
-/
def mulNonnegInterval {α : Type} [Mul α] (I J : ScalarInterval α) : ScalarInterval α where
  lo := I.lo * J.lo
  hi := I.hi * J.hi

/--
Interval division for a nonnegative numerator interval and a strictly positive denominator
interval:

$[a,b]/[c,d]=[a/d,b/c]$ when $0\le a$ and $0<c$.

This is the perspective-division primitive. A pinhole projection combines affine camera arithmetic
with division of homogeneous pixel numerators by positive depth. Keeping this operation separate
makes the trust boundary explicit: an exporter may provide interval enclosures for
`u_num`, `v_num`, and `z`, and Lean proves the quotient enclosure is sound.
-/
def divNonnegByPosInterval {α : Type} [Div α] (num den : ScalarInterval α) :
    ScalarInterval α where
  lo := num.lo / den.hi
  hi := num.hi / den.lo

/--
Soundness of interval addition.

If $x\in I$ and $y\in J$, then $x+y\in I+J$.
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
intervals: $x\in I$ and $y\in J$ imply
$xy\in[I_{\mathrm{lo}}J_{\mathrm{lo}},I_{\mathrm{hi}}J_{\mathrm{hi}}]$.
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

If $x\in\mathrm{num}$, $z\in\mathrm{den}$, $\mathrm{num}$ is nonnegative, and the denominator
interval is bounded away from zero, then
$x/z\in[\mathrm{num}_{\mathrm{lo}}/\mathrm{den}_{\mathrm{hi}},
\mathrm{num}_{\mathrm{hi}}/\mathrm{den}_{\mathrm{lo}}]$.

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

For $u=f_xx_n+c_x$, with $f_x\ge 0$ and $x_n\ge 0$, interval arithmetic gives
$u\in f_{x,I}x_{n,I}+c_{x,I}$.
-/
def pinholePixelIntervalNonnegative {α : Type} [Add α] [Mul α]
    (fI coordI cI : ScalarInterval α) : ScalarInterval α :=
  addInterval (mulNonnegInterval fI coordI) cI

/--
Soundness of the one-axis pinhole interval formula.

This is the camera-parameter uncertainty theorem used by the 3D geometry certificate workflow: if
focal length, principal point, and normalized coordinate are each inside certified intervals, then
the resulting pixel coordinate is inside the interval computed by
`pinholePixelIntervalNonnegative`.
-/
theorem pinholePixelIntervalNonnegative_sound
    {α : Type} [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {fI coordI cI : ScalarInterval α} {f coord c : α}
    (hfNonneg : NonnegativeInterval fI)
    (hcoordNonneg : NonnegativeInterval coordI)
    (hf : InInterval fI f)
    (hcoord : InInterval coordI coord)
    (hc : InInterval cI c) :
    InInterval (pinholePixelIntervalNonnegative fI coordI cI) (f * coord + c) :=
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

Here `uNumI` and `vNumI` enclose the first two rows of $P[X,Y,Z,1]^\mathsf{T}$, while `zI` encloses
the positive depth row.  The actual projected pixel is
$(u_{\mathrm{num}}/z,v_{\mathrm{num}}/z)$.

This formulation matches real exported tensors better than a detached `Vec3` API: camera matrices,
3D corners, and interval bounds can all be produced by the same tensor pipeline, and Lean proves the
final quotient/bbox claim.
-/
def homogeneousProjectionIntervalNonnegative {α : Type} [Div α]
    (uNumI vNumI zI : ScalarInterval α) : PixelInterval2D α where
  x := divNonnegByPosInterval uNumI zI
  y := divNonnegByPosInterval vNumI zI

/-- A 3D point as a tensor of shape `[3]`. -/
abbrev Point3 (α : Type) [TorchLean.Storage α] := TorchLean.Tensor α [3]

/--
A 3×4 camera projection matrix.

For a homogeneous point $[X,Y,Z,1]$, the raw camera coordinates are:

$$
u_{\mathrm{num}}=P_0\mathbin{\cdot}[X,Y,Z,1],\qquad
v_{\mathrm{num}}=P_1\mathbin{\cdot}[X,Y,Z,1],\qquad
z=P_2\mathbin{\cdot}[X,Y,Z,1].
$$

The projected pixel is $(u_{\mathrm{num}}/z,v_{\mathrm{num}}/z)$, checked only when $z>0$.
-/
abbrev CameraP (α : Type) [TorchLean.Storage α] := TorchLean.Tensor α [3, 4]

/-- A 2D box `[xmin, ymin, xmax, ymax]`. -/
abbrev Box2D (α : Type) [TorchLean.Storage α] := TorchLean.Tensor α [4]

/-- Matrix scalar accessor for tensor-shaped matrices. -/
def matGet {α : Type} [TorchLean.Storage α] {rows cols : Nat}
    (x : TorchLean.Tensor α [rows, cols]) (i : Fin rows) (j : Fin cols) : α :=
  TorchLean.Tensor.item (Spec.get (Spec.get x i) j)

/-- Extract the `i`-th supplied 3D point as a `[3]` tensor. -/
def corner {α : Type} [TorchLean.Storage α] {pointCount : Nat}
    (corners : TorchLean.Tensor α [pointCount, 3]) (i : Fin pointCount) : Point3 α :=
  TorchLean.Tensor.dim (fun j => TorchLean.Tensor.scalar (matGet corners i j))

/-- Raw homogeneous camera coordinate `P[row] · [X,Y,Z,1]`. -/
def cameraCoord {α : Type} [TorchLean.Storage α] [OfNat α 1] [Add α] [Mul α]
    (P : CameraP α) (x : Point3 α) (row : Fin 3) : α :=
  matGet P row ⟨0, by decide⟩ * TorchLean.Tensor.getScalar x ⟨0, by decide⟩ +
  matGet P row ⟨1, by decide⟩ * TorchLean.Tensor.getScalar x ⟨1, by decide⟩ +
  matGet P row ⟨2, by decide⟩ * TorchLean.Tensor.getScalar x ⟨2, by decide⟩ +
  matGet P row ⟨3, by decide⟩ * (1 : α)

/-- Positive-depth denominator used by pinhole projection. -/
def projectZ {α : Type} [TorchLean.Storage α] [OfNat α 1] [Add α] [Mul α]
    (P : CameraP α) (x : Point3 α) : α :=
  cameraCoord P x ⟨2, by decide⟩

/-- Projected x/pixel coordinate. Meaningful when `projectZ P x ≠ 0`. -/
def projectX {α : Type} [TorchLean.Storage α] [OfNat α 1] [Add α] [Mul α] [Div α]
    (P : CameraP α) (x : Point3 α) : α :=
  cameraCoord P x ⟨0, by decide⟩ / projectZ P x

/-- Projected y/pixel coordinate. Meaningful when `projectZ P x ≠ 0`. -/
def projectY {α : Type} [TorchLean.Storage α] [OfNat α 1] [Add α] [Mul α] [Div α]
    (P : CameraP α) (x : Point3 α) : α :=
  cameraCoord P x ⟨1, by decide⟩ / projectZ P x

/-- A compact exported 3D-box/camera certificate. -/
structure BoxCameraCert (α : Type) [TorchLean.Storage α] where
  /-- Number of exported 3D points; a cuboid uses eight. -/
  pointCount : Nat := 8
  /-- Image width in pixels. -/
  width : α
  /-- Image height in pixels. -/
  height : α
  /-- Tolerance used when checking the claimed enclosing box. -/
  tol : α
  /-- 3×4 camera projection matrix. -/
  camera : CameraP α
  /-- Exported 3D points, indexed by the declared count. -/
  corners : TorchLean.Tensor α [pointCount, 3]
  /-- Claimed 2D box `[xmin,ymin,xmax,ymax]`. -/
  bbox : Box2D α

/-- Left edge `xmin` of the claimed 2D bounding box. -/
def xmin {α : Type} [TorchLean.Storage α] (cert : BoxCameraCert α) : α :=
  TorchLean.Tensor.getScalar cert.bbox ⟨0, by decide⟩
/-- Top edge `ymin` of the claimed 2D bounding box. -/
def ymin {α : Type} [TorchLean.Storage α] (cert : BoxCameraCert α) : α :=
  TorchLean.Tensor.getScalar cert.bbox ⟨1, by decide⟩
/-- Right edge `xmax` of the claimed 2D bounding box. -/
def xmax {α : Type} [TorchLean.Storage α] (cert : BoxCameraCert α) : α :=
  TorchLean.Tensor.getScalar cert.bbox ⟨2, by decide⟩
/-- Bottom edge `ymax` of the claimed 2D bounding box. -/
def ymax {α : Type} [TorchLean.Storage α] (cert : BoxCameraCert α) : α :=
  TorchLean.Tensor.getScalar cert.bbox ⟨3, by decide⟩

/-- The pixel interval is contained in the claimed 2D box. -/
def PixelIntervalInsideBBox {α : Type} [TorchLean.Storage α] [LE α]
    (cert : BoxCameraCert α) (pix : PixelInterval2D α) : Prop :=
  xmin cert ≤ pix.x.lo ∧ pix.x.hi ≤ xmax cert ∧
    ymin cert ≤ pix.y.lo ∧ pix.y.hi ≤ ymax cert

/--
If a pixel interval is contained in the claimed bbox, every concrete pixel inside that interval is
also contained in the bbox.
-/
theorem pixel_inside_bbox_of_interval_inside
    {α : Type} [TorchLean.Storage α] [Field α] [LinearOrder α] [IsStrictOrderedRing α]
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
Pinhole-intrinsics interval-to-bbox theorem for nonnegative normalized coordinates.

Assume independent intervals for focal lengths (`fx`, `fy`), normalized coordinates
$x_n=X/Z$ and $y_n=Y/Z$, and principal point (`cx`, `cy`). If the interval arithmetic result

$u\in f_{x,I}x_{n,I}+c_{x,I}$ and $v\in f_{y,I}y_{n,I}+c_{y,I}$

is contained in the claimed bbox, then every concrete camera/pixel choice satisfying those input
intervals projects inside the bbox.

This is a quadrant-restricted uncertainty bridge: uncertainty begins at camera
intrinsics and normalized camera coordinates, propagates through the pinhole equations, and ends as
a certified bbox-enclosure statement.  The theorem is phrased over normalized coordinates so callers
can plug in whichever depth-interval or model-uncertainty method they use to bound `X/Z` and `Y/Z`.
-/
theorem pinhole_intrinsics_nonnegative_interval_inside_bbox_sound
    {α : Type} [TorchLean.Storage α] [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {cert : BoxCameraCert α}
    {fxI fyI xnI ynI cxI cyI : ScalarInterval α}
    {fx fy xn yn cx cy : α}
    (hfxNonneg : NonnegativeInterval fxI)
    (hfyNonneg : NonnegativeInterval fyI)
    (hxnNonneg : NonnegativeInterval xnI)
    (hynNonneg : NonnegativeInterval ynI)
    (hinside :
      PixelIntervalInsideBBox cert
        { x := pinholePixelIntervalNonnegative fxI xnI cxI
          y := pinholePixelIntervalNonnegative fyI ynI cyI })
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
      InInterval (pinholePixelIntervalNonnegative fxI xnI cxI) (fx * xn + cx) :=
    pinholePixelIntervalNonnegative_sound hfxNonneg hxnNonneg hfx hxn hcx
  have hy :
      InInterval (pinholePixelIntervalNonnegative fyI ynI cyI) (fy * yn + cy) :=
    pinholePixelIntervalNonnegative_sound hfyNonneg hynNonneg hfy hyn hcy
  exact pixel_inside_bbox_of_interval_inside
    (cert := cert)
    (pix := { x := pinholePixelIntervalNonnegative fxI xnI cxI
              y := pinholePixelIntervalNonnegative fyI ynI cyI })
    hinside
    ⟨hx, hy⟩

/--
Homogeneous projection interval-to-bbox theorem for nonnegative numerators.

Instead of assuming the projected pixel has
already been bounded, it starts from intervals over the homogeneous camera outputs:

* `uNumI` encloses the x numerator $P_0\mathbin{\cdot}[X,Y,Z,1]$;
* `vNumI` encloses the y numerator $P_1\mathbin{\cdot}[X,Y,Z,1]$; and
* `zI` encloses the positive depth denominator $P_2\mathbin{\cdot}[X,Y,Z,1]$.

If the quotient intervals
$[u_{\mathrm{lo}}/z_{\mathrm{hi}},u_{\mathrm{hi}}/z_{\mathrm{lo}}]$ and
$[v_{\mathrm{lo}}/z_{\mathrm{hi}},v_{\mathrm{hi}}/z_{\mathrm{lo}}]$
are inside the claimed 2D box, then every concrete camera
projection represented by those intervals is inside the box.  This is exactly the guard we want for
uncertain camera intrinsics/extrinsics, bounded corner perturbations, or mixed numeric backends:
they may produce ranges for homogeneous coordinates, but Lean checks the perspective divide and
the final bbox containment.
-/
theorem homogeneous_projection_nonnegative_interval_inside_bbox_sound
    {α : Type} [TorchLean.Storage α] [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {cert : BoxCameraCert α}
    {uNumI vNumI zI : ScalarInterval α}
    {uNum vNum z : α}
    (huNumNonneg : NonnegativeInterval uNumI)
    (hvNumNonneg : NonnegativeInterval vNumI)
    (hzPos : 0 < zI.lo)
    (hinside :
      PixelIntervalInsideBBox cert
        (homogeneousProjectionIntervalNonnegative uNumI vNumI zI))
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
    (pix := homogeneousProjectionIntervalNonnegative uNumI vNumI zI)
    hinside
    ⟨hx, hy⟩

/-- Projected x-coordinate of corner `i`. -/
def certProjectX {α : Type} [TorchLean.Storage α] [OfNat α 1] [Add α] [Mul α] [Div α]
    (cert : BoxCameraCert α) (i : Fin cert.pointCount) : α :=
  projectX cert.camera (corner cert.corners i)

/-- Projected y-coordinate of corner `i`. -/
def certProjectY {α : Type} [TorchLean.Storage α] [OfNat α 1] [Add α] [Mul α] [Div α]
    (cert : BoxCameraCert α) (i : Fin cert.pointCount) : α :=
  projectY cert.camera (corner cert.corners i)

/-- Camera depth of corner `i`. -/
def certProjectZ {α : Type} [TorchLean.Storage α] [OfNat α 1] [Add α] [Mul α]
    (cert : BoxCameraCert α) (i : Fin cert.pointCount) : α :=
  projectZ cert.camera (corner cert.corners i)

/-! ## Mathematical certificate predicates -/

/-- Image dimensions must be positive. -/
def PositiveImageSize {α : Type} [TorchLean.Storage α] [OfNat α 0] [LT α] (cert : BoxCameraCert α) :
    Prop :=
  0 < cert.width ∧ 0 < cert.height

/-- Validate the reported 2D box and tolerance. -/
def BBoxOrdered {α : Type} [TorchLean.Storage α] [OfNat α 0] [LE α] (cert : BoxCameraCert α) :
    Prop :=
  0 ≤ cert.tol ∧ xmin cert ≤ xmax cert ∧ ymin cert ≤ ymax cert

/-- The reported 2D box itself should lie inside the image frame. -/
def BBoxInsideImage {α : Type} [TorchLean.Storage α] [OfNat α 0] [LE α] (cert : BoxCameraCert α) :
    Prop :=
  0 ≤ xmin cert ∧ xmax cert ≤ cert.width ∧
  0 ≤ ymin cert ∧ ymax cert ≤ cert.height

/-- Every 3D corner lies in front of the camera. -/
def PositiveDepths {α : Type} [TorchLean.Storage α] [OfNat α 0] [OfNat α 1] [Add α] [Mul α] [LT α]
    (cert : BoxCameraCert α) : Prop :=
  ∀ i : Fin cert.pointCount, 0 < certProjectZ cert i

/-- Every projected corner lies inside the image rectangle. -/
def ProjectedInImage {α : Type} [TorchLean.Storage α] [OfNat α 0] [OfNat α 1] [Add α] [Mul α]
    [Div α] [LE α] (cert : BoxCameraCert α) : Prop :=
  ∀ i : Fin cert.pointCount,
    0 ≤ certProjectX cert i ∧ certProjectX cert i ≤ cert.width ∧
    0 ≤ certProjectY cert i ∧ certProjectY cert i ≤ cert.height

/-- The reported 2D box encloses all projected 3D corners, up to tolerance. -/
def BBoxEnclosesProjection {α : Type} [TorchLean.Storage α] [OfNat α 1] [Add α] [Sub α] [Mul α]
    [Div α] [LE α] (cert : BoxCameraCert α) : Prop :=
  ∀ i : Fin cert.pointCount,
    xmin cert - cert.tol ≤ certProjectX cert i ∧
      certProjectX cert i ≤ xmax cert + cert.tol ∧
    ymin cert - cert.tol ≤ certProjectY cert i ∧
      certProjectY cert i ≤ ymax cert + cert.tol

/-! ## Robust projection predicates

The single-artifact checker above says the exported projection is coherent at one numeric point.
For real model outputs we also want robustness statements around that point.

There are two complementary robustness layers in this file:

* `homogeneous_projection_nonnegative_interval_inside_bbox_sound` starts from nonnegative
  intervals over homogeneous camera outputs `(u_num, v_num, z)` and proves the perspective-divided
  pixels stay inside the claimed box.
* `bbox_encloses_perturbed_of_margin` starts from a nominal projected pixel and proves that any
  bounded pixel-space perturbation stays inside the claimed box if there is enough slack.

Together they cover the two most common exporter contracts: interval-enclosed camera arithmetic and
bounded downstream pixel perturbation.
-/

/--
The projected corners are inside the claimed 2D box with an explicit positive slack `margin`.

Compared with `BBoxEnclosesProjection`, this predicate is stricter: corners must land inside
$[x_{\min}+\mathrm{margin},x_{\max}-\mathrm{margin}]
\times[y_{\min}+\mathrm{margin},y_{\max}-\mathrm{margin}]$.  The margin is what pays for
future uncertainty.
-/
def BBoxEnclosesProjectionWithMargin {α : Type} [TorchLean.Storage α] [OfNat α 1] [Add α] [Sub α]
    [Mul α] [Div α] [LE α] (cert : BoxCameraCert α) (margin : α) : Prop :=
  ∀ i : Fin cert.pointCount,
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
def ProjectionPerturbationWithin {α : Type} [TorchLean.Storage α] [OfNat α 1] [Add α] [Sub α]
    [Mul α] [Div α] [LE α] (cert : BoxCameraCert α) (eps : α)
    (px py : Fin cert.pointCount → α) : Prop :=
  ∀ i : Fin cert.pointCount,
    certProjectX cert i - eps ≤ px i ∧
      px i ≤ certProjectX cert i + eps ∧
    certProjectY cert i - eps ≤ py i ∧
      py i ≤ certProjectY cert i + eps

/--
The claimed 2D box encloses every perturbed projected corner.

This is the robust postcondition downstream consumers want: even after a bounded perturbation, the
candidate 3D artifact still projects inside the detector's claimed 2D box.
-/
def BBoxEnclosesPerturbedProjection {α : Type} [TorchLean.Storage α] [LE α]
    (cert : BoxCameraCert α) (px py : Fin cert.pointCount → α) : Prop :=
  ∀ i : Fin cert.pointCount,
    xmin cert ≤ px i ∧ px i ≤ xmax cert ∧
      ymin cert ≤ py i ∧ py i ≤ ymax cert

/--
Projection-interval robustness theorem.

If every nominal projected corner has at least `margin` pixels of slack inside the box, and every
perturbed projection is within $\varepsilon\le\mathrm{margin}$ pixels of that nominal projection,
then all perturbed
projections remain inside the original claimed 2D box.

This is deliberately independent of how the perturbation was produced: it can come from Float
rounding, a renderer approximation, a model uncertainty interval, or an external exporter.  The
theorem is the certified geometry guard.
-/
theorem bbox_encloses_perturbed_of_margin
    {α : Type} [TorchLean.Storage α] [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {cert : BoxCameraCert α} {eps margin : α} {px py : Fin cert.pointCount → α}
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
The checked 3D box statement exposed by the certificate workflow.

This says the exported detector artifact is internally coherent under its stated camera model:
positive image dimensions, ordered in-frame 2D box, positive camera depth, projected corners inside
the image, and claimed 2D box enclosure.
-/
def Verified3DBox {α : Type} [TorchLean.Storage α] [OfNat α 0] [OfNat α 1] [Add α] [Sub α] [Mul α]
    [Div α] [LE α] [LT α] (cert : BoxCameraCert α) : Prop :=
  PositiveImageSize cert ∧
    BBoxOrdered cert ∧
    BBoxInsideImage cert ∧
    PositiveDepths cert ∧
    ProjectedInImage cert ∧
    BBoxEnclosesProjection cert

/-! ## Boolean checker and soundness theorems -/

/-- Boolean check for positive image dimensions. -/
def checkPositiveImageSize {α : Type} [TorchLean.Storage α] [OfNat α 0] [LT α]
    [DecidableRel (α := α) (· < ·)] (cert : BoxCameraCert α) : Bool :=
  decide (0 < cert.width) && decide (0 < cert.height)

/-- Boolean check for `BBoxOrdered`. -/
def checkBBoxOrdered {α : Type} [TorchLean.Storage α] [OfNat α 0] [LE α]
    [DecidableRel (α := α) (· ≤ ·)] (cert : BoxCameraCert α) : Bool :=
  decide (0 ≤ cert.tol) && decide (xmin cert ≤ xmax cert) && decide (ymin cert ≤ ymax cert)

/-- Boolean check for `BBoxInsideImage`. -/
def checkBBoxInsideImage {α : Type} [TorchLean.Storage α] [OfNat α 0] [LE α]
    [DecidableRel (α := α) (· ≤ ·)] (cert : BoxCameraCert α) : Bool :=
  decide (0 ≤ xmin cert) &&
    decide (xmax cert ≤ cert.width) &&
    decide (0 ≤ ymin cert) &&
    decide (ymax cert ≤ cert.height)

/-- Boolean check for `PositiveDepths`. -/
def checkPositiveDepths {α : Type} [TorchLean.Storage α] [OfNat α 0] [OfNat α 1] [Add α] [Mul α]
    [LT α] [DecidableRel (α := α) (· < ·)] (cert : BoxCameraCert α) : Bool :=
  (List.finRange cert.pointCount).all (fun i => decide (0 < certProjectZ cert i))

/-- Boolean check for `ProjectedInImage`. -/
def checkProjectedInImage {α : Type} [TorchLean.Storage α] [OfNat α 0] [OfNat α 1] [Add α] [Mul α]
    [Div α] [LE α] [DecidableRel (α := α) (· ≤ ·)] (cert : BoxCameraCert α) : Bool :=
  (List.finRange cert.pointCount).all (fun i =>
    decide
      (0 ≤ certProjectX cert i ∧ certProjectX cert i ≤ cert.width ∧
       0 ≤ certProjectY cert i ∧ certProjectY cert i ≤ cert.height))

/-- Boolean check for `BBoxEnclosesProjection`. -/
def checkBBoxEnclosesProjection {α : Type} [TorchLean.Storage α] [OfNat α 1] [Add α] [Sub α] [Mul α]
    [Div α] [LE α] [DecidableRel (α := α) (· ≤ ·)] (cert : BoxCameraCert α) : Bool :=
  (List.finRange cert.pointCount).all (fun i =>
    decide
      (xmin cert - cert.tol ≤ certProjectX cert i ∧
       certProjectX cert i ≤ xmax cert + cert.tol ∧
       ymin cert - cert.tol ≤ certProjectY cert i ∧
       certProjectY cert i ≤ ymax cert + cert.tol))

/-- Full Boolean checker for one exported 3D-box/camera artifact. -/
def checkCert {α : Type} [TorchLean.Storage α] [OfNat α 0] [OfNat α 1] [Add α] [Sub α] [Mul α]
    [Div α] [LE α] [LT α] [DecidableRel (α := α) (· ≤ ·)] [DecidableRel (α := α) (· < ·)]
    (cert : BoxCameraCert α) : Bool :=
  checkPositiveImageSize cert &&
    checkBBoxOrdered cert &&
    checkBBoxInsideImage cert &&
    checkPositiveDepths cert &&
    checkProjectedInImage cert &&
    checkBBoxEnclosesProjection cert

/-- Acceptance by the image-size check yields the corresponding positive-size proposition. -/
theorem checkPositiveImageSize_sound {α : Type} [TorchLean.Storage α] [OfNat α 0] [LT α]
    [DecidableRel (α := α) (· < ·)] {cert : BoxCameraCert α}
    (h : checkPositiveImageSize cert = true) :
    PositiveImageSize cert := by
  simp [checkPositiveImageSize] at h
  exact h

/-- Acceptance by the box-order check yields ordered horizontal and vertical endpoints. -/
theorem checkBBoxOrdered_sound {α : Type} [TorchLean.Storage α] [OfNat α 0] [LE α]
    [DecidableRel (α := α) (· ≤ ·)] {cert : BoxCameraCert α}
    (h : checkBBoxOrdered cert = true) :
    BBoxOrdered cert := by
  simp [checkBBoxOrdered] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

/-- Acceptance by the containment check places every claimed box edge inside the image. -/
theorem checkBBoxInsideImage_sound {α : Type} [TorchLean.Storage α] [OfNat α 0] [LE α]
    [DecidableRel (α := α) (· ≤ ·)] {cert : BoxCameraCert α}
    (h : checkBBoxInsideImage cert = true) :
    BBoxInsideImage cert := by
  simp [checkBBoxInsideImage] at h
  exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩

/-- Acceptance by the depth check proves that all supplied points are in front of the
camera. -/
theorem checkPositiveDepths_sound {α : Type} [TorchLean.Storage α] [OfNat α 0] [OfNat α 1] [Add α]
    [Mul α] [LT α] [DecidableRel (α := α) (· < ·)] {cert : BoxCameraCert α}
    (h : checkPositiveDepths cert = true) :
    PositiveDepths cert := by
  intro i
  have hi : decide (0 < certProjectZ cert i) = true :=
    List.all_eq_true.mp h i (List.mem_finRange i)
  exact of_decide_eq_true hi

/-- Acceptance by the projection check places every projected corner within the image rectangle. -/
theorem checkProjectedInImage_sound {α : Type} [TorchLean.Storage α] [OfNat α 0] [OfNat α 1]
    [Add α] [Mul α] [Div α] [LE α] [DecidableRel (α := α) (· ≤ ·)]
    {cert : BoxCameraCert α} (h : checkProjectedInImage cert = true) :
    ProjectedInImage cert := by
  intro i
  have hi :
      decide
         (0 ≤ certProjectX cert i ∧ certProjectX cert i ≤ cert.width ∧
         0 ≤ certProjectY cert i ∧ certProjectY cert i ≤ cert.height) = true :=
    List.all_eq_true.mp h i (List.mem_finRange i)
  exact of_decide_eq_true hi

/-- Acceptance by the enclosure check places every projected corner inside the claimed box
tolerance. -/
theorem checkBBoxEnclosesProjection_sound {α : Type} [TorchLean.Storage α] [OfNat α 1]
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
    List.all_eq_true.mp h i (List.mem_finRange i)
  exact of_decide_eq_true hi

/--
Soundness of the executable checker.

This is the main theorem for the implementation: if the Boolean checker accepts an artifact, the
artifact satisfies the mathematical `Verified3DBox` predicate.
-/
theorem checkCert_sound {α : Type} [TorchLean.Storage α] [OfNat α 0] [OfNat α 1] [Add α] [Sub α]
    [Mul α] [Div α] [LE α] [LT α] [DecidableRel (α := α) (· ≤ ·)] [DecidableRel (α := α) (· < ·)]
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

/-- A verified artifact necessarily has positive image width and height. -/
theorem Verified3DBox.positive_image_size {α : Type} [TorchLean.Storage α] [OfNat α 0] [OfNat α 1]
    [Add α] [Sub α] [Mul α] [Div α] [LE α] [LT α]
    {cert : BoxCameraCert α} (h : Verified3DBox cert) :
    0 < cert.width ∧ 0 < cert.height :=
  h.1

/-- The claimed 2D bounding box of a verified artifact lies within its image dimensions. -/
theorem Verified3DBox.bbox_inside_image {α : Type} [TorchLean.Storage α] [OfNat α 0] [OfNat α 1]
    [Add α] [Sub α] [Mul α] [Div α] [LE α] [LT α]
    {cert : BoxCameraCert α} (h : Verified3DBox cert) :
    0 ≤ xmin cert ∧ xmax cert ≤ cert.width ∧
      0 ≤ ymin cert ∧ ymax cert ≤ cert.height :=
  h.2.2.1

/-- Every corner of a verified artifact has positive camera-space depth. -/
theorem Verified3DBox.corner_positive_depth {α : Type} [TorchLean.Storage α] [OfNat α 0] [OfNat α 1]
    [Add α] [Sub α] [Mul α] [Div α] [LE α] [LT α]
    {cert : BoxCameraCert α} (h : Verified3DBox cert) (i : Fin cert.pointCount) :
    0 < certProjectZ cert i :=
  h.2.2.2.1 i

/-- Every projected corner of a verified artifact lies within the image rectangle. -/
theorem Verified3DBox.projected_corner_in_image {α : Type} [TorchLean.Storage α] [OfNat α 0]
    [OfNat α 1] [Add α] [Sub α] [Mul α] [Div α] [LE α] [LT α]
    {cert : BoxCameraCert α} (h : Verified3DBox cert) (i : Fin cert.pointCount) :
    0 ≤ certProjectX cert i ∧ certProjectX cert i ≤ cert.width ∧
      0 ≤ certProjectY cert i ∧ certProjectY cert i ≤ cert.height :=
  h.2.2.2.2.1 i

/-- Every projected corner lies within the verified artifact's claimed box, up to its
tolerance. -/
theorem Verified3DBox.projected_corner_in_claimed_bbox {α : Type} [TorchLean.Storage α] [OfNat α 0]
    [OfNat α 1] [Add α] [Sub α] [Mul α] [Div α] [LE α] [LT α]
    {cert : BoxCameraCert α} (h : Verified3DBox cert) (i : Fin cert.pointCount) :
    xmin cert - cert.tol ≤ certProjectX cert i ∧
      certProjectX cert i ≤ xmax cert + cert.tol ∧
      ymin cert - cert.tol ≤ certProjectY cert i ∧
      certProjectY cert i ≤ ymax cert + cert.tol :=
  h.2.2.2.2.2 i

/-! ## Float JSON checker for exported artifacts -/

/-- Expected schema string for JSON artifacts accepted by this checker. -/
def formatString : String := "torchlean.camera.box3d.v1"

/--
Parse a certificate, rejecting malformed tensor dimensions and non-finite values.
The optional `point_count` must match the number of triples in `corners3d`; otherwise the count
is inferred from that array. An incomplete triple is rejected by the matrix-size check.
-/
def parseJsonCert (j : Lean.Json) : IO (BoxCameraCert Float) := do
  expectFormat j formatString
  let width ← expectFiniteFloat (← expectField j "image_width" "top-level") "top-level.image_width"
  let height ←
    expectFiniteFloat (← expectField j "image_height" "top-level") "top-level.image_height"
  let tol ← expectFiniteFloat (← expectField j "tol" "top-level") "top-level.tol"
  let cameraFlat ← expectFieldFiniteFloatArray j "camera_P" "top-level"
  let cornersFlat ← expectFieldFiniteFloatArray j "corners3d" "top-level"
  let bboxFlat ← expectFieldFiniteFloatArray j "bbox2d" "top-level"
  let camera : CameraP Float ← NN.Verification.Util.Tensor.requireMatOfFlatArray
    "top-level.camera_P" 3 4 cameraFlat
  let pointCount ← match ← optionalField? j "point_count" "top-level" with
    | some count => expectNat count "top-level.point_count"
    | none => pure (cornersFlat.size / 3)
  let corners : TorchLean.Tensor Float [pointCount, 3] ←
    NN.Verification.Util.Tensor.requireMatOfFlatArray
      "top-level.corners3d" pointCount 3 cornersFlat
  let bbox : Box2D Float ← NN.Verification.Util.Tensor.requireVecOfArray
    "top-level.bbox2d" 4 bboxFlat
  pure {
    pointCount := pointCount
    width := width
    height := height
    tol := tol
    camera := camera
    corners := corners
    bbox := bbox
  }

/--
Load and check a JSON 3D-box/camera certificate.

This is the bridge to Cube R-CNN/SAM-3D-style outputs. The JSON is untrusted; Lean recomputes the
projection and accepts only if the tensor artifact passes the checker.
-/
def checkFile (path : String) : IO Bool := do
  let j ← readJsonObjectFile path
  let cert ← parseJsonCert j
  let ok := checkCert (α := Float) cert
  if ok then
    IO.println s!"3D camera-box certificate verified: {path}"
  else
    IO.println s!"3D camera-box certificate rejected: {path}"
  pure ok

/-- Check a camera-box certificate and raise a readable CLI error if it is rejected. -/
def checkOrThrow (path : String) : IO Unit := do
  let ok ← checkFile path
  if !ok then
    throw <| IO.userError s!"camera-box3d certificate check failed: {path}"

end NN.Verification.Geometry3D.Box3D
