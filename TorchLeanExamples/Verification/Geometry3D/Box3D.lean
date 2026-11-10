/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Numeric.Interval
public import TorchLean.Verify.JSON
public import TorchLean.Verify.Tensor

/-!
# Tensor-native 3D box camera certificates

Cube R-CNN, SAM 3D and similar systems are treated here as untrusted producers of tensors. One
exported artifact carries a camera projection matrix `P : Tensor α [3,4]`, a configurable number
of 3D points `corners : Tensor α [pointCount,3]`, the image dimensions and a claimed 2D enclosing
box. The checker recomputes the pinhole projection of every point and accepts when

1. the image has positive width and height, and the tolerance is nonnegative,
2. the claimed 2D box is ordered and lies inside the image,
3. every point has positive camera depth,
4. every projected pixel lies inside the image, and
5. the claimed box encloses every projected point, up to the tolerance.

Eight points, the corners of a cuboid, are the default. The conditions are pointwise over whatever
points the artifact supplies, so the cuboid shape itself is not checked. The JSON parser rejects an
artifact with no points, since the pointwise conditions would then hold vacuously.

Matrices, points and boxes use the shared tensor API. The interval theorems hold in any ordered
field and take the input enclosures as hypotheses. The JSON workflow checks host `Float`
projections of one artifact. It does not produce those enclosures, and no theorem here connects
its `Float` arithmetic to the ordered-field theorems.
-/

@[expose] public section

namespace TorchLean.Examples.Geometry3D

/-!
## Small interval arithmetic layer

Intervals are FloatLib's endpoint pairs `FloatLib.Numerics.Interval α`, added exactly by
`Numeric.intervalAdd`. The pair does not store `lo ≤ hi`, so the ordered-field
camera uncertainty theorems get the ordering from their membership hypotheses.
-/

/-- Membership in a closed scalar interval. -/
def InInterval {α : Type} [LE α] (I : FloatLib.Numerics.Interval α) (x : α) : Prop :=
  I.lo ≤ x ∧ x ≤ I.hi

/-- The interval has nonnegative lower endpoint, hence all members are nonnegative. -/
def NonnegativeInterval {α : Type} [OfNat α 0] [LE α] (I : FloatLib.Numerics.Interval α) :
    Prop :=
  0 ≤ I.lo

/--
Interval product $[a,b][c,d]=[ac,bd]$, which encloses every product of members when both intervals
are nonnegative (`mulNonnegInterval_sound`).
-/
def mulNonnegInterval {α : Type} [Mul α] (I J : FloatLib.Numerics.Interval α) :
    FloatLib.Numerics.Interval α where
  lo := I.lo * J.lo
  hi := I.hi * J.hi

/--
Interval quotient $[a,b]/[c,d]=[a/d,b/c]$ for a nonnegative numerator and a strictly positive
denominator, that is $0\le a$ and $0<c$.

A pinhole projection is affine camera arithmetic followed by this perspective division of the
homogeneous pixel numerators by a positive depth. Keeping the division as its own step puts the
trust boundary in one place: an exporter may supply interval enclosures for `u_num`, `v_num` and
`z`, and Lean proves the quotient enclosure sound (`divNonnegByPosInterval_sound`).
-/
def divNonnegByPosInterval {α : Type} [Div α] (num den : FloatLib.Numerics.Interval α) :
    FloatLib.Numerics.Interval α where
  lo := num.lo / den.hi
  hi := num.hi / den.lo

/--
Soundness of `mulNonnegInterval`. If `I` and `J` have nonnegative lower endpoints, multiplication
is monotone over both intervals, so $x\in I$ and $y\in J$ imply
$xy\in[I_{\mathrm{lo}}J_{\mathrm{lo}},I_{\mathrm{hi}}J_{\mathrm{hi}}]$.
-/
theorem mulNonnegInterval_sound
    {α : Type} [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {I J : FloatLib.Numerics.Interval α} {x y : α}
    (hI : NonnegativeInterval I) (hJ : NonnegativeInterval J)
    (hx : InInterval I x) (hy : InInterval J y) :
    InInterval (mulNonnegInterval I J) (x * y) := by
  rcases hx with ⟨hxlo, hxhi⟩
  rcases hy with ⟨hylo, hyhi⟩
  have hx_nonneg : 0 ≤ x := le_trans hI hxlo
  have hy_nonneg : 0 ≤ y := le_trans hJ hylo
  have hIhi_nonneg : 0 ≤ I.hi := le_trans hx_nonneg hxhi
  dsimp [InInterval, mulNonnegInterval]
  exact ⟨mul_le_mul hxlo hylo hJ hx_nonneg, mul_le_mul hxhi hyhi hy_nonneg hIhi_nonneg⟩

/--
Soundness of perspective-style interval division. If `num` has a nonnegative lower endpoint, the
denominator has a positive lower endpoint (`0 < den.lo`), $x\in\mathrm{num}$ and
$z\in\mathrm{den}$, then
$x/z\in[\mathrm{num}_{\mathrm{lo}}/\mathrm{den}_{\mathrm{hi}},
\mathrm{num}_{\mathrm{hi}}/\mathrm{den}_{\mathrm{lo}}]$.
-/
theorem divNonnegByPosInterval_sound
    {α : Type} [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {num den : FloatLib.Numerics.Interval α} {x z : α}
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
  exact ⟨div_le_div₀ hx_nonneg hxlo hz_pos hzhi, div_le_div₀ hnum_hi_nonneg hxhi hdenPos hzlo⟩

/--
Pinhole pixel interval for one axis, from intervals for the focal length, the normalized
coordinate and the principal point. For $u=f_xx_n+c_x$, with $f_x\ge 0$ and $x_n\ge 0$, interval
arithmetic gives $u\in f_{x,I}x_{n,I}+c_{x,I}$, and the y axis uses the same function.
-/
def pinholeInterval {α : Type} [Add α] [Mul α]
    (fI coordI cI : FloatLib.Numerics.Interval α) : FloatLib.Numerics.Interval α :=
  Numeric.intervalAdd (mulNonnegInterval fI coordI) cI

/--
Soundness of the one-axis pinhole interval formula. If the focal-length and normalized-coordinate
intervals are nonnegative and the focal length, normalized coordinate and principal point lie in
their intervals, then the pixel coordinate `f * coord + c` lies in the interval computed by
`pinholeInterval`.
-/
theorem pinholeInterval_sound
    {α : Type} [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {fI coordI cI : FloatLib.Numerics.Interval α} {f coord c : α}
    (hfNonneg : NonnegativeInterval fI)
    (hcoordNonneg : NonnegativeInterval coordI)
    (hf : InInterval fI f)
    (hcoord : InInterval coordI coord)
    (hc : InInterval cI c) :
    InInterval (pinholeInterval fI coordI cI) (f * coord + c) :=
  Numeric.intervalAdd_sound
    (mulNonnegInterval_sound hfNonneg hcoordNonneg hf hcoord)
    hc

/-- A pair of x/y pixel intervals enclosing a projected point under camera uncertainty. -/
structure PixelInterval2D (α : Type) where
  /-- Pixel-x interval. -/
  x : FloatLib.Numerics.Interval α
  /-- Pixel-y interval. -/
  y : FloatLib.Numerics.Interval α

/-- A concrete pixel lies inside a 2D pixel interval. -/
def PixelInInterval2D {α : Type} [LE α] (pix : PixelInterval2D α) (px py : α) : Prop :=
  InInterval pix.x px ∧ InInterval pix.y py

/--
Pixel interval of a homogeneous projection. `uNumI` and `vNumI` enclose the first two rows of
$P[X,Y,Z,1]^\mathsf{T}$, `zI` encloses the positive depth row, and each numerator interval is
divided by `zI`, since the projected pixel is $(u_{\mathrm{num}}/z,v_{\mathrm{num}}/z)$.
-/
def PixelInterval2D.ofHomogeneous {α : Type} [Div α]
    (uNumI vNumI zI : FloatLib.Numerics.Interval α) : PixelInterval2D α where
  x := divNonnegByPosInterval uNumI zI
  y := divNonnegByPosInterval vNumI zI

/-- A 3D point as a tensor of shape `[3]`. -/
abbrev Point3 (α : Type) [Storage α] := Tensor α [3]

/--
A 3×4 camera projection matrix.

For a homogeneous point $[X,Y,Z,1]$, the raw camera coordinates are:

$$
u_{\mathrm{num}}=P_0\mathbin{\cdot}[X,Y,Z,1],\qquad
v_{\mathrm{num}}=P_1\mathbin{\cdot}[X,Y,Z,1],\qquad
z=P_2\mathbin{\cdot}[X,Y,Z,1].
$$

The projected pixel is $(u_{\mathrm{num}}/z,v_{\mathrm{num}}/z)$, and the checker requires $z>0$
for every point.
-/
abbrev CameraP (α : Type) [Storage α] := Tensor α [3, 4]

/-- A 2D box `[xmin, ymin, xmax, ymax]`. -/
abbrev Box2D (α : Type) [Storage α] := Tensor α [4]

/-- Extract the `i`-th supplied 3D point as a `[3]` tensor. -/
def corner {α : Type} [Storage α] {pointCount : Nat}
    (corners : Tensor α [pointCount, 3]) (i : Fin pointCount) : Point3 α :=
  Tensor.dim (fun j => Tensor.scalar (Tensor.get2 corners i j))

/-- Raw homogeneous camera coordinate `P[row] · [X,Y,Z,1]`. -/
def cameraCoord {α : Type} [Storage α] [OfNat α 1] [Add α] [Mul α]
    (P : CameraP α) (x : Point3 α) (row : Fin 3) : α :=
  Tensor.get2 P row 0 * Tensor.getScalar x 0 +
  Tensor.get2 P row 1 * Tensor.getScalar x 1 +
  Tensor.get2 P row 2 * Tensor.getScalar x 2 +
  Tensor.get2 P row 3 * (1 : α)

/-- Camera depth `P[2] · [X,Y,Z,1]`, the denominator of the pinhole projection. -/
def projectZ {α : Type} [Storage α] [OfNat α 1] [Add α] [Mul α]
    (P : CameraP α) (x : Point3 α) : α :=
  cameraCoord P x 2

/-- Projected pixel x-coordinate, meaningful when `projectZ P x ≠ 0`. -/
def projectX {α : Type} [Storage α] [OfNat α 1] [Add α] [Mul α] [Div α]
    (P : CameraP α) (x : Point3 α) : α :=
  cameraCoord P x 0 / projectZ P x

/-- Projected pixel y-coordinate, meaningful when `projectZ P x ≠ 0`. -/
def projectY {α : Type} [Storage α] [OfNat α 1] [Add α] [Mul α] [Div α]
    (P : CameraP α) (x : Point3 α) : α :=
  cameraCoord P x 1 / projectZ P x

/-- A compact exported 3D-box/camera certificate. -/
structure BoxCameraCert (α : Type) [Storage α] where
  /-- Number of exported 3D points (default eight, for a cuboid). -/
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
  corners : Tensor α [pointCount, 3]
  /-- Claimed 2D box `[xmin,ymin,xmax,ymax]`. -/
  bbox : Box2D α

/-- Left edge `xmin` of the claimed 2D bounding box. -/
def xmin {α : Type} [Storage α] (cert : BoxCameraCert α) : α :=
  Tensor.getScalar cert.bbox 0

/-- Top edge `ymin` of the claimed 2D bounding box. -/
def ymin {α : Type} [Storage α] (cert : BoxCameraCert α) : α :=
  Tensor.getScalar cert.bbox 1

/-- Right edge `xmax` of the claimed 2D bounding box. -/
def xmax {α : Type} [Storage α] (cert : BoxCameraCert α) : α :=
  Tensor.getScalar cert.bbox 2

/-- Bottom edge `ymax` of the claimed 2D bounding box. -/
def ymax {α : Type} [Storage α] (cert : BoxCameraCert α) : α :=
  Tensor.getScalar cert.bbox 3

/-- The pixel interval is contained in the claimed 2D box. -/
def PixelIntervalInsideBBox {α : Type} [Storage α] [LE α]
    (cert : BoxCameraCert α) (pix : PixelInterval2D α) : Prop :=
  xmin cert ≤ pix.x.lo ∧ pix.x.hi ≤ xmax cert ∧
    ymin cert ≤ pix.y.lo ∧ pix.y.hi ≤ ymax cert

/--
If a pixel interval is contained in the claimed bbox, every concrete pixel inside that interval is
also contained in the bbox.
-/
theorem pixel_inside_bbox_of_interval
    {α : Type} [Storage α] [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {cert : BoxCameraCert α} {pix : PixelInterval2D α} {px py : α}
    (hinside : PixelIntervalInsideBBox cert pix)
    (hpix : PixelInInterval2D pix px py) :
    xmin cert ≤ px ∧ px ≤ xmax cert ∧ ymin cert ≤ py ∧ py ≤ ymax cert := by
  rcases hinside with ⟨hxlo, hxhi, hylo, hyhi⟩
  rcases hpix with ⟨⟨hpxlo, hpxhi⟩, ⟨hpylo, hpyhi⟩⟩
  exact ⟨hxlo.trans hpxlo, hpxhi.trans hxhi, hylo.trans hpylo, hpyhi.trans hyhi⟩

/--
Pinhole-intrinsics interval-to-bbox theorem for nonnegative normalized coordinates.

Take independent intervals for the focal lengths (`fx`, `fy`), the normalized coordinates
$x_n=X/Z$ and $y_n=Y/Z$ and the principal point (`cx`, `cy`), with the focal-length and
normalized-coordinate intervals nonnegative. If the interval arithmetic results

$u\in f_{x,I}x_{n,I}+c_{x,I}$ and $v\in f_{y,I}y_{n,I}+c_{y,I}$

are contained in the claimed bbox, then every choice of values in those intervals projects inside
the bbox.

The nonnegativity hypotheses restrict the theorem to one quadrant. Uncertainty enters at the
camera intrinsics and normalized coordinates, passes through the pinhole equations and comes out
as a bbox-enclosure statement. Stating it over normalized coordinates lets callers bound `X/Z` and
`Y/Z` with whatever depth-interval or model-uncertainty method they use.
-/
theorem pinholeInterval_inside_bbox
    {α : Type} [Storage α] [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {cert : BoxCameraCert α}
    {fxI fyI xnI ynI cxI cyI : FloatLib.Numerics.Interval α}
    {fx fy xn yn cx cy : α}
    (hfxNonneg : NonnegativeInterval fxI)
    (hfyNonneg : NonnegativeInterval fyI)
    (hxnNonneg : NonnegativeInterval xnI)
    (hynNonneg : NonnegativeInterval ynI)
    (hinside :
      PixelIntervalInsideBBox cert
        { x := pinholeInterval fxI xnI cxI
          y := pinholeInterval fyI ynI cyI })
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
      InInterval (pinholeInterval fxI xnI cxI) (fx * xn + cx) :=
    pinholeInterval_sound hfxNonneg hxnNonneg hfx hxn hcx
  have hy :
      InInterval (pinholeInterval fyI ynI cyI) (fy * yn + cy) :=
    pinholeInterval_sound hfyNonneg hynNonneg hfy hyn hcy
  exact pixel_inside_bbox_of_interval
    (cert := cert)
    (pix := { x := pinholeInterval fxI xnI cxI
              y := pinholeInterval fyI ynI cyI })
    hinside
    ⟨hx, hy⟩

/--
Homogeneous projection interval-to-bbox theorem for nonnegative numerators.

The hypotheses bound the homogeneous camera outputs before the perspective divide:

* `uNumI` encloses the x numerator $P_0\mathbin{\cdot}[X,Y,Z,1]$,
* `vNumI` encloses the y numerator $P_1\mathbin{\cdot}[X,Y,Z,1]$,
* `zI` encloses the positive depth denominator $P_2\mathbin{\cdot}[X,Y,Z,1]$ (`0 < zI.lo`).

If both numerator intervals are nonnegative and the quotient intervals
$[u_{\mathrm{lo}}/z_{\mathrm{hi}},u_{\mathrm{hi}}/z_{\mathrm{lo}}]$ and
$[v_{\mathrm{lo}}/z_{\mathrm{hi}},v_{\mathrm{hi}}/z_{\mathrm{lo}}]$
are inside the claimed 2D box, then every projection `(uNum / z, vNum / z)` with values in those
intervals is inside the box. Uncertain intrinsics or extrinsics, bounded corner perturbations and
mixed numeric backends can all produce ranges for the homogeneous coordinates, and Lean checks the
perspective divide and the final bbox containment.
-/
theorem ofHomogeneous_inside_bbox
    {α : Type} [Storage α] [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {cert : BoxCameraCert α}
    {uNumI vNumI zI : FloatLib.Numerics.Interval α}
    {uNum vNum z : α}
    (huNumNonneg : NonnegativeInterval uNumI)
    (hvNumNonneg : NonnegativeInterval vNumI)
    (hzPos : 0 < zI.lo)
    (hinside :
      PixelIntervalInsideBBox cert
        (PixelInterval2D.ofHomogeneous uNumI vNumI zI))
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
  exact pixel_inside_bbox_of_interval
    (cert := cert)
    (pix := PixelInterval2D.ofHomogeneous uNumI vNumI zI)
    hinside
    ⟨hx, hy⟩

/-- Projected x-coordinate of corner `i`. -/
def certProjectX {α : Type} [Storage α] [OfNat α 1] [Add α] [Mul α] [Div α]
    (cert : BoxCameraCert α) (i : Fin cert.pointCount) : α :=
  projectX cert.camera (corner cert.corners i)

/-- Projected y-coordinate of corner `i`. -/
def certProjectY {α : Type} [Storage α] [OfNat α 1] [Add α] [Mul α] [Div α]
    (cert : BoxCameraCert α) (i : Fin cert.pointCount) : α :=
  projectY cert.camera (corner cert.corners i)

/-- Camera depth of corner `i`. -/
def certProjectZ {α : Type} [Storage α] [OfNat α 1] [Add α] [Mul α]
    (cert : BoxCameraCert α) (i : Fin cert.pointCount) : α :=
  projectZ cert.camera (corner cert.corners i)

/-! ## Mathematical certificate predicates -/

/-- The image width and height are positive. -/
def PositiveImageSize {α : Type} [Storage α] [OfNat α 0] [LT α] (cert : BoxCameraCert α) :
    Prop :=
  0 < cert.width ∧ 0 < cert.height

/-- The tolerance is nonnegative and the claimed 2D box is ordered on both axes. -/
def BBoxOrdered {α : Type} [Storage α] [OfNat α 0] [LE α] (cert : BoxCameraCert α) :
    Prop :=
  0 ≤ cert.tol ∧ xmin cert ≤ xmax cert ∧ ymin cert ≤ ymax cert

/-- The claimed 2D box lies inside the image frame. -/
def BBoxInsideImage {α : Type} [Storage α] [OfNat α 0] [LE α] (cert : BoxCameraCert α) :
    Prop :=
  0 ≤ xmin cert ∧ xmax cert ≤ cert.width ∧
  0 ≤ ymin cert ∧ ymax cert ≤ cert.height

/-- Every 3D corner lies in front of the camera. -/
def PositiveDepths {α : Type} [Storage α] [OfNat α 0] [OfNat α 1] [Add α] [Mul α] [LT α]
    (cert : BoxCameraCert α) : Prop :=
  ∀ i : Fin cert.pointCount, 0 < certProjectZ cert i

/-- Every projected corner lies inside the image rectangle. -/
def ProjectedInImage {α : Type} [Storage α] [OfNat α 0] [OfNat α 1] [Add α] [Mul α]
    [Div α] [LE α] (cert : BoxCameraCert α) : Prop :=
  ∀ i : Fin cert.pointCount,
    0 ≤ certProjectX cert i ∧ certProjectX cert i ≤ cert.width ∧
    0 ≤ certProjectY cert i ∧ certProjectY cert i ≤ cert.height

/-- The reported 2D box encloses all projected 3D corners, up to tolerance. -/
def Encloses {α : Type} [Storage α] [OfNat α 1] [Add α] [Sub α] [Mul α]
    [Div α] [LE α] (cert : BoxCameraCert α) : Prop :=
  ∀ i : Fin cert.pointCount,
    xmin cert - cert.tol ≤ certProjectX cert i ∧
      certProjectX cert i ≤ xmax cert + cert.tol ∧
    ymin cert - cert.tol ≤ certProjectY cert i ∧
      certProjectY cert i ≤ ymax cert + cert.tol

/-! ## Robust projection predicates

The certificate predicates above describe the exported projection at one numeric point. Real model
outputs also need statements that survive uncertainty around that point, and there are two of them
here:

* `ofHomogeneous_inside_bbox` starts from nonnegative
  intervals over the homogeneous camera outputs `(u_num, v_num, z)` and proves that the
  perspective-divided pixels stay inside the claimed box.
* `enclosesPerturbed_of_margin` starts from a nominal projected pixel and proves that any
  bounded pixel-space perturbation stays inside the claimed box when there is enough slack.

They match two exporter contracts: interval-enclosed camera arithmetic and bounded downstream
pixel perturbation.
-/

/--
The projected corners are inside the claimed 2D box with slack `margin`, that is inside
$[x_{\min}+\mathrm{margin},x_{\max}-\mathrm{margin}]
\times[y_{\min}+\mathrm{margin},y_{\max}-\mathrm{margin}]$.

With a nonnegative margin and tolerance the predicate is stricter than `Encloses`,
and the slack is what absorbs a later perturbation (`enclosesPerturbed_of_margin`).
-/
def EnclosesWithMargin {α : Type} [Storage α] [OfNat α 1] [Add α] [Sub α]
    [Mul α] [Div α] [LE α] (cert : BoxCameraCert α) (margin : α) : Prop :=
  ∀ i : Fin cert.pointCount,
    xmin cert + margin ≤ certProjectX cert i ∧
      certProjectX cert i ≤ xmax cert - margin ∧
    ymin cert + margin ≤ certProjectY cert i ∧
      certProjectY cert i ≤ ymax cert - margin

/--
`px, py` describe an interval-perturbed projection of every corner: for each corner `i`, the
coordinates `px i` and `py i` lie within `eps` pixels of the nominal projection recomputed from
the certificate. The perturbation may come from the exporter, a renderer, rounding or model
uncertainty.
-/
def ProjectionPerturbationWithin {α : Type} [Storage α] [OfNat α 1] [Add α] [Sub α]
    [Mul α] [Div α] [LE α] (cert : BoxCameraCert α) (eps : α)
    (px py : Fin cert.pointCount → α) : Prop :=
  ∀ i : Fin cert.pointCount,
    certProjectX cert i - eps ≤ px i ∧
      px i ≤ certProjectX cert i + eps ∧
    certProjectY cert i - eps ≤ py i ∧
      py i ≤ certProjectY cert i + eps

/--
The claimed 2D box encloses every perturbed projected corner. Downstream consumers want this
postcondition: after a bounded perturbation, the candidate 3D artifact projects inside the
detector's claimed 2D box.
-/
def EnclosesPerturbed {α : Type} [Storage α] [LE α]
    (cert : BoxCameraCert α) (px py : Fin cert.pointCount → α) : Prop :=
  ∀ i : Fin cert.pointCount,
    xmin cert ≤ px i ∧ px i ≤ xmax cert ∧
      ymin cert ≤ py i ∧ py i ≤ ymax cert

/--
Projection-interval robustness theorem. If every nominal projected corner has at least `margin`
pixels of slack inside the box, and every perturbed projection is within
$\varepsilon\le\mathrm{margin}$ pixels of that nominal projection, then every perturbed projection
lies inside the claimed 2D box.

The perturbation can come from anywhere: Float rounding, a renderer approximation, a model
uncertainty interval or an external exporter.
-/
theorem enclosesPerturbed_of_margin
    {α : Type} [Storage α] [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {cert : BoxCameraCert α} {eps margin : α} {px py : Fin cert.pointCount → α}
    (hle : eps ≤ margin)
    (hmargin : EnclosesWithMargin cert margin)
    (hperturb : ProjectionPerturbationWithin cert eps px py) :
    EnclosesPerturbed cert px py := by
  intro i
  rcases hmargin i with ⟨hxlo, hxhi, hylo, hyhi⟩
  rcases hperturb i with ⟨hpxlo, hpxhi, hpylo, hpyhi⟩
  refine ⟨?_, ?_, ?_, ?_⟩ <;> linarith

/--
The statement the certificate workflow checks: the exported detector artifact is coherent under its
own camera model. The image has positive dimensions, the tolerance is nonnegative and the claimed 2D
box is ordered and inside the frame. Every point has positive camera depth and projects inside the
image, and the claimed box encloses every projection up to the tolerance.
-/
def Verified {α : Type} [Storage α] [OfNat α 0] [OfNat α 1] [Add α] [Sub α] [Mul α]
    [Div α] [LE α] [LT α] (cert : BoxCameraCert α) : Prop :=
  PositiveImageSize cert ∧
    BBoxOrdered cert ∧
    BBoxInsideImage cert ∧
    PositiveDepths cert ∧
    ProjectedInImage cert ∧
    Encloses cert

/-! ## Boolean checker and soundness theorems -/

/-- Boolean check for positive image dimensions. -/
def checkPositiveImageSize {α : Type} [Storage α] [OfNat α 0] [LT α]
    [DecidableRel (α := α) (· < ·)] (cert : BoxCameraCert α) : Bool :=
  decide (0 < cert.width) && decide (0 < cert.height)

/-- Boolean check for `BBoxOrdered`. -/
def checkBBoxOrdered {α : Type} [Storage α] [OfNat α 0] [LE α]
    [DecidableRel (α := α) (· ≤ ·)] (cert : BoxCameraCert α) : Bool :=
  decide (0 ≤ cert.tol) && decide (xmin cert ≤ xmax cert) && decide (ymin cert ≤ ymax cert)

/-- Boolean check for `BBoxInsideImage`. -/
def checkBBoxInsideImage {α : Type} [Storage α] [OfNat α 0] [LE α]
    [DecidableRel (α := α) (· ≤ ·)] (cert : BoxCameraCert α) : Bool :=
  decide (0 ≤ xmin cert) &&
    decide (xmax cert ≤ cert.width) &&
    decide (0 ≤ ymin cert) &&
    decide (ymax cert ≤ cert.height)

/-- Boolean check for `PositiveDepths`. -/
def checkPositiveDepths {α : Type} [Storage α] [OfNat α 0] [OfNat α 1] [Add α] [Mul α]
    [LT α] [DecidableRel (α := α) (· < ·)] (cert : BoxCameraCert α) : Bool :=
  (List.finRange cert.pointCount).all (fun i => decide (0 < certProjectZ cert i))

/-- Boolean check for `ProjectedInImage`. -/
def checkProjectedInImage {α : Type} [Storage α] [OfNat α 0] [OfNat α 1] [Add α] [Mul α]
    [Div α] [LE α] [DecidableRel (α := α) (· ≤ ·)] (cert : BoxCameraCert α) : Bool :=
  (List.finRange cert.pointCount).all (fun i =>
    decide
      (0 ≤ certProjectX cert i ∧ certProjectX cert i ≤ cert.width ∧
       0 ≤ certProjectY cert i ∧ certProjectY cert i ≤ cert.height))

/-- Boolean check for `Encloses`. -/
def checkEncloses {α : Type} [Storage α] [OfNat α 1] [Add α] [Sub α] [Mul α]
    [Div α] [LE α] [DecidableRel (α := α) (· ≤ ·)] (cert : BoxCameraCert α) : Bool :=
  (List.finRange cert.pointCount).all (fun i =>
    decide
      (xmin cert - cert.tol ≤ certProjectX cert i ∧
       certProjectX cert i ≤ xmax cert + cert.tol ∧
       ymin cert - cert.tol ≤ certProjectY cert i ∧
       certProjectY cert i ≤ ymax cert + cert.tol))

/-- Full Boolean checker for one exported 3D-box/camera artifact. -/
def checkCert {α : Type} [Storage α] [OfNat α 0] [OfNat α 1] [Add α] [Sub α] [Mul α]
    [Div α] [LE α] [LT α] [DecidableRel (α := α) (· ≤ ·)] [DecidableRel (α := α) (· < ·)]
    (cert : BoxCameraCert α) : Bool :=
  checkPositiveImageSize cert &&
    checkBBoxOrdered cert &&
    checkBBoxInsideImage cert &&
    checkPositiveDepths cert &&
    checkProjectedInImage cert &&
    checkEncloses cert

/-- Acceptance by the image-size check yields the corresponding positive-size proposition. -/
theorem checkPositiveImageSize_sound {α : Type} [Storage α] [OfNat α 0] [LT α]
    [DecidableRel (α := α) (· < ·)] {cert : BoxCameraCert α}
    (h : checkPositiveImageSize cert = true) :
    PositiveImageSize cert := by
  simp [checkPositiveImageSize] at h
  exact h

/-- Acceptance by the box-order check yields a nonnegative tolerance and ordered box endpoints. -/
theorem checkBBoxOrdered_sound {α : Type} [Storage α] [OfNat α 0] [LE α]
    [DecidableRel (α := α) (· ≤ ·)] {cert : BoxCameraCert α}
    (h : checkBBoxOrdered cert = true) :
    BBoxOrdered cert := by
  simp [checkBBoxOrdered] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

/-- Acceptance by the containment check places every claimed box edge inside the image. -/
theorem checkBBoxInsideImage_sound {α : Type} [Storage α] [OfNat α 0] [LE α]
    [DecidableRel (α := α) (· ≤ ·)] {cert : BoxCameraCert α}
    (h : checkBBoxInsideImage cert = true) :
    BBoxInsideImage cert := by
  simp [checkBBoxInsideImage] at h
  exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩

/-- Acceptance by the depth check proves that all supplied points are in front of the
camera. -/
theorem checkPositiveDepths_sound {α : Type} [Storage α] [OfNat α 0] [OfNat α 1] [Add α]
    [Mul α] [LT α] [DecidableRel (α := α) (· < ·)] {cert : BoxCameraCert α}
    (h : checkPositiveDepths cert = true) :
    PositiveDepths cert := by
  intro i
  have hi : decide (0 < certProjectZ cert i) = true :=
    List.all_eq_true.mp h i (List.mem_finRange i)
  exact of_decide_eq_true hi

/-- Acceptance by the projection check places every projected corner within the image rectangle. -/
theorem checkProjectedInImage_sound {α : Type} [Storage α] [OfNat α 0] [OfNat α 1]
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

/-- Acceptance by the enclosure check places every projected corner inside the claimed box, up to
the tolerance. -/
theorem checkEncloses_sound {α : Type} [Storage α] [OfNat α 1]
    [Add α] [Sub α] [Mul α] [Div α] [LE α] [DecidableRel (α := α) (· ≤ ·)]
    {cert : BoxCameraCert α} (h : checkEncloses cert = true) :
    Encloses cert := by
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
Soundness of the executable checker: if `checkCert` accepts an artifact, the artifact satisfies
the mathematical `Verified` predicate. The consequence theorems below unpack that predicate.
-/
theorem checkCert_sound {α : Type} [Storage α] [OfNat α 0] [OfNat α 1] [Add α] [Sub α]
    [Mul α] [Div α] [LE α] [LT α] [DecidableRel (α := α) (· ≤ ·)] [DecidableRel (α := α) (· < ·)]
    {cert : BoxCameraCert α} (h : checkCert cert = true) :
    Verified cert := by
  simp [checkCert] at h
  rcases h with ⟨⟨⟨⟨⟨hSize, hBox⟩, hBoxIn⟩, hDepth⟩, hImage⟩, hEnclose⟩
  exact ⟨
    checkPositiveImageSize_sound hSize,
    checkBBoxOrdered_sound hBox,
    checkBBoxInsideImage_sound hBoxIn,
    checkPositiveDepths_sound hDepth,
    checkProjectedInImage_sound hImage,
    checkEncloses_sound hEnclose
  ⟩

/-! ## Consequence theorems for verified artifacts -/

/-- A verified artifact has positive image width and height. -/
theorem Verified.positive_image_size {α : Type} [Storage α] [OfNat α 0] [OfNat α 1]
    [Add α] [Sub α] [Mul α] [Div α] [LE α] [LT α]
    {cert : BoxCameraCert α} (h : Verified cert) :
    0 < cert.width ∧ 0 < cert.height :=
  h.1

/-- The claimed 2D bounding box of a verified artifact lies within its image dimensions. -/
theorem Verified.bbox_inside_image {α : Type} [Storage α] [OfNat α 0] [OfNat α 1]
    [Add α] [Sub α] [Mul α] [Div α] [LE α] [LT α]
    {cert : BoxCameraCert α} (h : Verified cert) :
    0 ≤ xmin cert ∧ xmax cert ≤ cert.width ∧
      0 ≤ ymin cert ∧ ymax cert ≤ cert.height :=
  h.2.2.1

/-- Every corner of a verified artifact has positive camera-space depth. -/
theorem Verified.certProjectZ_pos {α : Type} [Storage α] [OfNat α 0] [OfNat α 1]
    [Add α] [Sub α] [Mul α] [Div α] [LE α] [LT α]
    {cert : BoxCameraCert α} (h : Verified cert) (i : Fin cert.pointCount) :
    0 < certProjectZ cert i :=
  h.2.2.2.1 i

/-- Every projected corner of a verified artifact lies within the image rectangle. -/
theorem Verified.corner_in_image {α : Type} [Storage α] [OfNat α 0]
    [OfNat α 1] [Add α] [Sub α] [Mul α] [Div α] [LE α] [LT α]
    {cert : BoxCameraCert α} (h : Verified cert) (i : Fin cert.pointCount) :
    0 ≤ certProjectX cert i ∧ certProjectX cert i ≤ cert.width ∧
      0 ≤ certProjectY cert i ∧ certProjectY cert i ≤ cert.height :=
  h.2.2.2.2.1 i

/-- Every projected corner lies within the verified artifact's claimed box, up to its
tolerance. -/
theorem Verified.corner_in_bbox {α : Type} [Storage α] [OfNat α 0]
    [OfNat α 1] [Add α] [Sub α] [Mul α] [Div α] [LE α] [LT α]
    {cert : BoxCameraCert α} (h : Verified cert) (i : Fin cert.pointCount) :
    xmin cert - cert.tol ≤ certProjectX cert i ∧
      certProjectX cert i ≤ xmax cert + cert.tol ∧
      ymin cert - cert.tol ≤ certProjectY cert i ∧
      certProjectY cert i ≤ ymax cert + cert.tol :=
  h.2.2.2.2.2 i

/-! ## Float JSON checker for exported artifacts -/

/--
Schema string for camera-box artifacts. An artifact declares `point_count` (`8` for a cuboid),
which must be positive and match the number of triples in `corners3d`.
-/
def formatString : String := "torchlean.camera.box3d.v2"

/--
Parse a certificate, rejecting a wrong `format`, malformed tensor dimensions and non-finite values.

The artifact must declare a positive `point_count` with `corners3d.size = 3 * point_count`. An
artifact with no points is rejected, since every pointwise condition would hold vacuously.
-/
def parseJsonCert (j : Lean.Json) : IO (BoxCameraCert Float) := do
  let o ← IO.ofExcept <| Json.object "top-level" j
  let format ← IO.ofExcept <|
    Json.field "top-level" "format" o >>= Json.string "top-level.format"
  IO.ofExcept <| Json.expectFormat format formatString
  let pointCount ← match o.get? "point_count" with
    | some count => IO.ofExcept <| Json.expectNat "top-level.point_count" count
    | none => throw <| IO.userError s!"top-level.point_count: required by `{formatString}`"
  if pointCount = 0 then
    throw <| IO.userError "top-level.point_count: a certificate must carry at least one point"
  let width ← IO.ofExcept <|
    Json.field "top-level" "image_width" o >>=
      Json.parseFiniteFloat "top-level.image_width"
  let height ← IO.ofExcept <|
    Json.field "top-level" "image_height" o >>=
      Json.parseFiniteFloat "top-level.image_height"
  let tol ← IO.ofExcept <|
    Json.field "top-level" "tol" o >>= Json.parseFiniteFloat "top-level.tol"
  let cameraFlat ← IO.ofExcept <|
    Json.field "top-level" "camera_P" o >>=
      Json.parseFiniteFloatArray "top-level.camera_P"
  let cornersFlat ← IO.ofExcept <|
    Json.field "top-level" "corners3d" o >>=
      Json.parseFiniteFloatArray "top-level.corners3d"
  let bboxFlat ← IO.ofExcept <|
    Json.field "top-level" "bbox2d" o >>=
      Json.parseFiniteFloatArray "top-level.bbox2d"
  let camera : CameraP Float ← Tensor.requireArray
    "top-level.camera_P" [3, 4] cameraFlat
  let corners : Tensor Float [pointCount, 3] ←
    Tensor.requireArray
      "top-level.corners3d" [pointCount, 3] cornersFlat
  let bbox : Box2D Float ← Tensor.requireArray
    "top-level.bbox2d" [4] bboxFlat
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
Load and check a JSON 3D-box/camera certificate, such as one built from Cube R-CNN or SAM 3D
outputs. The JSON is untrusted: Lean recomputes the projection in `Float`, prints the verdict and
returns `true` only if every projected depth is finite and `checkCert` accepts.
-/
def checkFile (path : String) : IO Bool := do
  let cert ← parseJsonCert (← Json.readFile path)
  let finiteDepths := (List.finRange cert.pointCount).all fun i => (certProjectZ cert i).isFinite
  let ok := finiteDepths && checkCert (α := Float) cert
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

end TorchLean.Examples.Geometry3D
