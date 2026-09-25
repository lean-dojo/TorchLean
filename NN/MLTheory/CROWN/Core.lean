/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.BoundOps
public import NN.MLTheory.CROWN.Flatbox
public import NN.Spec.Core.TensorOps

/-!
# CROWN Core

CROWN core definitions for TorchLean.

This file defines:
- Box: element-wise lower/upper bounds (intervals) on tensors
- AffineBounds: per-output affine forms w.r.t. input x, and constant
- Utilities for splitting positive/negative parts and safe affine eval

References:
- Zhang et al.,
  "Efficient Neural Network Robustness Certification with General Activation Functions" (CROWN),
  arXiv:1811.00866.
- Xu et al., "Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond"
  (auto_LiRPA), NeurIPS 2020, arXiv:2002.12920. This generalizes LiRPA/CROWN to arbitrary
  computational graphs and introduces techniques (e.g., loss fusion) widely used in scalable
  certified training.
-/

@[expose] public section


namespace NN.MLTheory.CROWN

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/--
Interval box `lo <= x <= hi` over a tensor shape.

This is the fundamental object for interval bound propagation (IBP).
-/
structure Box (α : Type) [TorchLean.Storage α] (s : Shape) where
  /-- Elementwise lower bound. -/
  lo : Tensor α s
  /-- Elementwise upper bound. -/
  hi : Tensor α s

namespace Box

variable {s : Shape}

/-- Pointwise box width `hi - lo`. -/
def width (b : Box α s) : Tensor α s :=
  Tensor.subSpec b.hi b.lo

/-- Pointwise box center `(lo + hi)/2`. -/
def center (b : Box α s) : Tensor α s :=
  Tensor.scaleSpec (Tensor.addSpec b.lo b.hi) ((1 / 2))

/-- Pointwise box radius `(hi - lo)/2`. -/
def radius (b : Box α s) : Tensor α s :=
  Tensor.scaleSpec (Tensor.subSpec b.hi b.lo) ((1 / 2))

/--
Boolean `a <= b` using the backend's strict order and Boolean equality.

Reflexivity guards reject NaNs even on backends whose strict comparison uses a total
implementation order. For a logical containment theorem, use `containsDecBool`.
-/
def leBool (a b : α) : Bool :=
  a == a && b == b && (decide (b > a) || a == b)

/--
Executable containment check: returns `true` iff every component is within bounds.

This uses `leBool` and is therefore available for any `Context α`.
-/
def containsBool : ∀ {s : Shape}, Box α s → Tensor α s → Bool
| .scalar, b, x =>
  leBool b.lo.item x.item && leBool x.item b.hi.item
| .dim n inner, b, x =>
  (List.finRange n).foldl
    (fun acc i =>
      acc && containsBool (s := inner)
        ⟨b.lo.unstack i, b.hi.unstack i⟩ (x.unstack i))
    true

/-- Logical containment: every component of `x` lies between `lo` and `hi`. -/
def contains : ∀ {s : Shape}, Box α s → Tensor α s → Prop
| .scalar, b, x => b.lo.item ≤ x.item ∧ x.item ≤ b.hi.item
| .dim _ inner, b, x =>
  ∀ i, contains (s := inner) ⟨b.lo.unstack i, b.hi.unstack i⟩ (x.unstack i)

/-!
`containsBool` above is a minimal `Context`-only checker that avoids requiring decidable `≤`.

For backends that *do* provide decidable `≤` (e.g. `ℝ`, `Float`, `ExecFloat.Binary 8 23`), we also
expose a
checker that uses `≤` directly. We implement it by structural recursion (rather than
`decide (Box.contains ...)`) so it remains executable.
-/

/-- Boolean `a ≤ b` using an explicit `DecidableRel (· ≤ ·)` argument. -/
@[inline] def leDecBool [DecidableRel ((· ≤ ·) : α → α → Prop)] (a b : α) : Bool :=
  decide (a ≤ b)

/-- Boolean containment check using decidable `≤` and a finite fold over indices. -/
def containsDecBool [DecidableRel ((· ≤ ·) : α → α → Prop)] :
    ∀ {s : Shape}, Box α s → Tensor α s → Bool
| .scalar, b, x =>
    leDecBool b.lo.item x.item && leDecBool x.item b.hi.item
| .dim n inner, b, x =>
    (List.finRange n).all (fun i =>
      containsDecBool (s := inner) ⟨b.lo.unstack i, b.hi.unstack i⟩ (x.unstack i))

omit [Storage α] in
private theorem le_decBool_sound [DecidableRel ((· ≤ ·) : α → α → Prop)] (a b : α) :
    leDecBool a b = true → a ≤ b := by
  intro h
  exact of_decide_eq_true (by simpa [leDecBool] using h)

/-- Soundness of `containsDecBool`: if the Boolean checker returns `true`, then `Box.contains`
  holds. -/
theorem containsDecBool_sound [DecidableRel ((· ≤ ·) : α → α → Prop)] :
    ∀ {s : Shape} (b : Box α s) (x : Tensor α s),
      containsDecBool (s := s) b x = true → contains (α := α) b x
  | .scalar, b, x, hv => by
      have hEq : (leDecBool b.lo.item x.item && leDecBool x.item b.hi.item) = true := by
        simpa [containsDecBool] using hv
      have h' : leDecBool b.lo.item x.item = true ∧ leDecBool x.item b.hi.item = true :=
        Eq.mp (Bool.and_eq_true _ _) hEq
      constructor
      · exact le_decBool_sound b.lo.item x.item h'.1
      · exact le_decBool_sound x.item b.hi.item h'.2
  | .dim n inner, b, x, hv => by
      intro i
      have hi' :
          containsDecBool (s := inner)
            ⟨b.lo.unstack i, b.hi.unstack i⟩ (x.unstack i) = true := by
        have := (List.all_eq_true.mp (by simpa [containsDecBool] using hv)) i (List.mem_finRange i)
        simpa [containsDecBool] using this
      exact containsDecBool_sound (s := inner)
        ⟨b.lo.unstack i, b.hi.unstack i⟩ (x.unstack i) hi'

-- Dirac box around a given tensor
/-- Degenerate (Dirac) box with `lo = hi = t`. -/
def point (t : Tensor α s) : Box α s := { lo := t, hi := t }

/-- A tensor is contained in its own degenerate box. -/
theorem contains_point_self [Std.Refl ((· ≤ ·) : α → α → Prop)] :
    ∀ {s : Shape} (t : Tensor α s), contains (point t) t
  | .scalar, t => by
      exact ⟨Std.Refl.refl t.item, Std.Refl.refl t.item⟩
  | .dim _ inner, t => by
      intro i
      exact contains_point_self (s := inner) (t.unstack i)

end Box

namespace FlatBox

/-- Cast the lower endpoint to a checked vector dimension. -/
def loAsDim (B : FlatBox α) {m : Nat} (h : B.dim = m) : Tensor α [m] :=
  Tensor.castShape B.lo (congrArg (fun extent => Shape.dim extent .scalar) h)

/-- Cast the upper endpoint to a checked vector dimension. -/
def hiAsDim (B : FlatBox α) {m : Nat} (h : B.dim = m) : Tensor α [m] :=
  Tensor.castShape B.hi (congrArg (fun extent => Shape.dim extent .scalar) h)

/--
View a flattened interval box as a shape-indexed vector box after checking the dimension.

Most CROWN affine evaluators expect a `Box α (.dim m .scalar)`, while graph propagation stores
runtime-shaped `FlatBox` values. This helper keeps the dependent casts in one place.
-/
def getScalarBox (B : FlatBox α) {m : Nat} (h : B.dim = m) : Box α (.dim m .scalar) :=
  { lo := B.loAsDim h
    hi := B.hiAsDim h }

end FlatBox

/-!
CROWN affine bounds use one matrix row per output scalar and one column per input scalar.
A tensor of shape `s` contributes `s.size` coordinates to this flat representation. Graph nodes
retain their tensor shapes, and each transfer checks its operator's shape contract.

The graph `matmul` contract requires rank at least two and exactly the same leading shape on both
operands: `(...×m×n) · (...×n×p) → (...×m×p)`. The leading shape can contain any number of axes;
implicit broadcasting and vector operands are not supported.
-/

/--
Affine form `y = A*x + c` over flat vectors.

This is the representation used by CROWN/DeepPoly-style affine bound propagation.
-/
structure AffineVec (α : Type) [TorchLean.Storage α] (inDim outDim : Nat) where
  /-- Coefficient matrix; each row is an output affine coefficient vector. -/
  A : Tensor α [outDim, inDim]
  /-- Constant offset vector. -/
  c : Tensor α [outDim]

namespace AffineVec

variable {inDim outDim : Nat}
variable [BoundOps α]
open BoundOps

-- Evaluate affine upper/lower bound over a box using interval arithmetic splitting
/--
Evaluate an affine form on an input box, producing an output box.

This performs the standard interval evaluation for linear forms by taking, per coefficient `a`,
the appropriate endpoint (`lo` or `hi`) to minimize/maximize `a*x`.
-/
def evalOnBox (aff : AffineVec α inDim outDim) (B : Box α (.dim inDim .scalar)) : Box α (.dim
  outDim .scalar) :=
  let outLo := Tensor.dim (fun i =>
    let sum := (List.finRange inDim).foldl
      (fun acc j =>
        let aij := get2 aff.A i j
        let lo := B.lo.getScalar j
        let hi := B.hi.getScalar j
        BoundOps.addDown acc (min2 (BoundOps.mulDown aij lo) (BoundOps.mulDown aij hi)))
      0
    Tensor.scalar (BoundOps.addDown sum (aff.c.getScalar i)))
  let outHi := Tensor.dim (fun i =>
    let sum := (List.finRange inDim).foldl
      (fun acc j =>
        let aij := get2 aff.A i j
        let lo := B.lo.getScalar j
        let hi := B.hi.getScalar j
        BoundOps.addUp acc (max2 (BoundOps.mulUp aij lo) (BoundOps.mulUp aij hi)))
      0
    Tensor.scalar (BoundOps.addUp sum (aff.c.getScalar i)))
  { lo := outLo, hi := outHi }

/--
Evaluate an affine form on a flattened graph box after checking the input dimension.

Graph-level CROWN stores boxes as `FlatBox`; affine evaluation works over vector-shaped boxes. This
helper keeps that cast at the CROWN boundary instead of repeating it in verifier workflows.
-/
def evalOnFlatBox (aff : AffineVec α inDim outDim) (B : FlatBox α) (h : B.dim = inDim) :
    Box α (.dim outDim .scalar) :=
  aff.evalOnBox (B.getScalarBox h)

-- Compose two affine bounds: aff2 ∘ aff1 where aff1 maps R^{n}→R^{h}, aff2 maps R^{h}→R^{m}
/-- Compose two affine forms: `(aff2 ∘ aff1)(x) = aff2(aff1(x))`. -/
def compose {n h m : Nat}
  (aff2 : AffineVec α h m) (aff1 : AffineVec α n h) : AffineVec α n m :=
  let newA := Spec.matMulSpec aff2.A aff1.A
  let newc := Tensor.addSpec (Spec.matVecMulSpec aff2.A aff1.c) aff2.c
  { A := newA, c := newc }

-- Affine for linear layer: y = W x + b
/-- Build an affine form from a linear layer `y = W*x + b`. -/
def ofLinear (W : Tensor α [outDim, inDim]) (b : Tensor α [outDim])
  :
  AffineVec α inDim outDim :=
  { A := W, c := b }

end AffineVec

/- Interval arithmetic (IBP) for vectors -/
namespace IBP

variable [BoundOps α]
open BoundOps

/-- Positive part of a weight matrix, `W⁺ = max(W, 0)`, used by sign-split bound rules. -/
def matPos {m n : Nat}
    (W : Tensor α [m, n]) : Tensor α [m, n] :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      let w := get2 W i j
      Tensor.scalar (if w > 0 then w else 0)))

/-- Negative part of a weight matrix, `W⁻ = min(W, 0)`, used by sign-split bound rules. -/
def matNeg {m n : Nat}
    (W : Tensor α [m, n]) : Tensor α [m, n] :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      let w := get2 W i j
      Tensor.scalar (if w > 0 then 0 else w)))

/--
Interval bound propagation for a linear layer.

Given interval inputs `xB` and `bB`, this returns an interval box for `W*x + b` by:
- computing per-coefficient endpoint products with directed rounding (`BoundOps.mulDown`/`mulUp`),
  and
- summing with directed rounding (`addDown`/`addUp`).
-/
def linear {m n : Nat}
  (W : Tensor α [m, n])
  (xB : Box α (.dim n .scalar))
  (bB : Box α (.dim m .scalar)) : Box α (.dim m .scalar) :=
  let loOut := Tensor.dim (fun i =>
    let sum := (List.finRange n).foldl
      (fun acc j =>
        let aij := get2 W i j
        let xlo := xB.lo.getScalar j
        let xhi := xB.hi.getScalar j
        BoundOps.addDown acc
          (min2 (BoundOps.mulDown aij xlo) (BoundOps.mulDown aij xhi)))
      0
    Tensor.scalar (BoundOps.addDown sum (bB.lo.getScalar i)))
  let hiOut := Tensor.dim (fun i =>
    let sum := (List.finRange n).foldl
      (fun acc j =>
        let aij := get2 W i j
        let xlo := xB.lo.getScalar j
        let xhi := xB.hi.getScalar j
        BoundOps.addUp acc
          (max2 (BoundOps.mulUp aij xlo) (BoundOps.mulUp aij xhi)))
      0
    Tensor.scalar (BoundOps.addUp sum (bB.hi.getScalar i)))
  { lo := loOut, hi := hiOut }

end IBP


end NN.MLTheory.CROWN
