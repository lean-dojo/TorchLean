/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.BoundOps
public import TorchLean.Verify.CROWN.FlatBox
public import TorchLean.Tensor.Arithmetic
public import TorchLean.Tensor.Layers.Linear

/-!
# CROWN core: boxes and affine forms

The shape-indexed objects that IBP and CROWN work with:

- `Box`, elementwise lower and upper bounds on a tensor, with the Boolean containment test
  `Box.containsDecBool` (sound by `Box.containsDecBool_sound`).
- Affine forms `W x + b` as `TorchLean.Linear`, evaluated on boxes by `TorchLean.Linear.evalOnBox`.
- Sign splits of weight matrices, IBP for linear layers, and affine evaluation on boxes.

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


namespace TorchLean.CROWN

open Tensor

variable {α : Type} [Storage α] [Context α]

/--
Interval box `lo <= x <= hi` over a tensor shape, the basic object of interval bound propagation
(IBP).
-/
structure Box (α : Type) [Storage α] (s : Shape) where
  /-- Elementwise lower bound. -/
  lo : Tensor α s
  /-- Elementwise upper bound. -/
  hi : Tensor α s

namespace Box

variable {s : Shape}

/-- Logical containment: every component of `x` lies between `lo` and `hi`. -/
def Contains : ∀ {s : Shape}, Box α s → Tensor α s → Prop
| .scalar, b, x => b.lo.item ≤ x.item ∧ x.item ≤ b.hi.item
| .dim _ inner, b, x =>
  ∀ i, Contains (s := inner) ⟨b.lo.unstack i, b.hi.unstack i⟩ (x.unstack i)

/-- Decide `Contains` coordinatewise with the decidable `≤` of `α`, by recursion over the shape. -/
def containsDecBool [DecidableRel ((· ≤ ·) : α → α → Prop)] :
    ∀ {s : Shape}, Box α s → Tensor α s → Bool
| .scalar, b, x =>
    decide (b.lo.item ≤ x.item) && decide (x.item ≤ b.hi.item)
| .dim n inner, b, x =>
    (List.finRange n).all (fun i =>
      containsDecBool (s := inner) ⟨b.lo.unstack i, b.hi.unstack i⟩ (x.unstack i))

/-- Soundness of `containsDecBool`: if the Boolean checker returns `true`, then `Box.Contains`
  holds. -/
theorem containsDecBool_sound [DecidableRel ((· ≤ ·) : α → α → Prop)] :
    ∀ {s : Shape} (b : Box α s) (x : Tensor α s),
      containsDecBool (s := s) b x = true → Contains (α := α) b x
  | .scalar, b, x, hv => by
      simpa [containsDecBool, Contains] using hv
  | .dim n inner, b, x, hv => by
      intro i
      have hi' :
          containsDecBool (s := inner)
            ⟨b.lo.unstack i, b.hi.unstack i⟩ (x.unstack i) = true := by
        have := (List.all_eq_true.mp (by simpa [containsDecBool] using hv)) i (List.mem_finRange i)
        simpa [containsDecBool] using this
      exact containsDecBool_sound (s := inner)
        ⟨b.lo.unstack i, b.hi.unstack i⟩ (x.unstack i) hi'

/-- Degenerate (Dirac) box with `lo = hi = t`. -/
def point (t : Tensor α s) : Box α s := { lo := t, hi := t }

/-- A tensor is contained in its own degenerate box. -/
theorem contains_point_self [Std.Refl ((· ≤ ·) : α → α → Prop)] :
    ∀ {s : Shape} (t : Tensor α s), Contains (point t) t
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
View a flattened interval box as a shape-indexed vector box, given `B.dim = m`.

Most CROWN affine evaluators expect a `Box α (.dim m .scalar)`, while graph propagation stores
runtime-shaped `FlatBox` values. The dependent casts live here, in one place.
-/
def getScalarBox (B : FlatBox α) {m : Nat} (h : B.dim = m) : Box α (.dim m .scalar) :=
  { lo := B.loAsDim h
    hi := B.hiAsDim h }

end FlatBox

/-!
CROWN affine bounds use one matrix row per output scalar and one column per input scalar.
A tensor of shape `s` contributes `s.size` coordinates to this flat representation. Graph nodes
retain their tensor shapes, and each transfer checks its operator's shape contract.

Graph `matmul` uses the canonical IR contract, including vector promotion and broadcasting of
leading batch axes. The sign splits and affine evaluation below work on the flattened coefficient
matrix, so the graph operands themselves keep whatever shapes that contract allows.
-/

/- Interval arithmetic (IBP) for vectors -/
namespace IBP

variable [BoundOps α]
open BoundOps

/-- Positive part of a weight matrix, `W⁺ = max(W, 0)`, used by sign-split bound rules. -/
def matPos {m n : Nat}
    (W : Tensor α [m, n]) : Tensor α [m, n] :=
  Tensor.map (fun w => if w > 0 then w else 0) W

/-- Negative part of a weight matrix, `W⁻ = min(W, 0)`, used by sign-split bound rules. -/
def matNeg {m n : Nat}
    (W : Tensor α [m, n]) : Tensor α [m, n] :=
  Tensor.map (fun w => if w > 0 then 0 else w) W

/--
Interval bound propagation through a linear layer: a box for `W*x + b` from the input box `xB`
and the bias box `bB`.

For each weight, the lower bound takes the smaller of its two endpoint products rounded down
(`BoundOps.mulDown`) and the upper bound the larger rounded up (`mulUp`). The sums round with
`addDown` and `addUp`.
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
          (Context.minimum (BoundOps.mulDown aij xlo) (BoundOps.mulDown aij xhi)))
      0
    Tensor.scalar (BoundOps.addDown sum (bB.lo.getScalar i)))
  let hiOut := Tensor.dim (fun i =>
    let sum := (List.finRange n).foldl
      (fun acc j =>
        let aij := get2 W i j
        let xlo := xB.lo.getScalar j
        let xhi := xB.hi.getScalar j
        BoundOps.addUp acc
          (Context.maximum (BoundOps.mulUp aij xlo) (BoundOps.mulUp aij xhi)))
      0
    Tensor.scalar (BoundOps.addUp sum (bB.hi.getScalar i)))
  { lo := loOut, hi := hiOut }

end IBP

/--
Evaluate an affine form `W x + b` on an input box, producing an output box.

It runs `IBP.linear` with the bias `b` as a point box, so for each coefficient `a` the endpoint of
`x` that minimizes or maximizes `a*x` gives the lower or upper bound.
-/
def _root_.TorchLean.Linear.evalOnBox {inDim outDim : Nat} [BoundOps α]
    (aff : Linear α inDim outDim) (B : Box α (.dim inDim .scalar)) :
    Box α (.dim outDim .scalar) :=
  IBP.linear aff.weights B (Box.point aff.bias)

/--
Evaluate an affine form on a flattened graph box whose dimension is `inDim`.

Graph-level CROWN stores boxes as `FlatBox`, while affine evaluation works on vector-shaped boxes.
The cast happens here at the CROWN boundary, so verifier workflows do not repeat it.
-/
def _root_.TorchLean.Linear.evalOnFlatBox {inDim outDim : Nat} [BoundOps α]
    (aff : Linear α inDim outDim) (B : FlatBox α) (h : B.dim = inDim) :
    Box α (.dim outDim .scalar) :=
  aff.evalOnBox (B.getScalarBox h)

end TorchLean.CROWN
