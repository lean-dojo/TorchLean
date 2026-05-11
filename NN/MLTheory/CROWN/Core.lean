/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.BoundOps
public import NN.Runtime.Context
public import NN.Spec.Core.Tensor.Linalg
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

open _root_.Spec
open _root_.Spec.Tensor

variable {α : Type} [Context α]

/--
Interval box `lo <= x <= hi` over a tensor shape.

This is the fundamental object for interval bound propagation (IBP).
-/
structure Box (α : Type) (s : Shape) where
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
  Tensor.scaleSpec (Tensor.addSpec b.lo b.hi) (Numbers.pointfive)

/-- Pointwise box radius `(hi - lo)/2`. -/
def radius (b : Box α s) : Tensor α s :=
  Tensor.scaleSpec (Tensor.subSpec b.hi b.lo) (Numbers.pointfive)

/--
Boolean `a <= b` using only the backend's decidable `>` from `Context`.

This is useful for executable checks in backends that do not provide a decidable `<=`.
-/
def leBool (a b : α) : Bool :=
  if decide (a > b) then false else true

/--
Executable containment check: returns `true` iff every component is within bounds.

This uses `le_bool` and is therefore available for any `Context α`.
-/
def containsBool : ∀ {s : Shape}, Box α s → Tensor α s → Bool
| .scalar, ⟨Tensor.scalar l, Tensor.scalar h⟩, Tensor.scalar v =>
  leBool l v && leBool v h
| .dim n _, ⟨Tensor.dim lo, Tensor.dim hi⟩, Tensor.dim x =>
  (List.finRange n).foldl (fun acc i => acc && containsBool ⟨lo i, hi i⟩ (x i)) true

/-- Logical containment: every component of `x` lies between `lo` and `hi`. -/
def contains : ∀ {s : Shape}, Box α s → Tensor α s → Prop
| .scalar, ⟨Tensor.scalar l, Tensor.scalar h⟩, Tensor.scalar v => l ≤ v ∧ v ≤ h
| .dim n _, ⟨Tensor.dim lo, Tensor.dim hi⟩, Tensor.dim x =>
  ∀ i : Fin n, contains ⟨lo i, hi i⟩ (x i)

/-!
`containsBool` above is a minimal `Context`-only checker that avoids requiring decidable `≤`.

For backends that *do* provide decidable `≤` (e.g. `ℝ`, `Float`, `IEEE32Exec`), we also expose a
checker that uses `≤` directly. We implement it by structural recursion (rather than
`decide (Box.contains ...)`) so it remains executable.
-/

/-- Boolean `a ≤ b` using an explicit `DecidableRel (· ≤ ·)` argument. -/
@[inline] def leDecBool [DecidableRel ((· ≤ ·) : α → α → Prop)] (a b : α) : Bool :=
  decide (a ≤ b)

/-- Boolean containment check using decidable `≤` and a finite fold over indices. -/
def containsDecBool [DecidableRel ((· ≤ ·) : α → α → Prop)] :
    ∀ {s : Shape}, Box α s → Tensor α s → Bool
| .scalar, ⟨Tensor.scalar l, Tensor.scalar h⟩, Tensor.scalar v =>
    leDecBool l v && leDecBool v h
| .dim n s', ⟨Tensor.dim lo, Tensor.dim hi⟩, Tensor.dim x =>
    (List.finRange n).all (fun i => containsDecBool (s := s') ⟨lo i, hi i⟩ (x i))

private theorem le_decBool_sound [DecidableRel ((· ≤ ·) : α → α → Prop)] (a b : α) :
    leDecBool a b = true → a ≤ b := by
  intro h
  exact of_decide_eq_true (by simpa [leDecBool] using h)

/-- Soundness of `containsDecBool`: if the Boolean checker returns `true`, then `Box.contains`
  holds. -/
theorem containsDecBool_sound [DecidableRel ((· ≤ ·) : α → α → Prop)] :
    ∀ {s : Shape} (b : Box α s) (x : Tensor α s),
      containsDecBool (s := s) b x = true → contains (α := α) b x
  | .scalar, ⟨Tensor.scalar l, Tensor.scalar h⟩, Tensor.scalar v, hv => by
      have hEq : (leDecBool l v && leDecBool v h) = true := by
        simpa [containsDecBool] using hv
      have h' : leDecBool l v = true ∧ leDecBool v h = true :=
        Eq.mp (Bool.and_eq_true (leDecBool l v) (leDecBool v h)) hEq
      constructor
      · exact le_decBool_sound l v h'.1
      · exact le_decBool_sound v h h'.2
  | .dim n s', ⟨Tensor.dim lo, Tensor.dim hi⟩, Tensor.dim x, hv => by
      intro i
      have hi' :
          containsDecBool (s := s') ⟨lo i, hi i⟩ (x i) = true := by
        have := (List.all_eq_true.mp (by simpa [containsDecBool] using hv)) i (List.mem_finRange i)
        simpa [containsDecBool] using this
      exact containsDecBool_sound (s := s') ⟨lo i, hi i⟩ (x i) hi'

-- Dirac box around a given tensor
/-- Degenerate (Dirac) box with `lo = hi = t`. -/
def point (t : Tensor α s) : Box α s := { lo := t, hi := t }

end Box

/-! We primarily target vector inputs/outputs for CROWN in this initial
integration. To avoid over-generalizing shapes, we use a specialized
variant for 1D (flat) vectors. -/

/--
Affine form `y = A*x + c` over flat vectors.

This is the representation used by CROWN/DeepPoly-style affine bound propagation.
-/
structure AffineVec (α : Type) (inDim outDim : Nat) where
  /-- Coefficient matrix; each row is an output affine coefficient vector. -/
  A : Tensor α (.dim outDim (.dim inDim .scalar))
  /-- Constant offset vector. -/
  c : Tensor α (.dim outDim .scalar)

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
  match aff.A, aff.c, B.lo, B.hi with
  | .dim rows, .dim cvec, .dim loVec, .dim hiVec =>
    let outLo :=
      Tensor.dim (fun i =>
        -- sum over j of min(a_ij * x_j) + c_i
        let row := rows i
        match row with
        | .dim cols =>
          let s :=
            (List.finRange inDim).foldl
              (fun (acc : Tensor α .scalar) (j : Fin inDim) =>
                match acc, cols j, loVec j, hiVec j with
                | .scalar accv, .scalar aij, .scalar lo, .scalar hi =>
                  let p1 := BoundOps.mulDown aij lo
                  let p2 := BoundOps.mulDown aij hi
                  let mn := min2 p1 p2
                  Tensor.scalar (BoundOps.addDown accv mn)) (Tensor.scalar 0)
          match cvec i, s with
          | .scalar ci, .scalar sv => Tensor.scalar (BoundOps.addDown sv ci))
    let outHi :=
      Tensor.dim (fun i =>
        let row := rows i
        match row with
        | .dim cols =>
          let s :=
            (List.finRange inDim).foldl
              (fun (acc : Tensor α .scalar) (j : Fin inDim) =>
                match acc, cols j, loVec j, hiVec j with
                | .scalar accv, .scalar aij, .scalar lo, .scalar hi =>
                  let p1 := BoundOps.mulUp aij lo
                  let p2 := BoundOps.mulUp aij hi
                  let mx := max2 p1 p2
                  Tensor.scalar (BoundOps.addUp accv mx)) (Tensor.scalar 0)
          match cvec i, s with
          | .scalar ci, .scalar sv => Tensor.scalar (BoundOps.addUp sv ci))
    { lo := outLo, hi := outHi }

-- Compose two affine bounds: aff2 ∘ aff1 where aff1 maps R^{n}→R^{h}, aff2 maps R^{h}→R^{m}
/-- Compose two affine forms: `(aff2 ∘ aff1)(x) = aff2(aff1(x))`. -/
def compose {n h m : Nat}
  (aff2 : AffineVec α h m) (aff1 : AffineVec α n h) : AffineVec α n m :=
  let newA := Spec.matMulSpec aff2.A aff1.A
  let newc := Tensor.addSpec (Spec.matVecMulSpec aff2.A aff1.c) aff2.c
  { A := newA, c := newc }

-- Affine for linear layer: y = W x + b
/-- Build an affine form from a linear layer `y = W*x + b`. -/
def ofLinear (W : Tensor α (.dim outDim (.dim inDim .scalar))) (b : Tensor α (.dim outDim .scalar))
  :
  AffineVec α inDim outDim :=
  { A := W, c := b }

end AffineVec

/- Interval arithmetic (IBP) for vectors -/
namespace IBP

variable [BoundOps α]
open BoundOps

/--
Interval bound propagation for a linear layer.

Given interval inputs `xB` and `bB`, this returns an interval box for `W*x + b` by:
- computing per-coefficient endpoint products with directed rounding (`BoundOps.mulDown`/`mulUp`),
  and
- summing with directed rounding (`addDown`/`addUp`).
-/
def linear {m n : Nat}
  (W : Tensor α (.dim m (.dim n .scalar)))
  (xB : Box α (.dim n .scalar))
  (bB : Box α (.dim m .scalar)) : Box α (.dim m .scalar) :=
  match W, xB.lo, xB.hi, bB.lo, bB.hi with
  | .dim rows, .dim lo, .dim hi, .dim blo, .dim bhi =>
    let loOut := Tensor.dim (fun i =>
      match rows i, blo i with
      | .dim cols, .scalar bi =>
        let s :=
          (List.finRange n).foldl
            (fun (acc : Tensor α .scalar) (j : Fin n) =>
              match acc, cols j, lo j, hi j with
              | .scalar accv, .scalar aij, .scalar xlo, .scalar xhi =>
                let p1 := BoundOps.mulDown aij xlo
                let p2 := BoundOps.mulDown aij xhi
                let mn := min2 p1 p2
                Tensor.scalar (BoundOps.addDown accv mn)) (Tensor.scalar 0)
        match s with
        | .scalar sv => Tensor.scalar (BoundOps.addDown sv bi))
    let hiOut := Tensor.dim (fun i =>
      match rows i, bhi i with
      | .dim cols, .scalar bi =>
        let s :=
          (List.finRange n).foldl
            (fun (acc : Tensor α .scalar) (j : Fin n) =>
              match acc, cols j, lo j, hi j with
              | .scalar accv, .scalar aij, .scalar xlo, .scalar xhi =>
                let p1 := BoundOps.mulUp aij xlo
                let p2 := BoundOps.mulUp aij xhi
                let mx := max2 p1 p2
                Tensor.scalar (BoundOps.addUp accv mx)) (Tensor.scalar 0)
        match s with
        | .scalar sv => Tensor.scalar (BoundOps.addUp sv bi))
    { lo := loOut, hi := hiOut }

end IBP


end NN.MLTheory.CROWN
