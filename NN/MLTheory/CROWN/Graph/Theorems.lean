/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Backward

/-!
# CROWN Graph Theorems

Shape, dimension, and enclosure lemmas for the graph CROWN engine. Keeping these proof-facing facts
separate from the executable propagation passes makes the implementation files easier to browse.
-/

public section


namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN
open NN.IR
open Std

variable {α : Type} [Context α]
variable [BoundOps α]

open BoundOps

/-! Helper lemmas about shapes/dimensions of basic constructions -/

omit [BoundOps α] in
/-- `toFlatBox` creates a box whose `dim` matches the given `n`. -/
lemma toFlatBox_dim (n : Nat) (B : Box α (.dim n .scalar)) : (toFlatBox (α:=α) n B).dim = n := rfl

-- Note: `ofFlatBox` returns a `Box` whose shape is definitionally `.dim B.dim .scalar`.
-- We omit a separate lemma about a `.matches` predicate since it is not present here.

namespace Theorems

/-- Dimension lemma: linear IBP returns an output box with the expected dimension. -/
lemma ibp_linear_output_dim
  (p : LinParams α) (Xin : FlatBox α)
  (h : Xin.dim = p.n)
  (ps : ParamStore α) (id : Nat)
  (hstore : ps.linearWB[id]? = some p)
  : ((ibp_linear (α:=α) id ps Xin).map (·.dim) = some p.m) := by
  -- From the definition of `ibp_linear`, the result is `some (toFlatBox p.m yB)`.
  -- Mapping `(·.dim)` over that gives `some p.m` by `toFlatBox_dim`.
  simp [ibp_linear, h, hstore, toFlatBox_dim]

/-- Simple shape-preservation facts for FlatBox combinators used by IBP. -/
lemma box_add_dim (B1 B2 : FlatBox α) : (box_add (α:=α) B1 B2).dim = B1.dim := by
  cases B1 with
  | mk n1 lo1 hi1 =>
    cases B2 with
    | mk n2 lo2 hi2 =>
      by_cases h : n1 = n2
      · cases h; simp [box_add]
      · simp [box_add, h]

/-- `box_sub` preserves the left operand’s `dim` (even when the right operand has a mismatched dim).
  -/
lemma box_sub_dim (B1 B2 : FlatBox α) : (box_sub (α:=α) B1 B2).dim = B1.dim := by
  cases B1 with
  | mk n1 lo1 hi1 =>
    cases B2 with
    | mk n2 lo2 hi2 =>
      by_cases h : n1 = n2
      · cases h; simp [box_sub]
      · simp [box_sub, h]

omit [BoundOps α] in
/-- `box_relu` preserves `dim`. -/
lemma box_relu_dim (B : FlatBox α) : (box_relu (α:=α) B).dim = B.dim := by
  simp [box_relu]

omit [BoundOps α] in
/-- `box_square` preserves `dim`. -/
lemma box_square_dim (B : FlatBox α) : (boxSquare (α:=α) B).dim = B.dim := by
  cases B; simp [boxSquare]

/-! Canonical forms for box_add/box_sub when dimensions match -/

lemma box_add_on_eq (n : Nat)
  (lo1 hi1 lo2 hi2 : Tensor α (.dim n .scalar)) :
  box_add (α:=α) { dim := n, lo := lo1, hi := hi1 } { dim := n, lo := lo2, hi := hi2 }
    =
      { dim := n
        lo := Tensor.map2Spec BoundOps.addDown lo1 lo2
        hi := Tensor.map2Spec BoundOps.addUp hi1 hi2 } := by
  simp [box_add]

/-- Canonical form for `box_sub` when both boxes have the same dimension. -/
lemma box_sub_on_eq (n : Nat)
  (lo1 hi1 lo2 hi2 : Tensor α (.dim n .scalar)) :
  box_sub (α:=α) { dim := n, lo := lo1, hi := hi1 } { dim := n, lo := lo2, hi := hi2 }
    =
      { dim := n
        lo := Tensor.map2Spec BoundOps.subDown lo1 hi2
        hi := Tensor.map2Spec BoundOps.subUp hi1 lo2 } := by
  simp [box_sub]

/-! Declarative enclosure predicates used by downstream graph-soundness statements. -/

namespace Semantics

variable [LE α]

/-- `encloses B x` means vector `x` lies componentwise between `B.lo` and `B.hi`. -/
@[expose] public def encloses (B : FlatBox α) (x : Tensor α (.dim B.dim .scalar)) : Prop :=
  let fx := getDimScalarFn (α:=α) x
  let flo := getDimScalarFn (α:=α) B.lo
  let fhi := getDimScalarFn (α:=α) B.hi
  ∀ i : Fin B.dim,
    match flo i, fhi i, fx i with
    | .scalar l, .scalar u, .scalar v => l ≤ v ∧ v ≤ u

/- Enclosure for `box_add`: if x ∈ B1 and y ∈ B2, then x + y ∈ box_add B1 B2. -/

omit [BoundOps α] in
/-- If `x` is enclosed in `[lo1,hi1]` and `y` is enclosed in `[lo2,hi2]`, then `x+y` is enclosed in
`[lo1+lo2, hi1+hi2]`.

The scalar order fact is passed as `add_mono`: from `a ≤ b` and `c ≤ d`, derive
`a + c ≤ b + d`.
-/
theorem box_add_sound (n : Nat)
  (lo1 hi1 lo2 hi2 : Tensor α (.dim n .scalar))
  (add_mono : ∀ {a b c d : α}, a ≤ b → c ≤ d → a + c ≤ b + d)
  (x y : Tensor α (.dim n .scalar))
  (hx : encloses (α:=α) { dim := n, lo := lo1, hi := hi1 } x)
  (hy : encloses (α:=α) { dim := n, lo := lo2, hi := hi2 } y)
  : encloses (α:=α)
      { dim := n, lo := Tensor.addSpec lo1 lo2, hi := Tensor.addSpec hi1 hi2 }
      (Tensor.addSpec (α:=α) x y) := by
  cases lo1 with
  | dim flo1 =>
    cases hi1 with
    | dim fhi1 =>
      cases lo2 with
      | dim flo2 =>
        cases hi2 with
        | dim fhi2 =>
          cases x with
          | dim fx =>
            cases y with
            | dim fy =>
              simp [encloses, getDimScalarFn, Tensor.addSpec, Tensor.map2Spec] at hx hy ⊢
              intro i
              have hx_i := hx i
              have hy_i := hy i
              cases hL1 : flo1 i with
              | scalar l1 =>
                cases hU1 : fhi1 i with
                | scalar u1 =>
                  cases hX : fx i with
                  | scalar xv =>
                    cases hL2 : flo2 i with
                    | scalar l2 =>
                      cases hU2 : fhi2 i with
                      | scalar u2 =>
                        cases hY : fy i with
                        | scalar yv =>
                          have hx' : l1 ≤ xv ∧ xv ≤ u1 := by
                            simpa [hL1, hU1, hX] using hx_i
                          have hy' : l2 ≤ yv ∧ yv ≤ u2 := by
                            simpa [hL2, hU2, hY] using hy_i
                          simpa [hL1, hU1, hX, hL2, hU2, hY] using
                            And.intro (add_mono hx'.1 hy'.1) (add_mono hx'.2 hy'.2)

omit [BoundOps α] in
/-- If `x` is enclosed in `[lo1,hi1]` and `y` is enclosed in `[lo2,hi2]`, then `x-y` is enclosed in
`[lo1-hi2, hi1-lo2]`.

The scalar order fact is passed as `sub_mono`: from `a ≤ b` and `d ≤ c`, derive
`a - c ≤ b - d`.
-/
theorem box_sub_sound (n : Nat)
  (lo1 hi1 lo2 hi2 : Tensor α (.dim n .scalar))
  (sub_mono : ∀ {a b c d : α}, a ≤ b → d ≤ c → a - c ≤ b - d)
  (x y : Tensor α (.dim n .scalar))
  (hx : encloses (α:=α) { dim := n, lo := lo1, hi := hi1 } x)
  (hy : encloses (α:=α) { dim := n, lo := lo2, hi := hi2 } y)
  : encloses (α:=α)
      { dim := n, lo := Tensor.subSpec lo1 hi2, hi := Tensor.subSpec hi1 lo2 }
      (Tensor.subSpec (α:=α) x y) := by
  cases lo1 with
  | dim flo1 =>
    cases hi1 with
    | dim fhi1 =>
      cases lo2 with
      | dim flo2 =>
        cases hi2 with
        | dim fhi2 =>
          cases x with
          | dim fx =>
            cases y with
            | dim fy =>
              simp [encloses, getDimScalarFn, Tensor.subSpec, Tensor.map2Spec] at hx hy ⊢
              intro i
              have hx_i := hx i
              have hy_i := hy i
              cases hL1 : flo1 i with
              | scalar l1 =>
                cases hU1 : fhi1 i with
                | scalar u1 =>
                  cases hX : fx i with
                  | scalar xv =>
                    cases hL2 : flo2 i with
                    | scalar l2 =>
                      cases hU2 : fhi2 i with
                      | scalar u2 =>
                        cases hY : fy i with
                        | scalar yv =>
                          have hx' : l1 ≤ xv ∧ xv ≤ u1 := by
                            simpa [hL1, hU1, hX] using hx_i
                          have hy' : l2 ≤ yv ∧ yv ≤ u2 := by
                            simpa [hL2, hU2, hY] using hy_i
                          have hlo : l1 - u2 ≤ xv - yv := sub_mono hx'.1 hy'.2
                          have hhi : xv - yv ≤ u1 - l2 := sub_mono hx'.2 hy'.1
                          simpa [hL1, hU1, hX, hL2, hU2, hY] using And.intro hlo hhi

omit [BoundOps α] in
/-- Enclosure for `box_relu`: if x ∈ B then ReLU(x) ∈ box_relu B. -/
theorem box_relu_sound (n : Nat)
  (lo hi : Tensor α (.dim n .scalar))
  (relu_mono : ∀ {a b : α}, a ≤ b →
    Activation.Math.reluSpec (α:=α) a ≤ Activation.Math.reluSpec (α:=α) b)
  (x : Tensor α (.dim n .scalar))
  (hx : encloses (α:=α) { dim := n, lo := lo, hi := hi } x)
  : encloses (α:=α) (box_relu (α:=α) { dim := n, lo := lo, hi := hi })
      (castDimScalar (α:=α)
        (by
          have hdim : (box_relu (α:=α) { dim := n, lo := lo, hi := hi }).dim = n := by
            simpa using (Theorems.box_relu_dim (α:=α) { dim := n, lo := lo, hi := hi })
          exact hdim.symm)
        (Activation.reluSpec (α:=α) x)) := by
  cases lo with
  | dim flo =>
    cases hi with
    | dim fhi =>
      cases x with
      | dim fx =>
        simp [castDimScalar, box_relu, Activation.reluSpec, Tensor.mapSpec,
          encloses, getDimScalarFn] at hx ⊢
        intro i
        have hx_i := hx i
        cases hL : flo i with
        | scalar l =>
          cases hU : fhi i with
          | scalar u =>
            cases hX : fx i with
            | scalar v =>
              have hx' : l ≤ v ∧ v ≤ u := by
                simpa [hL, hU, hX] using hx_i
              have hlo : Activation.Math.reluSpec (α:=α) l ≤ Activation.Math.reluSpec (α:=α) v :=
                relu_mono hx'.1
              have hhi : Activation.Math.reluSpec (α:=α) v ≤ Activation.Math.reluSpec (α:=α) u :=
                relu_mono hx'.2
              simpa [Activation.Math.reluSpec, hL, hU, hX] using And.intro hlo hhi

/- Enclosure for `box_square`: if x ∈ B then x ⊙ x ∈ box_square B. -/

def sqLower (l u : α) : α :=
  let l2 := l * l
  let u2 := u * u
  if l < Numbers.zero then
    if Numbers.zero < u then Numbers.zero else (if l2 < u2 then l2 else u2)
  else (if l2 < u2 then l2 else u2)

def sqUpper (l u : α) : α :=
  let l2 := l * l
  let u2 := u * u
  if l2 > u2 then l2 else u2

omit [BoundOps α] in
theorem box_square_sound (B : FlatBox α)
  (sq_bound : ∀ {l u v : α}, l ≤ v → v ≤ u → sqLower (α:=α) l u ≤ v * v ∧ v * v ≤ sqUpper (α:=α) l
    u)
  (x : Tensor α (.dim B.dim .scalar))
  (hx : encloses (α:=α) B x)
  : encloses (α:=α) (boxSquare (α:=α) B)
      (castDimScalar (α:=α)
        (by simpa using (box_square_dim (α:=α) B).symm)
        (Tensor.mulSpec (α:=α) x x)) := by
  cases B with
  | mk n lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        cases x with
        | dim fx =>
          simp [castDimScalar, encloses, getDimScalarFn, boxSquare, Tensor.mulSpec,
            Tensor.map2Spec] at hx ⊢
          intro i
          have hx_i := hx i
          cases hL : flo i with
          | scalar l =>
            cases hU : fhi i with
            | scalar u =>
              cases hX : fx i with
              | scalar v =>
                have hx' : l ≤ v ∧ v ≤ u := by
                  simpa [hL, hU, hX] using hx_i
                have hbounds : sqLower (α:=α) l u ≤ v * v ∧ v * v ≤ sqUpper (α:=α) l u :=
                  sq_bound hx'.1 hx'.2
                have hbounds' :
                    (if l < Numbers.zero then
                        if Numbers.zero < u then Numbers.zero
                        else if l * l < u * u then l * l else u * u
                      else if l * l < u * u then l * l else u * u) ≤
                        v * v ∧
                      v * v ≤ (if u * u < l * l then l * l else u * u) := by
                  simpa [sqLower, sqUpper] using hbounds
                -- Reduce the enclosure goal to the same pointwise bound.
                simpa [boxSquare, castDimScalar, encloses, getDimScalarFn, Tensor.mulSpec,
                  Tensor.map2Spec, hL, hU, hX] using
                  hbounds'

end Semantics

end Theorems


end NN.MLTheory.CROWN.Graph
