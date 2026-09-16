/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine -- shake: keep

/-!
# CROWN Graph Theorems

Shape, dimension, and enclosure lemmas for the graph CROWN engine. Keeping these proof layer facts
separate from the executable propagation passes makes the implementation files easier to browse.
-/

public section


namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.IR
open Std

variable {α : Type} [TorchLean.Storage α] [Context α]
variable [BoundOps α]

open BoundOps

namespace Theorems

/-- Dimension lemma: linear IBP returns an output box with the expected dimension. -/
theorem ibp_linear_output_dim
  (p : LinParams α) (Xin : FlatBox α)
  (h : Xin.dim = p.n)
  (ps : ParamStore α) (id : Nat)
  (hstore : ps.linearWB[id]? = some p)
  : ((ibpLinear (α:=α) id ps Xin).map (·.dim) = some p.m) := by
  simp [ibpLinear, ibpLinearParams, h, hstore, toFlatBox]

/-- Simple shape-preservation facts for FlatBox combinators used by IBP. -/
theorem box_add_dim (B1 B2 : FlatBox α) : (boxAdd (α:=α) B1 B2).dim = B1.dim := by
  cases B1 with
  | mk n1 lo1 hi1 =>
    cases B2 with
    | mk n2 lo2 hi2 =>
      by_cases h : n1 = n2
      · cases h; simp [boxAdd]
      · simp [boxAdd, h]

/-- `boxSub` preserves the left operand’s `dim` (even when the right operand has a mismatched dim).
  -/
theorem box_sub_dim (B1 B2 : FlatBox α) : (boxSub (α:=α) B1 B2).dim = B1.dim := by
  cases B1 with
  | mk n1 lo1 hi1 =>
    cases B2 with
    | mk n2 lo2 hi2 =>
      by_cases h : n1 = n2
      · cases h; simp [boxSub]
      · simp [boxSub, h]

omit [BoundOps α] in
/-- `boxRelu` preserves `dim`. -/
theorem box_relu_dim (B : FlatBox α) : (boxRelu (α:=α) B).dim = B.dim := by
  simp [boxRelu]

omit [BoundOps α] in
/-- `boxSquare` preserves `dim`. -/
theorem box_square_dim (B : FlatBox α) : (boxSquare (α:=α) B).dim = B.dim := by
  cases B; simp [boxSquare]

/-! Canonical forms for boxAdd/boxSub when dimensions match -/

theorem box_add_on_eq (n : Nat)
  (lo1 hi1 lo2 hi2 : Tensor α [n]) :
  boxAdd (α:=α) { dim := n, lo := lo1, hi := hi1 } { dim := n, lo := lo2, hi := hi2 }
    =
      { dim := n
        lo := Tensor.map2Spec BoundOps.addDown lo1 lo2
        hi := Tensor.map2Spec BoundOps.addUp hi1 hi2 } := by
  simp [boxAdd]

/-- Canonical form for `boxSub` when both boxes have the same dimension. -/
theorem box_sub_on_eq (n : Nat)
  (lo1 hi1 lo2 hi2 : Tensor α [n]) :
  boxSub (α:=α) { dim := n, lo := lo1, hi := hi1 } { dim := n, lo := lo2, hi := hi2 }
    =
      { dim := n
        lo := Tensor.map2Spec BoundOps.subDown lo1 hi2
        hi := Tensor.map2Spec BoundOps.subUp hi1 lo2 } := by
  simp [boxSub]

/-! Declarative enclosure predicates used by downstream graph-soundness statements. -/

namespace Semantics

/-- `encloses B x` means vector `x` lies componentwise between `B.lo` and `B.hi`. -/
@[expose] public def encloses (B : FlatBox α) (x : Tensor α [B.dim]) : Prop :=
  ∀ i : Fin B.dim,
    Tensor.getScalar B.lo i ≤ Tensor.getScalar x i ∧
      Tensor.getScalar x i ≤ Tensor.getScalar B.hi i

/- Enclosure for `boxAdd`: if x ∈ B1 and y ∈ B2, then x + y ∈ boxAdd B1 B2. -/

omit [BoundOps α] in
/-- If `x` is enclosed in `[lo1,hi1]` and `y` is enclosed in `[lo2,hi2]`, then `x+y` is enclosed in
`[lo1+lo2, hi1+hi2]`.

The scalar order fact is passed as `add_mono`: from `a ≤ b` and `c ≤ d`, derive
`a + c ≤ b + d`.
-/
theorem box_add_sound (n : Nat)
  (lo1 hi1 lo2 hi2 : Tensor α [n])
  (add_mono : ∀ {a b c d : α}, a ≤ b → c ≤ d → a + c ≤ b + d)
  (x y : Tensor α [n])
  (hx : encloses (α:=α) { dim := n, lo := lo1, hi := hi1 } x)
  (hy : encloses (α:=α) { dim := n, lo := lo2, hi := hi2 } y)
  : encloses (α:=α)
      { dim := n, lo := Tensor.addSpec lo1 lo2, hi := Tensor.addSpec hi1 hi2 }
      (Tensor.addSpec (α:=α) x y) := by
  intro i
  have hx_i := hx i
  have hy_i := hy i
  simpa [encloses, Tensor.addSpec] using
    And.intro (add_mono hx_i.1 hy_i.1) (add_mono hx_i.2 hy_i.2)

omit [BoundOps α] in
/-- If `x` is enclosed in `[lo1,hi1]` and `y` is enclosed in `[lo2,hi2]`, then `x-y` is enclosed in
`[lo1-hi2, hi1-lo2]`.

The scalar order fact is passed as `sub_mono`: from `a ≤ b` and `d ≤ c`, derive
`a - c ≤ b - d`.
-/
theorem box_sub_sound (n : Nat)
  (lo1 hi1 lo2 hi2 : Tensor α [n])
  (sub_mono : ∀ {a b c d : α}, a ≤ b → d ≤ c → a - c ≤ b - d)
  (x y : Tensor α [n])
  (hx : encloses (α:=α) { dim := n, lo := lo1, hi := hi1 } x)
  (hy : encloses (α:=α) { dim := n, lo := lo2, hi := hi2 } y)
  : encloses (α:=α)
      { dim := n, lo := Tensor.subSpec lo1 hi2, hi := Tensor.subSpec hi1 lo2 }
      (Tensor.subSpec (α:=α) x y) := by
  intro i
  have hx_i := hx i
  have hy_i := hy i
  simpa [encloses, Tensor.subSpec] using
    And.intro (sub_mono hx_i.1 hy_i.2) (sub_mono hx_i.2 hy_i.1)

omit [BoundOps α] in
/-- Enclosure for `boxRelu`: if $x\in B$, then $\operatorname{ReLU}(x)$ belongs to the resulting
box. -/
theorem box_relu_sound (n : Nat)
  (lo hi : Tensor α [n])
  (relu_mono : ∀ {a b : α}, a ≤ b →
    Activation.Math.reluSpec (α:=α) a ≤ Activation.Math.reluSpec (α:=α) b)
  (x : Tensor α [n])
  (hx : encloses (α:=α) { dim := n, lo := lo, hi := hi } x)
  : encloses (α:=α) (boxRelu (α:=α) { dim := n, lo := lo, hi := hi })
      (castDimScalar (α:=α) rfl (Activation.reluSpec (α:=α) x)) := by
  rw [castDimScalar_self]
  change ∀ i : Fin n,
    Tensor.getScalar (Tensor.mapSpec (fun value =>
      Activation.Math.reluSpec (α := α) value) lo) i ≤
        Tensor.getScalar (Activation.reluSpec (α := α) x) i ∧
      Tensor.getScalar (Activation.reluSpec (α := α) x) i ≤
        Tensor.getScalar (Tensor.mapSpec (fun value =>
          Activation.Math.reluSpec (α := α) value) hi) i
  intro i
  have hx_i := hx i
  simpa [Activation.reluSpec] using
    And.intro (relu_mono hx_i.1) (relu_mono hx_i.2)

/- Enclosure for `box_square`: if x ∈ B then x ⊙ x ∈ box_square B. -/

/-- Lower bound for `v * v` on `[l, u]`.

Squaring is not monotone, so the sign matters here: an interval straddling zero attains `0`, and
otherwise the minimum sits at the endpoint nearer the origin. -/
def sqLower (l u : α) : α :=
  let l2 := l * l
  let u2 := u * u
  if l < 0 then
    if 0 < u then 0 else (if l2 < u2 then l2 else u2)
  else (if l2 < u2 then l2 else u2)

/-- Upper bound for `v * v` on `[l, u]`: the larger of the two squared endpoints.

Unlike `sqLower` there is no case split on the sign, because squaring is maximized at whichever
endpoint is farther from the origin whether or not the interval straddles zero. -/
def sqUpper (l u : α) : α :=
  let l2 := l * l
  let u2 := u * u
  if l2 > u2 then l2 else u2

omit [BoundOps α] in
/-- Coordinatewise squaring of a box encloses the elementwise product of an enclosed tensor.

The scalar bound is taken as a hypothesis rather than proved here, since it is the one step that
depends on the ordered-field structure of `α`; every instance discharges it separately. -/
theorem box_square_sound (B : FlatBox α)
  (sq_bound : ∀ {l u v : α}, l ≤ v → v ≤ u → sqLower (α:=α) l u ≤ v * v ∧ v * v ≤ sqUpper (α:=α) l
    u)
  (x : Tensor α [B.dim])
  (hx : encloses (α:=α) B x)
  : encloses (α:=α) (boxSquare (α:=α) B)
      (castDimScalar (α:=α) rfl (Tensor.mulSpec (α:=α) x x)) := by
  cases B with
  | mk n lo hi =>
      rw [castDimScalar_self]
      change ∀ i : Fin n,
        Tensor.getScalar (Tensor.ofFn (fun i => sqLower (lo.getScalar i)
          (hi.getScalar i))) i ≤
            Tensor.getScalar (Tensor.mulSpec x x) i ∧
          Tensor.getScalar (Tensor.mulSpec x x) i ≤
            Tensor.getScalar (Tensor.ofFn (fun i => sqUpper (lo.getScalar i)
              (hi.getScalar i))) i
      intro i
      have hx_i := hx i
      have hbounds := sq_bound hx_i.1 hx_i.2
      simpa [Tensor.mulSpec] using hbounds

end Semantics

end Theorems


end NN.MLTheory.CROWN.Graph
