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

namespace ConcatLayout

omit [Context α] [BoundOps α] in
/-- Concatenation preserves the scalar at each embedded parent coordinate. -/
@[simp] theorem concat_get (layout : ConcatLayout)
    (values : (parent : Fin layout.lengths.length) →
      Tensor α [(layout.parentShape parent).size])
    (parent : Fin layout.lengths.length) (index : Fin (layout.parentShape parent).size) :
    Tensor.getScalar (layout.concat values) (layout.flatEquiv ⟨parent, index⟩) =
      Tensor.getScalar (values parent) index := by
  simp only [concat, Tensor.getScalar_ofFn]
  exact congrArg (fun source => Tensor.getScalar (values source.1) source.2)
    (layout.flatEquiv.symm_apply_apply ⟨parent, index⟩)

omit [Context α] [BoundOps α] in
/-- Splitting recovers each occurrence, including repeated parents and zero-length segments. -/
@[simp] theorem split_concat (layout : ConcatLayout)
    (values : (parent : Fin layout.lengths.length) →
      Tensor α [(layout.parentShape parent).size])
    (parent : Fin layout.lengths.length) :
    layout.split (layout.concat values) parent = values parent := by
  apply Tensor.ext_vector
  intro index
  simp only [split, Tensor.getScalar_ofFn, concat_get]

omit [Context α] [BoundOps α] in
/-- Concatenating all coordinate slices recovers the original objective. -/
@[simp] theorem concat_split (layout : ConcatLayout)
    (value : Tensor α [layout.outputShape.size]) :
    layout.concat (layout.split value) = value := by
  apply Tensor.ext_vector
  intro index
  simp only [concat, split, Tensor.getScalar_ofFn]
  exact congrArg (Tensor.getScalar value) (layout.flatEquiv.apply_symm_apply index)

omit [BoundOps α] in
/-- Parent interval enclosure is preserved by concat without any scalar arithmetic. -/
theorem concat_encloses (layout : ConcatLayout)
    (lower upper values : (parent : Fin layout.lengths.length) →
      Tensor α [(layout.parentShape parent).size])
    (h : ∀ parent index,
      Tensor.getScalar (lower parent) index ≤ Tensor.getScalar (values parent) index ∧
      Tensor.getScalar (values parent) index ≤ Tensor.getScalar (upper parent) index) :
    ∀ index,
      Tensor.getScalar (layout.concat lower) index ≤
          Tensor.getScalar (layout.concat values) index ∧
        Tensor.getScalar (layout.concat values) index ≤
          Tensor.getScalar (layout.concat upper) index := by
  intro index
  simpa only [concat, Tensor.getScalar_ofFn] using
    h (layout.flatEquiv.symm index).1 (layout.flatEquiv.symm index).2

end ConcatLayout

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

end Semantics

end Theorems


end NN.MLTheory.CROWN.Graph
