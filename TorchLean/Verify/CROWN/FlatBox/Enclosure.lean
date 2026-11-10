/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.FlatBox.Transfer
public import TorchLean.Tensor.Context.Real

/-!
# CROWN Graph Theorems

Shape, dimension and enclosure lemmas for the graph CROWN engine, kept apart from the executable
passes. The `ConcatLayout` lemmas say that `concat` and `split` are inverse and that concatenation
preserves enclosure, and `Graph.boxAdd_dim` and its neighbours give the dimension of the IBP box
combinators.
-/

public section

namespace TorchLean.CROWN.Graph

open TorchLean.Tensor
open Std

variable {α : Type} [Storage α] [Context α]
variable [BoundOps α]

namespace ConcatLayout

omit [Context α] [BoundOps α] in
/-- Reading the concatenation at the flat position of `(parent, index)` gives entry `index` of that
parent. -/
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
/-- `split` undoes `concat` at every parent, including repeated parents and zero-length segments. -/
@[simp] theorem split_concat (layout : ConcatLayout)
    (values : (parent : Fin layout.lengths.length) →
      Tensor α [(layout.parentShape parent).size])
    (parent : Fin layout.lengths.length) :
    layout.split (layout.concat values) parent = values parent := by
  apply Tensor.ext_vector
  intro index
  simp only [split, Tensor.getScalar_ofFn, concat_get]

omit [Context α] [BoundOps α] in
/-- `concat` undoes `split`: rejoining the parent slices of a vector gives the vector back. -/
@[simp] theorem concat_split (layout : ConcatLayout)
    (value : Tensor α [layout.outputShape.size]) :
    layout.concat (layout.split value) = value := by
  apply Tensor.ext_vector
  intro index
  simp only [concat, split, Tensor.getScalar_ofFn]
  exact congrArg (Tensor.getScalar value) (layout.flatEquiv.apply_symm_apply index)

omit [BoundOps α] in
/-- If every parent's values lie between its lower and upper tensors, the concatenated values lie
between the concatenated bounds. Concatenation only moves entries, so no rounding enters. -/
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

/-- `boxAdd` keeps the dimension of its left operand, also on a dimension mismatch, where it returns
the left box. -/
theorem boxAdd_dim (B1 B2 : FlatBox α) : (boxAdd (α:=α) B1 B2).dim = B1.dim := by
  cases B1 with
  | mk n1 lo1 hi1 =>
    cases B2 with
    | mk n2 lo2 hi2 =>
      by_cases h : n1 = n2
      · cases h; simp [boxAdd]
      · simp [boxAdd, h]

/-- `boxSub` keeps the dimension of its left operand, also when the right operand's dimension
differs. -/
theorem boxSub_dim (B1 B2 : FlatBox α) : (boxSub (α:=α) B1 B2).dim = B1.dim := by
  cases B1 with
  | mk n1 lo1 hi1 =>
    cases B2 with
    | mk n2 lo2 hi2 =>
      by_cases h : n1 = n2
      · cases h; simp [boxSub]
      · simp [boxSub, h]

omit [BoundOps α] in
/-- `boxReLU` preserves `dim`. -/
theorem boxReLU_dim (B : FlatBox α) : (boxReLU (α:=α) B).dim = B.dim := by
  simp [boxReLU]

/-- `boxSquare` preserves `dim`. -/
theorem boxSquare_dim (B : FlatBox α) : (boxSquare (α:=α) B).dim = B.dim := by
  cases B; simp [boxSquare]

end TorchLean.CROWN.Graph
