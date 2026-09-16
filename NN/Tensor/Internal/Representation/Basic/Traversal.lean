/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Basic.Pointwise

/-!
# Tensor Buffer Traversal

Laws connecting native physical-buffer folds to finite row-major index folds.
These results keep executable reductions on specialized storage while exposing
the ordinary finite traversals needed by proofs.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u

private theorem list_foldl_flatten
    {α β : Type*} (lists : List (List β))
    (step : α → β → α) (initial : α) :
    lists.flatten.foldl step initial =
      lists.foldl (fun value entries => entries.foldl step value) initial := by
  induction lists generalizing initial with
  | nil => rfl
  | cons entries lists inductionHypothesis =>
      simp only [List.flatten_cons, List.foldl_append, List.foldl_cons]
      exact inductionHypothesis _

private theorem foldl_flatten_ofFn
    {α β : Type*} (outer inner : Nat)
    (entries : Fin outer → Fin inner → β)
    (step : α → β → α) (initial : α) :
    (List.ofFn fun outerIndex =>
        List.ofFn fun innerIndex =>
          entries outerIndex innerIndex).flatten.foldl step initial =
      Fin.foldl outer
        (fun value outerIndex =>
          Fin.foldl inner
            (fun value innerIndex =>
              step value (entries outerIndex innerIndex))
            value)
        initial := by
  rw [list_foldl_flatten]
  simp only [List.ofFn_eq_map, List.foldl_map,
    Fin.foldl_eq_foldl_finRange]

/-- Nested finite folds are one row-major fold over the product index. -/
theorem fin_foldl_product
    {α : Type u} (outer inner : Nat)
    (step : α → Fin outer → Fin inner → α) (initial : α) :
    Fin.foldl outer
        (fun value outerIndex =>
          Fin.foldl inner
            (fun value innerIndex =>
              step value outerIndex innerIndex)
            value)
        initial =
      Fin.foldl (outer * inner)
        (fun value index =>
          step value index.divNat index.modNat)
        initial := by
  symm
  rw [Fin.foldl_eq_foldl_finRange,
    ← List.ofFn_id (outer * inner), List.ofFn_mul,
    foldl_flatten_ofFn]
  apply congrArg fun outerStep =>
    Fin.foldl outer outerStep initial
  funext value outerIndex
  apply congrArg fun innerStep =>
    Fin.foldl inner innerStep value
  funext innerValue innerIndex
  have hIndex :
      (⟨outerIndex * inner + innerIndex, by
        calc
          ↑outerIndex * inner + innerIndex <
              (outerIndex + 1) * inner := by
            rw [Nat.add_mul, Nat.one_mul]
            exact Nat.add_lt_add_left innerIndex.isLt _
          _ ≤ outer * inner :=
            Nat.mul_le_mul_right inner outerIndex.isLt⟩ :
        Fin (outer * inner)) =
        Fin.mkDivMod outerIndex innerIndex := by
    apply Fin.ext
    simp [Nat.mul_comm, Nat.add_comm]
  rw [hIndex]
  simp only [id_eq, Fin.divNat_mkDivMod, Fin.modNat_mkDivMod]

namespace Rep

/-- A packed fold is the finite fold over the tensor's flat row-major indices. -/
theorem foldl_eq_fin_foldl {α β : Type} [Storage α] {s : Shape}
    (step : β → α → β) (initial : β) (x : Rep α s) :
    x.foldl step initial =
      Fin.foldl (Shape.size s)
        (fun value index => step value (x.getFlat index)) initial := by
  rw [foldl_eq_data_foldl, ← Array.foldl_toList]
  have hData : x.data = Array.ofFn x.getFlat := by
    apply Array.ext
    · simp
    · intro index hLeft hRight
      let flatIndex : Fin (Shape.size s) := ⟨index, by simpa using hLeft⟩
      simpa [flatIndex] using x.data_getFlat flatIndex
  rw [hData, Array.toList_ofFn, List.ofFn_eq_map, List.foldl_map,
    ← Fin.foldl_eq_foldl_finRange]

/-- Flat lookup into a stack selects the corresponding row and inner flat index. -/
theorem getFlat_stack_product {α : Type} [Storage α] {n : Nat} {s : Shape}
    (components : Fin n → Rep α s) (outer : Fin n)
    (inner : Fin (Shape.size s)) :
    (stack components).getFlat
        (Fin.cast (Shape.size_cons n s).symm (finProdFinEquiv (outer, inner))) =
      (components outer).getFlat inner := by
  let index : Fin (Shape.size (n :: s)) :=
    Fin.cast (Shape.size_cons n s).symm (finProdFinEquiv (outer, inner))
  have hCoordinate :
      Coord.unlinearize (s := n :: s) index =
        (outer, Coord.unlinearize (s := s) inner) := by
    apply Coord.linearize_injective
    rw [Coord.linearize_unlinearize, Coord.linearize_cons,
      Coord.linearize_unlinearize]
    apply Fin.ext
    simp [index]
  calc
    (stack components).getFlat index =
        (stack components) (Coord.unlinearize (s := n :: s) index) := by
      change _ = (stack components).getFlat
        (Coord.linearize (Coord.unlinearize index))
      rw [Coord.linearize_unlinearize]
    _ = (components outer) (Coord.unlinearize inner) := by
      rw [hCoordinate, stack_apply]
    _ = (components outer).getFlat inner := by
      change (components outer).getFlat (Coord.linearize (Coord.unlinearize inner)) = _
      rw [Coord.linearize_unlinearize]

/-- Folding a stack is folding each row in leading-axis order. -/
theorem foldl_stack {α β : Type} [Storage α] {n : Nat} {s : Shape}
    (step : β → α → β) (initial : β)
    (components : Fin n → Rep α s) :
    (stack components).foldl step initial =
      Fin.foldl n
        (fun value component => (components component).foldl step value)
        initial := by
  rw [foldl_eq_fin_foldl]
  symm
  simp_rw [foldl_eq_fin_foldl]
  rw [fin_foldl_product]
  apply congrArg (fun next => Fin.foldl (n * Shape.size s) next initial)
  funext value index
  apply congrArg (step value)
  have hProduct :
      finProdFinEquiv (index.divNat, index.modNat) = index := by
    exact (finProdFinEquiv :
      Fin n × Fin (Shape.size s) ≃ Fin (n * Shape.size s)).apply_symm_apply index
  have hIndex :
      Fin.cast (Shape.size_cons n s).symm
          (finProdFinEquiv (index.divNat, index.modNat)) =
        index := by
    apply Fin.ext
    simpa using congrArg Fin.val hProduct
  have hStack :=
    getFlat_stack_product components index.divNat index.modNat
  rw [hIndex] at hStack
  exact hStack.symm

end Rep

end TorchLean.Tensor.Internal
