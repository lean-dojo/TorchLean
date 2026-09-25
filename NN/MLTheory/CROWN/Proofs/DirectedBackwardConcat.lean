/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.DirectedBackwardSweep

/-!
# Concat occurrence accumulation in directed CROWN

The coordinate map splits the outgoing objective by parent occurrence. The fold then adds every
occurrence into the graph table, preserving repeated parents and empty slices.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph.Internal
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

private theorem zip_finRange {β : Type} (parents : Array Nat) (n : Nat)
    (hsize : parents.size = n) (f : Fin n → β) :
    parents.zip ((Array.finRange n).map f) =
      (Array.finRange n).map (fun i => (parents[i.val]!, f i)) := by
  apply Array.ext
  · simp [hsize]
  · intro i hi hj
    simp only [Array.size_zip, Array.size_map, Array.size_finRange, hsize, Nat.min_self] at hi
    simp [getElem!_pos (c := parents) (i := i) (by omega)]

/-- Folding a list of enclosed contributions adds exactly the sum of their real objectives. -/
theorem represents_fold {ι : Type} (indices : List ι) (parent : ι → Nat)
    (box : ι → FlatBox α) (a : ι → Nat → ℝ)
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : DirectedBackwardState α} {z : ℝ}
    (h : Represents dims v input k st z)
    (hparent : ∀ i, parent i < st.coeffs.size)
    (hlive : ∀ i, parent i ∈ pending input k)
    (hbox : ∀ i, RowEncloses (box i) (dims (parent i)) (a i)) :
    Represents dims v input k
      (indices.foldl (fun state i => addDirectedCoeff state (parent i) (box i)) st)
      (z + (indices.map (fun i => dot (dims (parent i)) (a i) (v (parent i)))).sum) := by
  induction indices generalizing st z with
  | nil => simpa using h
  | cons i indices ih =>
      have hadd := represents_add h (parent i) (hparent i) (hlive i) (box i) (a i) (hbox i)
      have hrest := ih hadd (fun j => by simpa using hparent j)
      simpa only [List.foldl_cons, List.map_cons, List.sum_cons, add_assoc] using hrest

/-- The concat coordinate bijection preserves the exact objective, with one term per occurrence. -/
theorem dot_concat (layout : ConcatLayout) (parents : Array Nat)
    (a y : Nat → ℝ) (v : Nat → Nat → ℝ)
    (hy : ∀ (parent : Fin layout.lengths.length) (i : Fin (layout.parentShape parent).size),
      y (layout.flatEquiv ⟨parent, i⟩).val = v (parents[parent.val]!) i.val) :
    dot layout.outputShape.size a y =
      ∑ parent : Fin layout.lengths.length,
        dot (layout.parentShape parent).size
          (extendFin (fun i => a (layout.flatEquiv ⟨parent, i⟩).val))
          (v (parents[parent.val]!)) := by
  simp only [dot]
  symm
  rw [← Fintype.sum_sigma']
  apply Fintype.sum_equiv layout.flatEquiv
  intro source
  rcases source with ⟨parent, i⟩
  simp only [extendFin_val, hy]

/-- The engine's concat split and occurrence fold enclose the exact routed objective. -/
theorem represents_concat {dims : Nat → Nat} {v : Nat → Nat → ℝ} {input k : Nat}
    {st : DirectedBackwardState α} {z : ℝ}
    (h : Represents dims v input k st z)
    (layout : ConcatLayout) (parents : Array Nat) (hsize : parents.size = layout.lengths.length)
    (hparent : ∀ p ∈ parents, p < st.coeffs.size)
    (hlive : ∀ p ∈ parents, p ∈ pending input k)
    (hdim : ∀ parent : Fin layout.lengths.length,
      dims (parents[parent.val]!) = (layout.parentShape parent).size)
    {aB : FlatBox α} {a y : Nat → ℝ}
    (ha : RowEncloses aB layout.outputShape.size a)
    (hy : ∀ (parent : Fin layout.lengths.length) (i : Fin (layout.parentShape parent).size),
      y (layout.flatEquiv ⟨parent, i⟩).val = v (parents[parent.val]!) i.val)
    {coefficients : Array (FlatBox α)}
    (hresult : splitDirectedCoeff layout aB = some coefficients) :
    Represents dims v input k
      ((parents.zip coefficients).foldl
        (fun state entry => addDirectedCoeff state entry.1 entry.2) st)
      (z + dot layout.outputShape.size a y) := by
  obtain ⟨adim, alo, ahi⟩ := aB
  obtain ⟨haDim, ha⟩ := ha
  dsimp only at haDim
  subst adim
  simp only [splitDirectedCoeff, ↓reduceDIte, castDimScalar_self,
    Option.some.injEq] at hresult
  subst coefficients
  let coefficient (parent : Fin layout.lengths.length) : FlatBox α :=
    { dim := (layout.parentShape parent).size
      lo := layout.split alo parent
      hi := layout.split ahi parent }
  let exactCoefficient (parent : Fin layout.lengths.length) : Nat → ℝ :=
    extendFin (fun i => a (layout.flatEquiv ⟨parent, i⟩).val)
  have hmem (parent : Fin layout.lengths.length) : parents[parent.val]! ∈ parents := by
    have hp : parent.val < parents.size := by omega
    rw [getElem!_pos (c := parents) (i := parent.val) hp]
    exact Array.getElem_mem hp
  have hb (parent : Fin layout.lengths.length) :
      RowEncloses (coefficient parent) (dims (parents[parent.val]!)) (exactCoefficient parent) := by
    rw [hdim parent]
    exact gather_encloses alo ahi a ⟨rfl, ha⟩ (fun i => layout.flatEquiv ⟨parent, i⟩)
  have hf := represents_fold (List.finRange layout.lengths.length)
    (fun parent => parents[parent.val]!) coefficient exactCoefficient h
    (fun parent => hparent _ (hmem parent)) (fun parent => hlive _ (hmem parent)) hb
  have hsum :
      ((List.finRange layout.lengths.length).map
        (fun parent => dot (dims (parents[parent.val]!))
          (exactCoefficient parent) (v (parents[parent.val]!)))).sum =
        dot layout.outputShape.size a y := by
    rw [List.sum_eq_foldl, List.foldl_map, List.finRange_foldl_add_eq_finset_sum]
    simp only [hdim]
    exact (dot_concat layout parents a y v hy).symm
  rw [hsum] at hf
  change Represents dims v input k
    ((parents.zip ((Array.finRange layout.lengths.length).map coefficient)).foldl
      (fun state entry => addDirectedCoeff state entry.1 entry.2) st)
    (z + dot layout.outputShape.size a y)
  rw [zip_finRange parents _ hsize coefficient, ← Array.foldl_toList, Array.toList_map,
    List.foldl_map]
  simpa only [Array.finRange, Array.toList_ofFn, List.finRange] using hf

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
