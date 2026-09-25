/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import all NN.MLTheory.CROWN.Graph.Engine.Base
import all NN.MLTheory.CROWN.Proofs.Conv
public import NN.MLTheory.CROWN.Proofs.DirectedIBPTensor

/-!
# Coordinates of the directed pooling iterator

The executable average-pool transfer enumerates flattened windows. Its quotient/remainder loop
recovers the same row-major coordinates as the tensor specification.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor
open TorchLean.Tensor.Internal (Coord)
open scoped BigOperators

private def coordinateStep (dims : Array Nat) (state : Array Nat × Nat) (axis : Nat) :
    Array Nat × Nat :=
  let tailSize := (dims.extract (axis + 1) dims.size).foldl (· * ·) 1
  (state.1.set! axis (state.2 / tailSize), state.2 % tailSize)

private theorem forIn_yield_foldl {ι β : Type} (indices : List ι)
    (step : β → ι → β) (initial : β) :
    forIn (m := Id) indices initial (fun i state => pure (.yield (step state i))) =
      indices.foldl step initial := by
  induction indices generalizing initial with
  | nil => rfl
  | cons i indices ih =>
      rw [List.forIn_cons]
      exact ih (step initial i)

private theorem flatCoordinates_eq_fold (dims : Array Nat) (index : Nat) :
    flatCoordinates dims index =
      ((List.range dims.size).foldl (coordinateStep dims)
        (Array.replicate dims.size 0, index)).1 := by
  unfold flatCoordinates
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size,
    Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, ← List.range_eq_range']
  change
    (forIn (m := Id) (List.range dims.size) (Array.replicate dims.size 0, index)
      (fun axis state => pure (.yield (coordinateStep dims state axis)))).1 = _
  rw [forIn_yield_foldl]

private theorem coordinate_tailSize (leading tail : List Nat) (head : Nat) :
    (((leading ++ head :: tail).toArray).extract (leading.length + 1)
      (leading ++ head :: tail).toArray.size).foldl (· * ·) 1 =
      Tensor.Internal.Shape.size tail := by
  rw [← Array.foldl_toList, Array.toList_extract, List.toList_toArray,
    List.size_toArray, ← List.drop_eq_extract]
  rw [List.drop_append]
  simp only [List.drop_eq_nil_of_le (Nat.le_add_right _ _), List.nil_append,
    Nat.add_sub_cancel_left, List.drop_succ_cons, List.drop_zero]
  exact List.prod_eq_foldl.symm.trans (Tensor.Internal.Shape.size_eq_prod tail).symm

private theorem coordinateStep_cons (leading tail before : List Nat)
    (hlen : before.length = leading.length) {n : Nat}
    (head : Fin n) (rest : Coord tail) :
    coordinateStep (leading ++ n :: tail).toArray
        ((before ++ List.replicate (tail.length + 1) 0).toArray,
          (Coord.linearize (s := n :: tail) (head, rest)).val) leading.length =
      ((before ++ head.val :: List.replicate tail.length 0).toArray,
        (Coord.linearize rest).val) := by
  have htail : 0 < Tensor.Internal.Shape.size tail :=
    (Nat.zero_le _).trans_lt (Coord.linearize rest).isLt
  have hdiv :
      (Coord.linearize (s := n :: tail) (head, rest)).val /
        Tensor.Internal.Shape.size tail = head.val := by
    rw [Coord.linearize_cons_val, Nat.add_mul_div_left _ _ htail]
    simp only [Nat.div_eq_of_lt (Coord.linearize rest).isLt, Nat.zero_add]
  have hmod :
      (Coord.linearize (s := n :: tail) (head, rest)).val %
        Tensor.Internal.Shape.size tail = (Coord.linearize rest).val := by
    rw [Coord.linearize_cons_val, Nat.add_mul_mod_self_left,
      Nat.mod_eq_of_lt (Coord.linearize rest).isLt]
  simp only [coordinateStep, coordinate_tailSize, hdiv, hmod, Prod.mk.injEq, and_true]
  apply Array.toList_inj.mp
  rw [Array.toList_set!, List.toList_toArray, List.toList_toArray,
    List.set_append_right _ _ (by omega), hlen, Nat.sub_self]
  rfl

private theorem coordinate_fold_suffix (tail : List Nat) (rest : Coord tail)
    (leading before : List Nat) (hlen : before.length = leading.length) :
    (List.range' leading.length tail.length).foldl
        (coordinateStep (leading ++ tail).toArray)
        ((before ++ List.replicate tail.length 0).toArray, (Coord.linearize rest).val) =
      ((before ++ Shape.Coord.toList tail rest).toArray, 0) := by
  induction tail generalizing leading before with
  | nil =>
      have hz : (Coord.linearize rest).val = 0 :=
        Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ (Coord.linearize rest).isLt)
      simp [hz, Shape.Coord.toList]
  | cons n tail ih =>
      rcases rest with ⟨head, rest⟩
      simp only [List.length_cons, List.range'_succ, List.foldl_cons]
      rw [coordinateStep_cons leading tail before hlen head rest]
      have hnext := ih rest (leading ++ [n]) (before ++ [head.val]) (by simp [hlen])
      simpa only [List.append_assoc, List.singleton_append, List.length_append,
        List.length_singleton, Shape.Coord.toList] using hnext

/-- The runtime loop recovers every valid tensor coordinate, at arbitrary rank. -/
private theorem flatCoordinates_linearize (s : Shape) (c : s.Coord) :
    (flatCoordinates s.toArray (Coord.linearize c).val).toList = Shape.Coord.toList s c := by
  rw [flatCoordinates_eq_fold]
  have h := coordinate_fold_suffix s c [] [] rfl
  simp only [List.length_nil, List.nil_append, ← List.range_eq_range'] at h
  simpa only [Shape.toArray, Shape.toList, List.size_toArray,
    List.toArray_replicate, List.toList_toArray] using congrArg (fun p => p.1.toList) h

/-- The usual row-major tensor coordinate represented by a flat graph index. -/
def flatCoordEquiv (s : Shape) : Fin s.size ≃ s.Coord :=
  ((Coord.equivFin s).trans (finCongr (Shape.internalSize_eq s))).symm

theorem linearize_flatCoordEquiv (s : Shape) (i : Fin s.size) :
    (Coord.linearize (flatCoordEquiv s i)).val = i.val := by
  simp only [flatCoordEquiv, Equiv.symm_trans_apply, Coord.linearize,
    Equiv.apply_symm_apply, finCongr_symm, finCongr_apply, Fin.val_cast]

theorem getScalar_flatten_coord {β : Type} [Storage β] {s : Shape}
    (x : Tensor β s) (i : Fin s.size) :
    (Tensor.flattenSpec x).getScalar i = x (flatCoordEquiv s i) := by
  rw [getScalar_eq_apply, Tensor.flattenSpec, Tensor.Internal.Rep.reshape_apply_coordEquiv]
  congr 1
  apply Coord.linearize_injective
  apply Fin.ext
  rw [reshapeCoordEquiv_linearize_val, vectorCoordinate_linearize_val,
    linearize_flatCoordEquiv]

private theorem flatCoordinates_eq_coord (s : Shape) (i : Fin s.size) :
    (flatCoordinates s.toArray i.val).toList =
      Shape.Coord.toList s (flatCoordEquiv s i) := by
  simpa only [linearize_flatCoordEquiv] using
    flatCoordinates_linearize s (flatCoordEquiv s i)

/-- The existing spatial-sum theorem, enumerated in the runtime's flat order. -/
private theorem foldlIndices_eq_flat_sum (s : Shape) (f : List Nat → ℝ) :
    Spec.Conv.Internal.foldlIndices s.toList 0 (fun acc indices => acc + f indices) =
      ∑ i : Fin s.size, f (flatCoordinates s.toArray i.val).toList := by
  rw [NN.MLTheory.CROWN.ConvProof.foldlIndices_eq_coord_sum, zero_add]
  simp only [flatCoordinates_eq_coord]
  exact ((flatCoordEquiv s).sum_comp (fun c => f (Shape.Coord.toList s c))).symm

end NN.MLTheory.CROWN.Graph.DirectedBackward
