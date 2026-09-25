/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import all NN.MLTheory.CROWN.Proofs.DirectedIBPAxisCoordinates
import all NN.MLTheory.CROWN.Proofs.DirectedIBPCoordinates
import all NN.MLTheory.CROWN.Proofs.GraphConcatPermutation
import all NN.MLTheory.CROWN.Graph.Engine.Base
public import NN.MLTheory.CROWN.Proofs.DirectedIBPAxisInverse
public import NN.MLTheory.CROWN.Proofs.DirectedIBPAxisCoordinates
public import NN.MLTheory.CROWN.Proofs.DirectedIBPCoordinates
public import NN.MLTheory.CROWN.Graph.Engine.Base

/-!
# The real evaluator and the backward flat permutation

The forward evaluator lowers an axis array to adjacent swaps. The backward operation inverts the
same array and linearizes its coordinates. Both implementations therefore describe one bijection
of real tensor entries, including tensors with arbitrary rank and zero-length dimensions.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor
open TorchLean.Tensor.Internal (Coord)

noncomputable section

theorem adjacentShape_axisMap (s : Shape) (swaps : List Nat)
    (e : Equiv.Perm (Fin s.length))
    (haxes : swaps.foldl Shape.swapAdjacentAxes (List.range s.length) =
      (axisArray e).toList) :
    s.applyAdjacentSwaps swaps = axisMap e s := by
  rw [applyAdjacentSwaps_eq_fold, swapFold_eq_axis_map, haxes]
  rfl

theorem adjacentCoordinateEquiv_axisMap (s : Shape) (swaps : List Nat)
    (e : Equiv.Perm (Fin s.length))
    (haxes : swaps.foldl Shape.swapAdjacentAxes (List.range s.length) =
      (axisArray e).toList) (c : s.Coord) :
    Shape.Coord.toList (s.applyAdjacentSwaps swaps) (adjacentCoordinateEquiv s swaps c) =
      axisMap e (Shape.Coord.toList s c) := by
  rw [adjacentCoordinateEquiv_toList, swapFold_eq_axis_map, coord_toList_length, haxes]
  rfl

/-- The forward permutation of flat tensor coordinates, obtained from the actual adjacent swaps. -/
def adjacentFlatEquiv (s : Shape) (swaps : List Nat) :
    Fin s.size ≃ Fin (s.applyAdjacentSwaps swaps).size :=
  (flatCoordEquiv s).trans
    ((adjacentCoordinateEquiv s swaps).trans (flatCoordEquiv _).symm)

theorem flatCoordEquiv_symm_val (s : Shape) (c : s.Coord) :
    ((flatCoordEquiv s).symm c).val = (Coord.linearize c).val := by
  simpa only [Equiv.apply_symm_apply] using
    (linearize_flatCoordEquiv s ((flatCoordEquiv s).symm c)).symm

private theorem array_getD_zero (xs : Array Nat) (i : Nat) :
    xs.getD i 0 = xs[i]! := by
  by_cases hi : i < xs.size
  · simp [Array.getD, hi]
  · simp [Array.getD, hi]

/-- The checked backward flat permutation is the inverse-read form of the forward swaps. -/
theorem flatAxisPermutation?_adjacent (s : Shape) (swaps : List Nat)
    (e : Equiv.Perm (Fin s.length))
    (haxes : swaps.foldl Shape.swapAdjacentAxes (List.range s.length) =
      (axisArray e).toList)
    {flat : Fin (s.applyAdjacentSwaps swaps).size → Fin (s.applyAdjacentSwaps swaps).size}
    (hflat : flatAxisPermutation? (s.applyAdjacentSwaps swaps) (axisArray e.symm)
      (s.applyAdjacentSwaps swaps).size = some flat) :
    flat = (finCongr (adjacentSwaps_size s swaps)).trans (adjacentFlatEquiv s swaps) := by
  unfold flatAxisPermutation? at hflat
  obtain ⟨target, htarget, hflat⟩ := Option.bind_eq_some_iff.mp hflat
  have ht : target = s := by
    have h := (permuteShape_properties htarget).2.2
    change target = axisMap e.symm (s.applyAdjacentSwaps swaps) at h
    rw [adjacentShape_axisMap s swaps e haxes, axisMap_symm e s rfl] at h
    exact h
  subst target
  have hguard :
      (((s.applyAdjacentSwaps swaps).size != (s.applyAdjacentSwaps swaps).size) ||
        (s.size != (s.applyAdjacentSwaps swaps).size)) = false := by
    simp [adjacentSwaps_size]
  simp only [hguard, Bool.false_eq_true, ite_false] at hflat
  split at hflat
  · contradiction
  next hn =>
    let : NeZero (s.applyAdjacentSwaps swaps).size := ⟨hn⟩
    rw [inversePerm_axisArray, Equiv.symm_symm] at hflat
    simp only [Except.toOption, Bind.bind, Option.bind, Pure.pure, Option.some.injEq] at hflat
    subst flat
    funext i
    let c := flatCoordEquiv s (Fin.cast (adjacentSwaps_size s swaps) i)
    let outc := adjacentCoordinateEquiv s swaps c
    have hc : (flatCoordinates s.toArray i.val).toList = Shape.Coord.toList s c := by
      exact flatCoordinates_eq_coord s (Fin.cast (adjacentSwaps_size s swaps) i)
    have hcoords :
        (axisArray e).map (fun j => (flatCoordinates s.toArray i.val).getD j 0) =
          (Shape.Coord.toList (s.applyAdjacentSwaps swaps) outc).toArray := by
      apply Array.toList_inj.mp
      rw [Array.toList_map, List.toList_toArray,
        adjacentCoordinateEquiv_axisMap s swaps e haxes c]
      simp only [axisMap, ← hc, Array.getElem!_toList, array_getD_zero]
    apply Fin.ext
    change
      (Fin.ofNat (s.applyAdjacentSwaps swaps).size
        (coordinatesFlatIndex (s.applyAdjacentSwaps swaps).toArray
          ((axisArray e).map (fun j => (flatCoordinates s.toArray i.val).getD j 0)))).val =
        ((flatCoordEquiv (s.applyAdjacentSwaps swaps)).symm outc).val
    rw [hcoords, coordinatesFlatIndex_coord, Fin.val_ofNat, flatCoordEquiv_symm_val]
    exact Nat.mod_eq_of_lt (by
      simpa only [Shape.internalSize_eq] using (Coord.linearize outc).isLt)

theorem realTensor_coord (s : Shape) (f : Nat → ℝ) (c : s.Coord) :
    realTensor s f c = f ((flatCoordEquiv s).symm c).val := by
  have h := getScalar_flatten_coord (realTensor s f) ((flatCoordEquiv s).symm c)
  simpa only [getScalar_flatten_realTensor, Equiv.apply_symm_apply] using h.symm

theorem adjacentFlatEquiv_values (s : Shape) (swaps : List Nat) (f g : Nat → ℝ)
    (hvalues : Tensor.permuteByAdjacentSwaps (realTensor s f) swaps =
      realTensor (s.applyAdjacentSwaps swaps) g) (i : Fin s.size) :
    g (adjacentFlatEquiv s swaps i).val = f i.val := by
  have h := permuteByAdjacentSwaps_coordinate (realTensor s f) swaps (flatCoordEquiv s i)
  rw [hvalues, realTensor_coord, realTensor_coord, Equiv.symm_apply_apply] at h
  exact h

/-- A successful real permutation evaluator supplies exactly the backward coordinate equation. -/
theorem permuteSomeTensor_flatEquation {s t : Shape} {forward : Array Nat}
    {f g : Nat → ℝ}
    (heval : NN.IR.Graph.permuteSomeTensor ⟨s, realTensor s f⟩ forward =
      .ok ⟨t, realTensor t g⟩) :
    s.size = t.size ∧
      ∀ inverse, (NN.IR.OpContracts.inversePerm forward).toOption = some inverse →
        ∀ flat, flatAxisPermutation? t inverse t.size = some flat →
          Function.Bijective flat ∧ ∀ i : Fin t.size, g (flat i).val = f i.val := by
  unfold NN.IR.Graph.permuteSomeTensor at heval
  cases hshape : Shape.permute? s forward.toList with
  | none => simp [hshape, NN.IR.throw_eq_error] at heval
  | some checkedShape =>
      obtain ⟨hlen, hnodup, _⟩ := permuteShape_properties hshape
      cases hplan : NN.IR.Graph.swapDepthsForPerm forward s.rank with
      | error err => simp only [hshape, hplan, Bind.bind, Except.bind, reduceCtorEq] at heval
      | ok swaps =>
          simp only [hshape, hplan, Bind.bind, Except.bind, Pure.pure, Except.pure,
            Except.ok.injEq, ← Array.foldl_toList, someTensor_swapFold] at heval
          have htarget := (SomeTensor.mk.inj heval).1
          subst t
          have hvalues := eq_of_heq (SomeTensor.mk.inj heval).2
          have haxes := swapDepthsForPerm_realizes forward s.rank hlen hnodup swaps hplan
          rw [Shape.rank_eq_length] at haxes
          have hperm : forward.toList.Perm (List.range s.length) := by
            rw [← haxes]
            exact swapFold_perm swaps.toList _
          obtain ⟨e, he⟩ := exists_axisArray hperm
          subst forward
          refine ⟨(adjacentSwaps_size s swaps.toList).symm, ?_⟩
          intro inverse hinverse flat hflat
          rw [inversePerm_axisArray] at hinverse
          obtain rfl := Option.some.inj hinverse
          rw [flatAxisPermutation?_adjacent s swaps.toList e haxes hflat]
          refine ⟨((finCongr (adjacentSwaps_size s swaps.toList)).trans
            (adjacentFlatEquiv s swaps.toList)).bijective, ?_⟩
          intro i
          exact adjacentFlatEquiv_values s swaps.toList f g hvalues
            (Fin.cast (adjacentSwaps_size s swaps.toList) i)

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
