/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import all NN.MLTheory.CROWN.Proofs.DirectedIBPCoordinates
import all NN.MLTheory.CROWN.Proofs.GraphConcatPermutation
import all NN.MLTheory.CROWN.Graph.Engine.Base
public import NN.MLTheory.CROWN.Proofs.DirectedIBPAxisPlan
public import NN.Proofs.Tensor.AxisLinear

/-!
# Adjacent swaps and row-major coordinates

The tensor evaluator's swaps are coordinate bijections. Encoding their coordinates with the
runtime's multiplication-and-addition loop agrees with ordinary tensor linearization.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor
open TorchLean.Tensor.Internal (Coord)
open Proofs.TensorAxis (swapCoordinate permutedCoordinate)

noncomputable section

theorem coord_toList_length (s : Shape) (c : s.Coord) :
    (Shape.Coord.toList s c).length = s.length := by
  induction s with
  | scalar => rfl
  | dim n s ih =>
      simpa only [Shape.Coord.toList, List.length_cons] using (congrArg Nat.succ (ih c.2))

theorem coord_toList_injective (s : Shape) :
    Function.Injective (Shape.Coord.toList s) := by
  intro a b h
  have := congrArg (Shape.Coord.ofList? s) h
  simpa only [Shape.Coord.ofList?_toList, Option.some.injEq] using this

private theorem coordinateIndex_fold_suffix (s : Shape) (c : s.Coord)
    (leading before : List Nat) (hlen : before.length = leading.length) (acc : Nat) :
    (List.range' leading.length s.length).foldl
        (fun index axis =>
          index * (leading ++ s).toArray[axis]! +
            (before ++ Shape.Coord.toList s c).toArray[axis]!) acc =
      Coord.linearizeAux s acc c := by
  induction s generalizing leading before acc with
  | scalar => rfl
  | dim n s ih =>
      rcases c with ⟨head, rest⟩
      simp only [List.length_cons, List.range'_succ, List.foldl_cons,
        Shape.Coord.toList, Coord.linearizeAux]
      have hd : (leading ++ n :: s).toArray[leading.length]! = n := by simp
      have hc :
          (before ++ head.val :: Shape.Coord.toList s rest).toArray[leading.length]! =
            head.val := by rw [← hlen]; simp
      rw [hd, hc]
      simpa only [List.append_assoc, List.singleton_append, List.length_append,
        List.length_singleton] using
        ih rest (leading ++ [n]) (before ++ [head.val]) (by simp [hlen])
          (acc * n + head.val)

private theorem coordinatesFlatIndex_coord (s : Shape) (c : s.Coord) :
    coordinatesFlatIndex s.toArray (Shape.Coord.toList s c).toArray =
      (Coord.linearize c).val := by
  unfold coordinatesFlatIndex
  simp only [Shape.toArray, Shape.toList, List.size_toArray, coord_toList_length,
    Nat.min_self, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size,
    Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, ← List.range_eq_range']
  change (forIn (m := Id) (List.range s.length) 0
    (fun axis index => pure (.yield
      (index * s.toArray[axis]! + (Shape.Coord.toList s c).toArray[axis]!)))).run = _
  rw [forIn_yield_foldl]
  change (List.range s.length).foldl
    (fun (index : Nat) axis =>
      index * s.toArray[axis]! + (Shape.Coord.toList s c).toArray[axis]!) 0 = _
  have h := coordinateIndex_fold_suffix s c [] [] rfl 0
  simpa only [Shape.toArray, Shape.toList, List.length_nil, List.nil_append,
    ← List.range_eq_range',
    Coord.linearizeAux_eq, Nat.zero_mul, Nat.zero_add] using h

theorem swapCoordinate_toList (s : Shape) (depth : Nat)
    (c : (s.swapAdjacentAtDepth depth).Coord) :
    Shape.Coord.toList s (swapCoordinate s depth c) =
      Shape.swapAdjacentAxes (Shape.Coord.toList (s.swapAdjacentAtDepth depth) c) depth := by
  rw [← swapAdjacentAtDepth_eq_swapAdjacentAxes]
  induction depth generalizing s with
  | zero =>
      cases s with
      | scalar => rfl
      | dim a s => cases s <;> rfl
  | succ depth ih =>
      cases s with
      | scalar => rfl
      | dim a s =>
          simpa only [swapCoordinate, Internal.swapAdjacentAxesCoordinate,
            Shape.swapAdjacentAtDepth, Shape.Coord.toList] using
            congrArg (c.1.val :: ·) (ih s c.2)

theorem swapCoordinate_bijective (s : Shape) (depth : Nat) :
    Function.Bijective (swapCoordinate s depth) := by
  induction depth generalizing s with
  | zero =>
      cases s with
      | scalar => exact Function.bijective_id
      | dim a s =>
          cases s with
          | scalar => exact Function.bijective_id
          | dim b s =>
              refine ⟨?_, ?_⟩
              · rintro ⟨x, y, z⟩ ⟨x', y', z'⟩ h
                change (y, x, z) = (y', x', z') at h
                cases h
                rfl
              · rintro ⟨x, y, z⟩
                exact ⟨(y, x, z), rfl⟩
  | succ depth ih =>
      cases s with
      | scalar => exact Function.bijective_id
      | dim a s =>
          exact (Equiv.prodCongr (Equiv.refl (Fin a))
            (Equiv.ofBijective (swapCoordinate s depth) (ih s))).bijective

theorem permutedCoordinate_bijective (s : Shape) (swaps : List Nat) :
    Function.Bijective (permutedCoordinate s swaps) := by
  induction swaps generalizing s with
  | nil => exact Function.bijective_id
  | cons depth swaps ih =>
      exact (swapCoordinate_bijective s depth).comp (ih (s.swapAdjacentAtDepth depth))

theorem permutedCoordinate_toList (s : Shape) (swaps : List Nat)
    (c : (s.applyAdjacentSwaps swaps).Coord) :
    Shape.Coord.toList s (permutedCoordinate s swaps c) =
      swaps.reverse.foldl Shape.swapAdjacentAxes
        (Shape.Coord.toList (s.applyAdjacentSwaps swaps) c) := by
  induction swaps generalizing s with
  | nil => rfl
  | cons depth swaps ih =>
      change ((s.swapAdjacentAtDepth depth).applyAdjacentSwaps swaps).Coord at c
      change Shape.Coord.toList s
        (swapCoordinate s depth (permutedCoordinate (s.swapAdjacentAtDepth depth) swaps c)) =
        (depth :: swaps).reverse.foldl Shape.swapAdjacentAxes
          (Shape.Coord.toList ((s.swapAdjacentAtDepth depth).applyAdjacentSwaps swaps) c)
      rw [swapCoordinate_toList, ih (s.swapAdjacentAtDepth depth) c]
      simp only [List.reverse_cons, List.foldl_append, List.foldl_cons, List.foldl_nil]

/-- Forward transport of an input coordinate through the evaluator's actual swaps. -/
def adjacentCoordinateEquiv (s : Shape) (swaps : List Nat) :
    s.Coord ≃ (s.applyAdjacentSwaps swaps).Coord :=
  (Equiv.ofBijective (permutedCoordinate s swaps)
    (permutedCoordinate_bijective s swaps)).symm

theorem adjacentCoordinateEquiv_toList (s : Shape) (swaps : List Nat) (c : s.Coord) :
    Shape.Coord.toList (s.applyAdjacentSwaps swaps) (adjacentCoordinateEquiv s swaps c) =
      swaps.foldl Shape.swapAdjacentAxes (Shape.Coord.toList s c) := by
  have h : permutedCoordinate s swaps (adjacentCoordinateEquiv s swaps c) = c :=
    (Equiv.ofBijective _ (permutedCoordinate_bijective s swaps)).apply_symm_apply c
  have hlist := congrArg (Shape.Coord.toList s) h
  rw [permutedCoordinate_toList] at hlist
  have hfold := congrArg (fun xs => swaps.foldl Shape.swapAdjacentAxes xs) hlist
  have hinv := swapFold_reverse swaps.reverse
    (Shape.Coord.toList (s.applyAdjacentSwaps swaps) (adjacentCoordinateEquiv s swaps c))
  simp only [List.reverse_reverse] at hinv
  exact hinv.symm.trans hfold

theorem someTensor_swapFold {β : Type} [Storage β] {s : Shape}
    (x : Tensor β s) (swaps : List Nat) :
    swaps.foldl SomeTensor.swapAdjacentAtDepth ⟨s, x⟩ =
      ⟨s.applyAdjacentSwaps swaps, Tensor.permuteByAdjacentSwaps x swaps⟩ := by
  induction swaps generalizing s with
  | nil => rfl
  | cons depth swaps ih => exact ih (Tensor.swapAdjacentAxes x depth)

theorem permuteByAdjacentSwaps_coordinate {β : Type} [Storage β] {s : Shape}
    (x : Tensor β s) (swaps : List Nat) (c : s.Coord) :
    Tensor.permuteByAdjacentSwaps x swaps (adjacentCoordinateEquiv s swaps c) = x c := by
  rw [Proofs.TensorAxis.permuteByAdjacentSwaps_apply]
  exact congrArg (fun coordinate => x coordinate)
    ((Equiv.ofBijective _ (permutedCoordinate_bijective s swaps)).apply_symm_apply c)

theorem adjacentSwaps_size (s : Shape) (swaps : List Nat) :
    (s.applyAdjacentSwaps swaps).size = s.size := by
  simp only [applyAdjacentSwaps_eq_fold, Shape.size_eq_prod]
  exact (swapFold_perm swaps s).prod_eq

theorem range_map_getElem! (xs : List Nat) :
    (List.range xs.length).map (fun i => xs[i]!) = xs := by
  apply List.ext_getElem (by simp)
  intro i hi₁ hi₂
  simp [hi₂]

theorem swapFold_eq_axis_map (xs swaps : List Nat) :
    swaps.foldl Shape.swapAdjacentAxes xs =
      (swaps.foldl Shape.swapAdjacentAxes (List.range xs.length)).map
        (fun i => xs[i]!) := by
  rw [← swapFold_map, range_map_getElem!]

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
