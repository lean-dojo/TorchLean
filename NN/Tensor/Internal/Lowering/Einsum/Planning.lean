/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Lowering.Einsum

/-!
# Algebraic contraction planning

This module isolates the mathematical certificate used when a generated
einsum kernel changes the nesting order of contracted axes. A permutation of
duplicate-free logical axes induces an equivalence of coordinate spaces.
Coordinate sums may be transported across that equivalence only when addition
is commutative, so ordered scalar folds keep their original traversal.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Lowering

universe u v

/--
A permutation of duplicate-free axes induces an equivalence between their
row-major coordinate spaces.

The equivalence changes only the order in which coordinates are presented.
Each named axis retains its original length and bounded coordinate.
-/
def coordinatePermutationEquiv {ι : Type u} [BEq ι] [LawfulBEq ι]
    (length : ι → Nat) {original planned : List ι}
    (hOriginal : original.Nodup) (hPermutation : planned.Perm original) :
    Coord (planned.map length) ≃ Coord (original.map length) :=
  let hPlanned : planned.Nodup :=
    hPermutation.nodup_iff.mpr hOriginal
  let hOriginalPlanned :
      ∀ _axis, _axis ∈ original → _axis ∈ planned :=
    fun _axis hAxis => hPermutation.mem_iff.mpr hAxis
  let hPlannedOriginal :
      ∀ _axis, _axis ∈ planned → _axis ∈ original :=
    fun _axis hAxis => hPermutation.mem_iff.mp hAxis
  (AxisTuple.coordEquiv length planned).trans <|
    (AxisTuple.selectEquiv hOriginal hPlanned
      hOriginalPlanned hPlannedOriginal).trans <|
      (AxisTuple.coordEquiv length original).symm

/--
Changing the nesting order of contracted axes preserves a coordinate sum when
the scalar addition is commutative.

The executable planner uses this theorem as its only algebraic permission to
reorder contraction coordinates. In particular, no corresponding theorem is
available for merely ordered floating-point addition.
-/
theorem coordinateSum_permute {R : Type v} [AddCommMonoid R]
    {ι : Type u} [BEq ι] [LawfulBEq ι]
    (length : ι → Nat) {original planned : List ι}
    (hOriginal : original.Nodup) (hPermutation : planned.Perm original)
    (values : Coord (original.map length) → R) (initial : R) :
    Semantics.coordinateSum (planned.map length)
        (fun coordinate =>
          values
            (coordinatePermutationEquiv length hOriginal hPermutation
              coordinate))
        initial =
      Semantics.coordinateSum (original.map length) values initial := by
  rw [Semantics.coordinateSum_eq_add_sum, Semantics.coordinateSum_eq_add_sum]
  apply congrArg (initial + ·)
  refine Fintype.sum_equiv
    (coordinatePermutationEquiv length hOriginal hPermutation) _ _ ?_
  intro coordinate
  rfl

/--
An executable coordinate map may use a compiler-generated direct projection
instead of evaluating the abstract permutation equivalence.

Pointwise equality with the equivalence is sufficient to retain the same
coordinate-sum certificate.
-/
theorem coordinateSum_permute_of_eq {R : Type v} [AddCommMonoid R]
    {ι : Type u} [BEq ι] [LawfulBEq ι]
    (length : ι → Nat) {original planned : List ι}
    (hOriginal : original.Nodup) (hPermutation : planned.Perm original)
    (coordinateMap : Coord (planned.map length) →
      Coord (original.map length))
    (hCoordinateMap :
      ∀ coordinate,
        coordinateMap coordinate =
          coordinatePermutationEquiv length hOriginal hPermutation
            coordinate)
    (values : Coord (original.map length) → R) (initial : R) :
    Semantics.coordinateSum (planned.map length)
        (fun coordinate => values (coordinateMap coordinate)) initial =
      Semantics.coordinateSum (original.map length) values initial := by
  rw [coordinateSum_congr
    (planned.map length)
    (fun coordinate => values (coordinateMap coordinate))
    (fun coordinate =>
      values
        (coordinatePermutationEquiv length hOriginal hPermutation
          coordinate))
    initial
    (fun coordinate => congrArg values (hCoordinateMap coordinate))]
  exact coordinateSum_permute length hOriginal hPermutation values initial

end TorchLean.Tensor.Internal.Lowering
