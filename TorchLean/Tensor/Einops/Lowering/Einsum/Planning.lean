/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Lowering.Einsum

/-!
# Algebraic contraction planning

A generated einsum kernel may change the nesting order of its contracted axes,
and the certificate for that step is proved here. A permutation of
duplicate-free logical axes induces an equivalence of coordinate spaces.
Transporting a coordinate sum along it needs an additive commutative monoid, so
ordered scalar folds keep their original traversal.
-/

@[expose] public section

namespace TorchLean.Einops.Lowering

open Shape (Coord)

universe u v

/--
A permutation of duplicate-free axes induces an equivalence between their
row-major coordinate spaces.

The equivalence only reorders the components: each named axis keeps its length
and its bounded coordinate.
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
In an additive commutative monoid, changing the nesting order of contracted
axes preserves a coordinate sum.

The executable planner uses this theorem as its only algebraic permission to
reorder contraction coordinates. Floating-point addition is not associative
and has no such theorem, so float contractions keep their source order.
-/
theorem coordinateSum_permute {R : Type v} [AddCommMonoid R]
    {ι : Type u} [BEq ι] [LawfulBEq ι]
    (length : ι → Nat) {original planned : List ι}
    (hOriginal : original.Nodup) (hPermutation : planned.Perm original)
    (values : Coord (original.map length) → R) (initial : R) :
    Coord.foldl (planned.map length)
        (fun total coordinate =>
          total +
            values
              (coordinatePermutationEquiv length hOriginal hPermutation
                coordinate))
        initial =
      Coord.foldl (original.map length)
        (fun total coordinate => total + values coordinate) initial := by
  rw [Semantics.coordinateSum_eq_add_sum, Semantics.coordinateSum_eq_add_sum]
  apply congrArg (initial + ·)
  refine Fintype.sum_equiv
    (coordinatePermutationEquiv length hOriginal hPermutation) _ _ ?_
  intro coordinate
  rfl

/--
The transport of `coordinateSum_permute` holds for any `coordinateMap` that
agrees pointwise with `coordinatePermutationEquiv`.

The compiler can then generate a direct projection for the coordinate map and
skip evaluating the abstract equivalence.
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
    Coord.foldl (planned.map length)
        (fun total coordinate => total + values (coordinateMap coordinate)) initial =
      Coord.foldl (original.map length)
        (fun total coordinate => total + values coordinate) initial := by
  rw [Semantics.coordinateSum_congr
    (planned.map length)
    (fun coordinate => values (coordinateMap coordinate))
    (fun coordinate =>
      values
        (coordinatePermutationEquiv length hOriginal hPermutation
          coordinate))
    initial
    (fun coordinate => congrArg values (hCoordinateMap coordinate))]
  exact coordinateSum_permute length hOriginal hPermutation values initial

end TorchLean.Einops.Lowering
