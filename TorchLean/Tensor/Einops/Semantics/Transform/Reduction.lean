/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Semantics.Transform.Geometry

/-!
# Reduction semantics

Independent unordered and row-major ordered denotations for checked
reductions, together with their aggregation and differential laws.

The unordered denotation applies a multiset function to each fiber. The
ordered one folds the fiber values in row-major order, which fixes the
evaluation order of IEEE floating-point and other nonassociative operations.
-/

@[expose] public section

namespace TorchLean.Einops

open scoped BigOperators

universe u v w

namespace Semantics

open Tensor (Rep)
open Shape (Coord)

/--
Independent reduction denotation: apply `aggregate` to the multiset of input
values in each fiber of the checked input-to-output coordinate map.

Using a multiset makes permutation invariance part of the function's type and
allows the output scalar type to differ from the input scalar type.
-/
def denoteReduce {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (aggregate : Multiset α → β)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor β :=
  Rep.reduce aggregate
    (checked.outputOfInput <|
      checked.valid.normalization.output_subset_of_reduce hKind)
    inputTensor

/--
Independent denotation for a reducer that is defined only on nonempty fibers.

The positive fiber-size premise is shape-level evidence that every aggregate
application receives at least one value.
-/
def denoteReduceNonempty {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (aggregate : (values : Multiset α) → values ≠ 0 → β)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hNonempty : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor β :=
  Rep.reduceNonempty aggregate
    (checked.outputOfInput <|
      checked.valid.normalization.output_subset_of_reduce hKind)
    (checked.reduce_fiber_card_pos hKind hNonempty)
    inputTensor

/--
Values in one reduction fiber, listed in physical row-major order.

The coordinate equivalence supplies the same fiber used by `denoteReduce`,
while `List.ofFn` fixes an order that remains meaningful for IEEE floating
point operations and other nonassociative scalar functions.
-/
noncomputable def orderedReductionValues {α : Type u} [Storage α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) : List α :=
  List.ofFn fun flatIndex : Fin (Shape.size checked.reductionShape) =>
    inputTensor <|
      ((checked.reductionFiberEquiv hKind outputCoordinate).symm <|
        AxisTuple.coordEquiv checked.value.axisLength checked.reducedAxes <|
          Coord.unlinearize flatIndex).1

/-- The ordered fiber list has the checked reduction-fiber cardinality. -/
@[simp] theorem orderedReductionValues_length {α : Type u}
    [Storage α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) :
    (orderedReductionValues checked hKind inputTensor outputCoordinate).length =
      checked.reductionFiberSize := by
  simp [orderedReductionValues]

/-- A positive checked fiber size makes every ordered fiber list nonempty. -/
theorem orderedReductionValues_ne_nil {α : Type u}
    [Storage α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) :
    orderedReductionValues checked hKind inputTensor outputCoordinate ≠ [] :=
  List.ne_nil_of_length_pos <| by
    rw [orderedReductionValues_length]
    exact hPositive

/--
Ordered reduction denotation that starts from a given initial accumulator.

The list fold visits removed-axis coordinates in row-major order. The finalizer
also receives the checked fiber cardinality, which supports operations such as
mean without recounting the list.
-/
noncomputable def denoteOrderedReduce
    {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) (initial : β) (finish : β → Nat → γ)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor γ :=
  Rep.ofFn fun outputCoordinate =>
    finish
      ((orderedReductionValues checked hKind inputTensor outputCoordinate).foldl
        step initial)
      checked.reductionFiberSize

/--
Ordered reduction denotation initialized from the first fiber value.

Using scalar minimum or maximum as `step` fixes its evaluation order without
inventing sentinel values. Positivity of the checked fiber size proves that
the first value exists, and the remaining row-major values are folded from
left to right. Exceptional-value behavior comes from the chosen scalar
operation.
-/
noncomputable def denoteOrderedReduceNonempty {α : Type u}
    [Storage α]
    (step : α → α → α)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor α :=
  Rep.ofFn fun outputCoordinate =>
    let values :=
      orderedReductionValues checked hKind inputTensor outputCoordinate
    values.tail.foldl step <|
      values.head <|
        orderedReductionValues_ne_nil checked hKind hPositive inputTensor
          outputCoordinate

/--
A sum reduction is the finite sum of `inputTensor` over any explicit reconstruction of complete
input coordinates from the retained output coordinate and removed-axis coordinates.

The premises state exactly that the reconstruction stays in the requested output fiber
(`retainsOutput`) and recovers every supplied removed-axis tuple (`recoversReduction`).
-/
theorem denoteReduce_sum_apply {α : Type u}
    [Storage α] [AddCommMonoid α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output)
    (inputCoordinate :
      AxisTuple checked.value.axisLength
          (checked.value.normalized.inputAxes.filter fun axis =>
            !checked.value.normalized.outputAxes.contains axis) →
        Coord checked.value.normalized.input)
    (retainsOutput :
      ∀ reducedCoordinate,
        checked.outputOfInput
            (checked.valid.normalization.output_subset_of_reduce
              hKind)
            (inputCoordinate reducedCoordinate) =
          outputCoordinate)
    (recoversReduction :
      ∀ reducedCoordinate,
        AxisTuple.select checked.reduction_axis_mem_input
            (checked.inputCoordEquiv
              (inputCoordinate reducedCoordinate)) =
          reducedCoordinate) :
    denoteReduce Multiset.sum checked hKind inputTensor outputCoordinate =
      ∑ reducedCoordinate, inputTensor (inputCoordinate reducedCoordinate) := by
  simp only [denoteReduce, Rep.reduce, Rep.get_ofFn]
  change
    (∑ inputCoordinate :
        Fiber
          (checked.outputOfInput <|
            checked.valid.normalization.output_subset_of_reduce hKind)
          outputCoordinate,
      inputTensor inputCoordinate.1) =
      ∑ reducedCoordinate, inputTensor (inputCoordinate reducedCoordinate)
  refine Fintype.sum_equiv
    (checked.reductionFiberEquiv hKind outputCoordinate) _ _ ?_
  intro fiberCoordinate
  let reducedCoordinate :=
    checked.reductionFiberEquiv hKind outputCoordinate fiberCoordinate
  let candidate :
      Fiber
        (checked.outputOfInput <|
          checked.valid.normalization.output_subset_of_reduce hKind)
        outputCoordinate :=
    ⟨inputCoordinate reducedCoordinate, retainsOutput reducedCoordinate⟩
  have candidateReduction :
      checked.reductionFiberEquiv hKind outputCoordinate candidate =
        reducedCoordinate := by
    rw [checked.reductionFiberEquiv_apply hKind outputCoordinate]
    simpa only [candidate] using recoversReduction reducedCoordinate
  have candidate_eq_fiberCoordinate : candidate = fiberCoordinate := by
    apply
      (checked.reductionFiberEquiv hKind outputCoordinate).injective
    rw [candidateReduction]
  simpa only [candidate, reducedCoordinate] using
    congrArg (fun coordinate => inputTensor coordinate.1)
      candidate_eq_fiberCoordinate.symm

/--
Sum reduction preserves the total additive sum of all tensor entries.

The output fibers partition the finite input-coordinate space, including
when that space is empty.
-/
@[grind =, einops =, einops_simps] theorem sum_denoteReduce_sum {α : Type u} [Storage α]
    [AddCommMonoid α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α) :
    (∑ outputCoordinate,
        denoteReduce Multiset.sum checked hKind inputTensor outputCoordinate) =
      ∑ inputCoordinate, inputTensor inputCoordinate :=
  Rep.sum_push
    (checked.outputOfInput <|
      checked.valid.normalization.output_subset_of_reduce hKind)
    inputTensor

/--
The adjoint of sum reduction under the finite tensor pairing is pullback
along the retained-coordinate projection.

At the tensor level this pullback broadcasts each output cotangent across all
input coordinates in its reduction fiber, which is the algebraic core of the
sum-reduction VJP.
-/
@[grind =, einops =, einops_simps] theorem dot_denoteReduce_sum {R : Type u} [Storage R]
    [NonUnitalNonAssocSemiring R]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor R)
    (outputTensor : checked.OutputTensor R) :
    Rep.dot
        (denoteReduce Multiset.sum checked hKind inputTensor) outputTensor =
      Rep.dot inputTensor
        (Rep.pull
          (checked.outputOfInput <|
            checked.valid.normalization.output_subset_of_reduce hKind)
          outputTensor) :=
  Rep.dot_push_eq_dot_pull
    (checked.outputOfInput <|
      checked.valid.normalization.output_subset_of_reduce hKind)
    inputTensor outputTensor

/--
If `aggregate` returns one of its inputs, the equal-share weights of a checked reduction with
positive fiber size sum to one in every fiber.

`Reduction.min_mem` and `Reduction.max_mem` supply this premise, so `n` coordinates tied at the
minimum or maximum each receive weight `1 / n`.
-/
theorem sum_equalShareTieWeight
    {R : Type u} [Storage R] [DivisionRing R] [CharZero R] [DecidableEq R]
    (aggregate : (values : Multiset R) → values ≠ 0 → R)
    (aggregate_mem :
      ∀ (values : Multiset R) (hValues : values ≠ 0),
        aggregate values hValues ∈ values)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor R)
    (outputCoordinate : Coord checked.value.output) :
    (∑ inputCoordinate :
        Fiber
          (checked.outputOfInput <|
            checked.valid.normalization.output_subset_of_reduce
              hKind)
          outputCoordinate,
      Rep.Tie.weight aggregate
        (checked.outputOfInput <|
          checked.valid.normalization.output_subset_of_reduce
            hKind)
        (checked.reduce_fiber_card_pos hKind hPositive)
        inputTensor outputCoordinate inputCoordinate) =
      1 :=
  Rep.Tie.sum_weight aggregate aggregate_mem
    (checked.outputOfInput <|
      checked.valid.normalization.output_subset_of_reduce hKind)
    (checked.reduce_fiber_card_pos hKind hPositive)
    inputTensor outputCoordinate

/--
The VJP of a checked mean reduction broadcasts the output cotangent and
scales it by the reciprocal of the reduction-fiber size.

The denominator is the product of the removed axis lengths certified by the
checked pattern. Positivity rules out an empty mean, and `CharZero` keeps this
positive natural number nonzero in the scalar division ring.
-/
@[einops_simps] theorem dot_denoteMeanReduce {R : Type u} [Storage R]
    [DivisionRing R] [CharZero R]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hNonempty : 0 < checked.reductionFiberSize)
    (inputTangent : checked.InputTensor R)
    (outputCotangent : checked.OutputTensor R) :
    Rep.dot
        (denoteReduceNonempty Reduction.mean checked hKind hNonempty
          inputTangent)
        outputCotangent =
      Rep.dot inputTangent
        (Tensor.map
          (fun value =>
            (checked.reductionFiberSize : R)⁻¹ * value)
          (Rep.pull
            (checked.outputOfInput <|
              checked.valid.normalization.output_subset_of_reduce
                hKind)
            outputCotangent)) := by
  let reductionProjection :=
    checked.outputOfInput <|
      checked.valid.normalization.output_subset_of_reduce hKind
  let fiberNonempty :
      ∀ outputCoordinate,
        0 < Fintype.card (Fiber reductionProjection outputCoordinate) :=
    checked.reduce_fiber_card_pos hKind hNonempty
  rw [show
    denoteReduceNonempty Reduction.mean checked hKind hNonempty inputTangent =
      Rep.reduceNonempty Reduction.mean reductionProjection
        fiberNonempty inputTangent by
    rfl]
  rw [Rep.Mean.dot_reduceNonempty_eq_dot_vjp]
  congr 1
  ext inputCoordinate
  simp only [Rep.Mean.vjp, Rep.get_ofFn, Tensor.map_apply,
    Rep.pull_apply]
  rw [checked.reduce_fiber_card hKind (reductionProjection inputCoordinate)]

end Semantics

end TorchLean.Einops
