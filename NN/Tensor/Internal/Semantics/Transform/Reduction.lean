/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Semantics.Transform.Geometry

/-!
# Reduction semantics

Independent unordered and row-major ordered denotations for checked
reductions, together with their aggregation and differential laws.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

open scoped BigOperators

universe u v w

namespace Semantics

open Check

/--
Independent reduction denotation: apply `aggregate` to the multiset of input
values in each certified input-to-output coordinate fiber.

Using a multiset makes permutation invariance part of the function's type and
allows the output scalar type to differ from the input scalar type.
-/
def denoteReduce {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (aggregate : Multiset α → β)
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor β :=
  Rep.reduce aggregate
    (checked.outputCoordinateOfInput <|
      checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
    inputTensor

/--
Independent denotation for a reducer that is defined only on nonempty fibers.

The positive fiber-size premise is shape-level evidence that every aggregate
application receives at least one value.
-/
def denoteReduceNonempty {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (aggregate : (values : Multiset α) → values ≠ 0 → β)
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hNonempty : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor β :=
  Rep.reduceNonempty aggregate
    (checked.outputCoordinateOfInput <|
      checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
    (fun outputCoordinate => by
      rw [checked.reduce_fiber_card hKind outputCoordinate]
      exact hNonempty)
    inputTensor

/--
Values in one reduction fiber, listed in physical row-major order.

The coordinate equivalence supplies the same fiber used by `denoteReduce`,
while `List.ofFn` fixes an order that remains meaningful for IEEE floating
point operations and other nonassociative scalar functions.
-/
noncomputable def orderedReductionValues {α : Type u} [Storage α]
    (checked : CheckedTransform)
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
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) :
    (orderedReductionValues checked hKind inputTensor outputCoordinate).length =
      checked.reductionFiberSize := by
  simp [orderedReductionValues]

/-- A positive checked fiber size makes every ordered fiber list nonempty. -/
theorem orderedReductionValues_ne_nil {α : Type u}
    [Storage α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) :
    orderedReductionValues checked hKind inputTensor outputCoordinate ≠ [] :=
  List.ne_nil_of_length_pos <| by
    rw [orderedReductionValues_length]
    exact hPositive

/--
Ordered reduction denotation with an explicit initial accumulator.

The list fold visits removed-axis coordinates in row-major order. The finalizer
also receives the checked fiber cardinality, which supports operations such as
mean without recounting the list.
-/
noncomputable def denoteOrderedReduce
    {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) (initial : β) (finish : β → Nat → γ)
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor γ :=
  Rep.ofFn fun outputCoordinate =>
    finish
      ((orderedReductionValues checked hKind inputTensor outputCoordinate).foldl
        step initial)
      checked.reductionFiberSize

/--
Ordered reduction denotation initialized from the first fiber value.

This form gives minimum and maximum a precise IEEE behavior without sentinel
values. Positivity of the checked fiber size proves that the first value
exists; the remaining row-major values are folded from left to right.
-/
noncomputable def denoteOrderedReduceNonempty {α : Type u}
    [Storage α]
    (step : α → α → α)
    (checked : CheckedTransform)
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
Evaluate a sum reduction using any explicit reconstruction of the complete
input coordinate from the retained output coordinate and removed-axis
coordinates.

The premises state exactly that the reconstruction remains in the requested
output fiber and recovers every supplied removed-axis tuple. This form is
independent of rank and is convenient when relating einops reductions to
established finite-sum operations.
-/
theorem denoteReduce_sum_apply_reconstructed {α : Type u}
    [Storage α] [AddCommMonoid α]
    (checked : CheckedTransform)
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
        checked.outputCoordinateOfInput
            (checked.valid.normalization.output_axes_subset_input_of_reduce
              hKind)
            (inputCoordinate reducedCoordinate) =
          outputCoordinate)
    (recoversReduction :
      ∀ reducedCoordinate,
        AxisTuple.select checked.reduction_axis_mem_input
            (checked.inputTensorCoordinateEquiv
              (inputCoordinate reducedCoordinate)) =
          reducedCoordinate) :
    denoteReduce Multiset.sum checked hKind inputTensor outputCoordinate =
      ∑ reducedCoordinate, inputTensor (inputCoordinate reducedCoordinate) := by
  simp only [denoteReduce, Rep.reduce, Rep.get_ofFn]
  change
    (∑ inputCoordinate :
        Fiber
          (checked.outputCoordinateOfInput <|
            checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
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
        (checked.outputCoordinateOfInput <|
          checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
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
@[grind =] theorem sum_denoteReduce_sum {α : Type u} [Storage α]
    [AddCommMonoid α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α) :
    (∑ outputCoordinate,
        denoteReduce Multiset.sum checked hKind inputTensor outputCoordinate) =
      ∑ inputCoordinate, inputTensor inputCoordinate :=
  Rep.sum_push
    (checked.outputCoordinateOfInput <|
      checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
    inputTensor

/--
The adjoint of sum reduction under the finite tensor pairing is pullback
along the retained-coordinate projection.

At the tensor level this pullback broadcasts each output cotangent across all
input coordinates in its reduction fiber, which is the algebraic core of the
sum-reduction VJP.
-/
@[grind =] theorem dot_denoteReduce_sum {R : Type u} [Storage R]
    [Semiring R]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor R)
    (outputTensor : checked.OutputTensor R) :
    Rep.dot
        (denoteReduce Multiset.sum checked hKind inputTensor) outputTensor =
      Rep.dot inputTensor
        (Rep.pull
          (checked.outputCoordinateOfInput <|
            checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
          outputTensor) :=
  Rep.dot_push_eq_dot_pull
    (checked.outputCoordinateOfInput <|
      checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
    inputTensor outputTensor

/--
The equal-share weights of minimum reduction sum to one in every checked
nonempty fiber.

Consequently, when `n` input coordinates share the minimum, each receives
weight `1 / n`. This is the library's explicit minimum tie convention.
-/
theorem sum_minEqualShareTieWeight
    {R : Type u} [Storage R] [Field R] [LinearOrder R] [CharZero R]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor R)
    (outputCoordinate : Coord checked.value.output) :
    (∑ inputCoordinate :
        Fiber
          (checked.outputCoordinateOfInput <|
            checked.valid.normalization.output_axes_subset_input_of_reduce
              hKind)
          outputCoordinate,
      Rep.equalShareTieWeight Reduction.min
        (checked.outputCoordinateOfInput <|
          checked.valid.normalization.output_axes_subset_input_of_reduce
            hKind)
        (checked.reduce_fiber_card_pos hKind hPositive)
        inputTensor outputCoordinate inputCoordinate) =
      1 :=
  Rep.sum_equalShareTieWeight Reduction.min Reduction.min_mem
    (checked.outputCoordinateOfInput <|
      checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
    (checked.reduce_fiber_card_pos hKind hPositive)
    inputTensor outputCoordinate

/--
The equal-share weights of maximum reduction sum to one in every checked
nonempty fiber.

Consequently, when `n` input coordinates share the maximum, each receives
weight `1 / n`. This is the library's explicit maximum tie convention.
-/
theorem sum_maxEqualShareTieWeight
    {R : Type u} [Storage R] [Field R] [LinearOrder R] [CharZero R]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor R)
    (outputCoordinate : Coord checked.value.output) :
    (∑ inputCoordinate :
        Fiber
          (checked.outputCoordinateOfInput <|
            checked.valid.normalization.output_axes_subset_input_of_reduce
              hKind)
          outputCoordinate,
      Rep.equalShareTieWeight Reduction.max
        (checked.outputCoordinateOfInput <|
          checked.valid.normalization.output_axes_subset_input_of_reduce
            hKind)
        (checked.reduce_fiber_card_pos hKind hPositive)
        inputTensor outputCoordinate inputCoordinate) =
      1 :=
  Rep.sum_equalShareTieWeight Reduction.max Reduction.max_mem
    (checked.outputCoordinateOfInput <|
      checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
    (checked.reduce_fiber_card_pos hKind hPositive)
    inputTensor outputCoordinate

/--
The equal-share linearization of any checked nonempty reducer is adjoint to
its equal-share VJP.

For `Reduction.min` and `Reduction.max`, the preceding normalization theorems
show that this splits cotangent equally among all tied extrema. At a tie this
is a named symmetric generalized derivative, not a claim of unique classical
differentiability.
-/
theorem dot_equalShareTieReduce
    {R : Type u} [Storage R] [Field R] [DecidableEq R]
    (aggregate : (values : Multiset R) → values ≠ 0 → R)
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor inputTangent : checked.InputTensor R)
    (outputCotangent : checked.OutputTensor R) :
    Rep.dot
        (Rep.equalShareTieDifferential aggregate
          (checked.outputCoordinateOfInput <|
            checked.valid.normalization.output_axes_subset_input_of_reduce
              hKind)
          (checked.reduce_fiber_card_pos hKind hPositive)
          inputTensor inputTangent)
        outputCotangent =
      Rep.dot inputTangent
        (Rep.equalShareTieVjp aggregate
          (checked.outputCoordinateOfInput <|
            checked.valid.normalization.output_axes_subset_input_of_reduce
              hKind)
          (checked.reduce_fiber_card_pos hKind hPositive)
          inputTensor outputCotangent) :=
  Rep.dot_equalShareTieDifferential_eq_dot_equalShareTieVjp
    aggregate
    (checked.outputCoordinateOfInput <|
      checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
    (checked.reduce_fiber_card_pos hKind hPositive)
    inputTensor inputTangent outputCotangent

/--
The VJP of a checked mean reduction broadcasts the output cotangent and
scales it by the reciprocal of the reduction-fiber size.

The denominator is the product of the removed axis lengths certified by the
checked pattern. Positivity rules out an empty mean, while `CharZero` ensures
that this positive natural number remains nonzero in the scalar division
ring.
-/
theorem dot_denoteMeanReduce {R : Type u} [Storage R]
    [DivisionRing R] [CharZero R]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hNonempty : 0 < checked.reductionFiberSize)
    (inputTangent : checked.InputTensor R)
    (outputCotangent : checked.OutputTensor R) :
    Rep.dot
        (denoteReduceNonempty Reduction.mean checked hKind hNonempty
          inputTangent)
        outputCotangent =
      Rep.dot inputTangent
        (Rep.map
          (fun value =>
            (checked.reductionFiberSize : R)⁻¹ * value)
          (Rep.pull
            (checked.outputCoordinateOfInput <|
              checked.valid.normalization.output_axes_subset_input_of_reduce
                hKind)
            outputCotangent)) := by
  let reductionProjection :=
    checked.outputCoordinateOfInput <|
      checked.valid.normalization.output_axes_subset_input_of_reduce hKind
  let fiberNonempty :
      ∀ outputCoordinate,
        0 < Fintype.card (Fiber reductionProjection outputCoordinate) :=
    fun outputCoordinate => by
      rw [checked.reduce_fiber_card hKind outputCoordinate]
      exact hNonempty
  rw [show
    denoteReduceNonempty Reduction.mean checked hKind hNonempty inputTangent =
      Rep.reduceNonempty Reduction.mean reductionProjection
        fiberNonempty inputTangent by
    rfl]
  rw [Rep.dot_reduceNonempty_mean_eq_dot_meanReduceVjp]
  congr 1
  ext inputCoordinate
  simp only [Rep.meanReduceVjp, Rep.get_ofFn, Rep.map_apply,
    Rep.pull_apply]
  rw [checked.reduce_fiber_card hKind (reductionProjection inputCoordinate)]

end Semantics

end TorchLean.Tensor.Internal
