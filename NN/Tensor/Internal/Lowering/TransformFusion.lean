/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Lowering.Repeat
public import NN.Tensor.Internal.Laws.Equivalence -- shake: keep

/-!
# Flat-index lowering for fused shape transformations

A sequence of rearrangements, repeats, and arbitrary coordinate pullbacks can
be represented by one output-to-source coordinate map. The compiler also
carries a proof that a flat-index map denotes the same coordinate program.
This module executes that certified flat map directly over native tensor
storage, avoiding multidimensional coordinate construction in the output
loop.

The kernel is independent of transform kind, tensor rank, scalar type, and
chain length. Its correctness theorem reconnects the executable flat-index
program to the ordinary coordinate pullback semantics.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u

namespace Lowering

open Check

/--
Certificates that relate flat and coordinate maps compose at every pair of
intermediate shapes.
-/
theorem flatMap_comp_correct {sourceShape inputShape outputShape : Shape}
    (outerMap : Coord inputShape → Coord sourceShape)
    (outerFlatMap :
      Fin (Shape.size inputShape) → Fin (Shape.size sourceShape))
    (hOuterMap :
      ∀ inputIndex,
        outerFlatMap inputIndex =
          Coord.linearize (outerMap (Coord.unlinearize inputIndex)))
    (innerMap : Coord outputShape → Coord inputShape)
    (innerFlatMap :
      Fin (Shape.size outputShape) → Fin (Shape.size inputShape))
    (hInnerMap :
      ∀ outputIndex,
        innerFlatMap outputIndex =
          Coord.linearize (innerMap (Coord.unlinearize outputIndex))) :
    ∀ outputIndex,
      (outerFlatMap ∘ innerFlatMap) outputIndex =
        Coord.linearize
          ((outerMap ∘ innerMap) (Coord.unlinearize outputIndex)) := by
  intro outputIndex
  simp only [Function.comp_apply]
  rw [hOuterMap, hInnerMap]
  simp only [Coord.unlinearize_linearize]

/--
Transporting an intermediate shape equality preserves a certified flat map.

The coordinate and flat transports are both identity operations after
substituting the shape equality, so this theorem keeps dependent shape
alignment out of generated scalar loops.
-/
theorem flatMap_comp_cast_correct
    {sourceShape previousShape inputShape : Shape}
    (hShape : previousShape = inputShape)
    (previousMap : Coord previousShape → Coord sourceShape)
    (previousFlatMap :
      Fin (Shape.size previousShape) → Fin (Shape.size sourceShape))
    (hPreviousMap :
      ∀ previousIndex,
        previousFlatMap previousIndex =
          Coord.linearize
            (previousMap (Coord.unlinearize previousIndex))) :
    ∀ inputIndex,
      (previousFlatMap ∘
          (finCongr (congrArg Shape.size hShape.symm)).toFun)
          inputIndex =
        Coord.linearize
          ((previousMap ∘
            (Equiv.cast (congrArg Coord hShape.symm)).toFun)
            (Coord.unlinearize inputIndex)) := by
  cases hShape
  intro inputIndex
  simpa using hPreviousMap inputIndex

/--
Recover a coordinate projection from an arbitrary row-major flat-index map.

This is the semantic view of a native flat pullback. It lets later consumers
compose directly with an already compiled source-index program without
reconstructing the checked transformation that produced it.
-/
def coordinateMapOfFlatMap {sourceShape outputShape : Shape}
    (flatMap :
      Fin (Shape.size outputShape) → Fin (Shape.size sourceShape)) :
    Coord outputShape → Coord sourceShape :=
  fun outputCoordinate =>
    Coord.unlinearize (flatMap (Coord.linearize outputCoordinate))

/-- Linearizing `coordinateMapOfFlatMap` recovers the original flat map. -/
theorem flatMap_coordinateMapOfFlatMap
    {sourceShape outputShape : Shape}
    (flatMap :
      Fin (Shape.size outputShape) → Fin (Shape.size sourceShape)) :
    ∀ outputIndex,
      flatMap outputIndex =
        Coord.linearize
          (coordinateMapOfFlatMap flatMap
            (Coord.unlinearize outputIndex)) := by
  intro outputIndex
  simp only [coordinateMapOfFlatMap, Coord.linearize_unlinearize]

/--
Primitive rearrangement is a flat-index pullback along any certified
representation of its output-to-input coordinate map.

This theorem lets elaboration replace the generic checked-plan interpreter
with a partially evaluated index program without relying on definitional
equality between the two implementations.
-/
theorem rearrangeTensor_eq_pullFlat_of_correct
    {α : Type u} [Storage α] (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (flatMap :
      Fin (Shape.size checked.value.output) →
        Fin (Shape.size checked.value.normalized.input))
    (hFlatMap :
      ∀ outputIndex,
        flatMap outputIndex =
          Coord.linearize
            (checked.inputCoordinateOfOutput
              (checked.valid.normalization
                |>.input_axes_subset_output_of_rearrange hKind)
              (Coord.unlinearize outputIndex)))
    (inputTensor : checked.InputTensor α) :
    rearrangeTensor checked hKind inputTensor =
      Rep.pullFlat flatMap inputTensor := by
  rw [rearrangeTensor_correct]
  symm
  apply Rep.pullFlat_eq_pull
  intro outputIndex
  simpa only [Check.CheckedTransform.rearrangeCoordinateEquiv_apply] using
    hFlatMap outputIndex

/--
Primitive rearrangement is exactly one certified flat-index pullback.

This form is used by downstream fusion passes that consume a rearranged tensor
without allocating its intermediate output buffer.
-/
theorem rearrangeTensor_eq_pullFlat
    {α : Type u} [Storage α] (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α) :
    rearrangeTensor checked hKind inputTensor =
      Rep.pullFlat
        (checked.inputFlatIndexOfOutput <|
          checked.valid.normalization.input_axes_subset_output_of_rearrange
            hKind)
        inputTensor := by
  apply rearrangeTensor_eq_pullFlat_of_correct
  exact checked.inputFlatIndexOfOutput_eq
    (checked.valid.normalization.input_axes_subset_output_of_rearrange hKind)

/--
Primitive repeat is a flat-index pullback along any certified representation
of its output-to-input coordinate map.
-/
theorem repeatTensor_eq_pullFlat_of_correct
    {α : Type u} [Storage α] (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .repeat)
    (flatMap :
      Fin (Shape.size checked.value.output) →
        Fin (Shape.size checked.value.normalized.input))
    (hFlatMap :
      ∀ outputIndex,
        flatMap outputIndex =
          Coord.linearize
            (checked.inputCoordinateOfOutput
              (checked.valid.normalization
                |>.input_axes_subset_output_of_repeat hKind)
              (Coord.unlinearize outputIndex)))
    (inputTensor : checked.InputTensor α) :
    repeatTensor checked hKind inputTensor =
      Rep.pullFlat flatMap inputTensor := by
  rw [repeatTensor_correct]
  symm
  exact Rep.pullFlat_eq_pull flatMap _ hFlatMap inputTensor

/--
Primitive repeat is exactly one certified flat-index pullback.

The statement is independent of rank, scalar type, and the number of repeated
axes, so consumer fusion needs no repeat-specific execution path.
-/
theorem repeatTensor_eq_pullFlat
    {α : Type u} [Storage α] (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .repeat)
    (inputTensor : checked.InputTensor α) :
    repeatTensor checked hKind inputTensor =
      Rep.pullFlat
        (checked.inputFlatIndexOfOutput <|
          checked.valid.normalization.input_axes_subset_output_of_repeat hKind)
        inputTensor := by
  apply repeatTensor_eq_pullFlat_of_correct
  exact checked.inputFlatIndexOfOutput_eq
    (checked.valid.normalization.input_axes_subset_output_of_repeat hKind)

/--
An equality with a flat pullback certifies direct reads from the source tensor.

Consumer fusion passes use this pointwise orientation because their generated
loops already hold the source tensor and the logical input index.
-/
theorem source_getFlat_eq_of_eq_pullFlat
    {α : Type u} [Storage α] {sourceShape inputShape : Shape}
    (inputTensor : Rep α inputShape)
    (sourceTensor : Rep α sourceShape)
    (inputFlatMap :
      Fin (Shape.size inputShape) → Fin (Shape.size sourceShape))
    (hTensor :
      inputTensor = Rep.pullFlat inputFlatMap sourceTensor) :
    ∀ inputIndex,
      sourceTensor.getFlat (inputFlatMap inputIndex) =
        inputTensor.getFlat inputIndex := by
  intro inputIndex
  rw [hTensor]
  simp only [Rep.getFlat_pullFlat]

/--
Execute a fused shape-only transformation through one certified flat-index
map and one final output allocation.

`inputMap` records the independent coordinate semantics. `inputFlatMap` is
the executable representation of that map, and `hInputMap` is the certificate
linking the two.
-/
def transformTensorFused {α : Type u} [Storage α]
    {sourceShape : Shape}
    (checked : CheckedTransform)
    (hAxes :
      ∀ ⦃axis⦄,
        axis ∈ checked.value.normalized.inputAxes →
          axis ∈ checked.value.normalized.outputAxes)
    (inputMap :
      Coord checked.value.normalized.input → Coord sourceShape)
    (inputFlatMap :
      Fin (Shape.size checked.value.normalized.input) →
        Fin (Shape.size sourceShape))
    (_hInputMap :
      ∀ inputIndex,
        inputFlatMap inputIndex =
          Coord.linearize (inputMap (Coord.unlinearize inputIndex)))
    (inputTensor : Rep α sourceShape) :
    checked.OutputTensor α :=
  Rep.pullFlat
    (inputFlatMap ∘ checked.inputFlatIndexOfOutput hAxes)
    inputTensor

/--
The native flat-index kernel equals the corresponding composed coordinate
pullback.

This theorem is the compiler correctness boundary for every fused rearrange
or repeat chain.
-/
@[grind =] theorem transformTensorFused_correct
    {α : Type u} [Storage α] {sourceShape : Shape}
    (checked : CheckedTransform)
    (hAxes :
      ∀ ⦃axis⦄,
        axis ∈ checked.value.normalized.inputAxes →
          axis ∈ checked.value.normalized.outputAxes)
    (inputMap :
      Coord checked.value.normalized.input → Coord sourceShape)
    (inputFlatMap :
      Fin (Shape.size checked.value.normalized.input) →
        Fin (Shape.size sourceShape))
    (hInputMap :
      ∀ inputIndex,
        inputFlatMap inputIndex =
          Coord.linearize (inputMap (Coord.unlinearize inputIndex)))
    (inputTensor : Rep α sourceShape) :
    transformTensorFused checked hAxes inputMap inputFlatMap hInputMap
        inputTensor =
      Rep.pull
        (inputMap ∘ checked.inputCoordinateOfOutput hAxes)
        inputTensor := by
  apply Rep.pullFlat_eq_pull
  intro outputIndex
  simp only [Function.comp_apply]
  rw [hInputMap]
  rw [checked.inputFlatIndexOfOutput_eq]
  simp only [Coord.unlinearize_linearize]

/--
Fused flat-index execution of a rearrangement equals the ordinary checked
rearrangement applied after the preceding pullback.
-/
@[grind =] theorem transformTensorFused_rearrange_correct
    {α : Type u} [Storage α] {sourceShape : Shape}
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputMap :
      Coord checked.value.normalized.input → Coord sourceShape)
    (inputFlatMap :
      Fin (Shape.size checked.value.normalized.input) →
        Fin (Shape.size sourceShape))
    (hInputMap :
      ∀ inputIndex,
        inputFlatMap inputIndex =
          Coord.linearize (inputMap (Coord.unlinearize inputIndex)))
    (inputTensor : Rep α sourceShape) :
    transformTensorFused checked
        (checked.valid.normalization.input_axes_subset_output_of_rearrange
          hKind)
        inputMap inputFlatMap hInputMap inputTensor =
      rearrangeTensor checked hKind
        (Rep.pull inputMap inputTensor) := by
  rw [transformTensorFused_correct, rearrangeTensor_correct]
  ext outputCoordinate
  simp only [Rep.pull_apply, Function.comp_apply,
    Semantics.denoteRearrange_apply]

/--
Fused flat-index execution of a repeat equals the ordinary checked repeat
applied after the preceding pullback.
-/
@[grind =] theorem transformTensorFused_repeat_correct
    {α : Type u} [Storage α] {sourceShape : Shape}
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .repeat)
    (inputMap :
      Coord checked.value.normalized.input → Coord sourceShape)
    (inputFlatMap :
      Fin (Shape.size checked.value.normalized.input) →
        Fin (Shape.size sourceShape))
    (hInputMap :
      ∀ inputIndex,
        inputFlatMap inputIndex =
          Coord.linearize (inputMap (Coord.unlinearize inputIndex)))
    (inputTensor : Rep α sourceShape) :
    transformTensorFused checked
        (checked.valid.normalization.input_axes_subset_output_of_repeat hKind)
        inputMap inputFlatMap hInputMap inputTensor =
      repeatTensor checked hKind
        (Rep.pull inputMap inputTensor) := by
  rw [transformTensorFused_correct, repeatTensor_correct]
  ext outputCoordinate
  simp only [Rep.pull_apply, Function.comp_apply,
    Semantics.denoteRepeat_apply]

end Lowering

end TorchLean.Tensor.Internal
