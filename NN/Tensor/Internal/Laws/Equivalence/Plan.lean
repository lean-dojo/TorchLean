/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Laws.Equivalence.Index

/-!
# Checked rearrangement equivalence

Extensional comparison, executable decisions, counterexamples, and compact
row-major certificates for checked rearrangement plans.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

open Rearrangement.Impl

namespace Check.CheckedTransform

/--
Extensional equivalence of two checked rearrangements with equal physical
input and output shapes.

The shape equalities transport the second plan's coordinates into the first
plan's coordinate types. Requiring equality at every output coordinate makes
the definition independent of axis spelling, ellipsis use, and grouping.
-/
def RearrangeEquivalent
    (first second : CheckedTransform)
    (hFirstKind : first.value.normalized.kind = .rearrange)
    (hSecondKind : second.value.normalized.kind = .rearrange)
    (hInputShape :
      first.value.normalized.input =
        second.value.normalized.input)
    (hOutputShape : first.value.output = second.value.output) :
    Prop :=
  ∀ outputCoordinate : Coord first.value.output,
    first.rearrangeCoordinateEquiv hFirstKind outputCoordinate =
      cast (congrArg Coord hInputShape.symm)
        (second.rearrangeCoordinateEquiv hSecondKind
          (cast (congrArg Coord hOutputShape) outputCoordinate))

/--
Decide whether two checked rearrangements have equal shapes and the same
coordinate map.

Unlike comparison of parsed syntax, this accepts patterns that use different
axis names or ellipsis spellings but denote the same tensor transformation.
-/
def rearrangeEquivalent?
    (first second : CheckedTransform)
    (hFirstKind : first.value.normalized.kind = .rearrange)
    (hSecondKind : second.value.normalized.kind = .rearrange) : Bool :=
  if hInputShape :
    first.value.normalized.input =
        second.value.normalized.input then
    if hOutputShape : first.value.output = second.value.output then
      decide
        (∀ outputIndex : Fin (Shape.size first.value.output),
          first.rearrangeCoordinateEquiv hFirstKind
              (Coord.unlinearize outputIndex) =
            cast (congrArg Coord hInputShape.symm)
              (second.rearrangeCoordinateEquiv hSecondKind
                (cast (congrArg Coord hOutputShape)
                  (Coord.unlinearize outputIndex))))
    else
      false
  else
    false

/--
Return an output coordinate where two equal-shaped rearrangements select
different input coordinates.

A shape mismatch returns `none`: the unequal shapes themselves are already a
complete reason for inequivalence. When the physical shapes agree, `none`
means the coordinate maps are equivalent.
-/
def rearrangeCounterexample?
    (first second : CheckedTransform)
    (hFirstKind : first.value.normalized.kind = .rearrange)
    (hSecondKind : second.value.normalized.kind = .rearrange) :
    Option (Coord first.value.output) :=
  if hInputShape :
    first.value.normalized.input =
        second.value.normalized.input then
    if hOutputShape : first.value.output = second.value.output then
      (List.ofFn fun outputIndex : Fin (Shape.size first.value.output) =>
        Coord.unlinearize outputIndex).find? fun outputCoordinate =>
        decide
          (first.rearrangeCoordinateEquiv hFirstKind outputCoordinate ≠
            cast (congrArg Coord hInputShape.symm)
              (second.rearrangeCoordinateEquiv hSecondKind
                (cast (congrArg Coord hOutputShape) outputCoordinate)))
    else
      none
  else
    none

/-- The Boolean comparison is exact once the common shapes are fixed. -/
theorem rearrangeEquivalent?_eq_true_iff_of_shapes
    {first second : CheckedTransform}
    {hFirstKind : first.value.normalized.kind = .rearrange}
    {hSecondKind : second.value.normalized.kind = .rearrange}
    (hInputShape :
      first.value.normalized.input =
        second.value.normalized.input)
    (hOutputShape : first.value.output = second.value.output) :
    first.rearrangeEquivalent? second hFirstKind hSecondKind = true ↔
      first.RearrangeEquivalent second hFirstKind hSecondKind
        hInputShape hOutputShape := by
  simp only [rearrangeEquivalent?, dite_eq_left hInputShape,
    dite_eq_left hOutputShape, decide_eq_true_eq, RearrangeEquivalent]
  constructor
  · intro hEquivalent outputCoordinate
    simpa using hEquivalent (Coord.linearize outputCoordinate)
  · intro hEquivalent outputIndex
    exact hEquivalent (Coord.unlinearize outputIndex)

/--
The Boolean comparison succeeds exactly when common input and output shapes
and an extensional coordinate proof exist.
-/
theorem rearrangeEquivalent?_eq_true_iff
    {first second : CheckedTransform}
    {hFirstKind : first.value.normalized.kind = .rearrange}
    {hSecondKind : second.value.normalized.kind = .rearrange} :
    first.rearrangeEquivalent? second hFirstKind hSecondKind = true ↔
      ∃ hInputShape :
          first.value.normalized.input =
            second.value.normalized.input,
        ∃ hOutputShape :
            first.value.output = second.value.output,
          first.RearrangeEquivalent second hFirstKind hSecondKind
            hInputShape hOutputShape := by
  by_cases hInputShape :
      first.value.normalized.input =
        second.value.normalized.input
  · by_cases hOutputShape :
        first.value.output = second.value.output
    · simp only [
        rearrangeEquivalent?_eq_true_iff_of_shapes hInputShape hOutputShape]
      constructor
      · intro hEquivalent
        exact ⟨hInputShape, hOutputShape, hEquivalent⟩
      · rintro ⟨otherInputShape, otherOutputShape, hEquivalent⟩
        simpa only [Subsingleton.elim otherInputShape hInputShape,
          Subsingleton.elim otherOutputShape hOutputShape] using hEquivalent
    · simp [rearrangeEquivalent?, hInputShape, hOutputShape]
  · simp [rearrangeEquivalent?, hInputShape]

/--
With common shapes, absence of a counterexample is equivalent to extensional
equivalence.
-/
theorem rearrangeCounterexample?_eq_none_iff
    {first second : CheckedTransform}
    {hFirstKind : first.value.normalized.kind = .rearrange}
    {hSecondKind : second.value.normalized.kind = .rearrange}
    (hInputShape :
      first.value.normalized.input =
        second.value.normalized.input)
    (hOutputShape : first.value.output = second.value.output) :
    first.rearrangeCounterexample? second hFirstKind hSecondKind = none ↔
      first.RearrangeEquivalent second hFirstKind hSecondKind
        hInputShape hOutputShape := by
  simp only [rearrangeCounterexample?, dite_eq_left hInputShape,
    dite_eq_left hOutputShape, List.find?_eq_none, RearrangeEquivalent]
  constructor
  · intro hNoCounterexample outputCoordinate
    have hNotDifferent :=
      hNoCounterexample outputCoordinate
        (by
          rw [List.mem_ofFn]
          exact ⟨Coord.linearize outputCoordinate,
            Coord.unlinearize_linearize outputCoordinate⟩)
    by_contra hDifferent
    exact hNotDifferent (Bool.decide_true hDifferent)
  · intro hEquivalent outputCoordinate _ hDifferent
    exact (of_decide_eq_true hDifferent) (hEquivalent outputCoordinate)

/--
Every returned counterexample is sound: at that output coordinate, the two
plans select different input coordinates after shape transport.
-/
theorem rearrangeCounterexample?_eq_some
    (first second : CheckedTransform)
    (hFirstKind : first.value.normalized.kind = .rearrange)
    (hSecondKind : second.value.normalized.kind = .rearrange)
    (outputCoordinate : Coord first.value.output)
    (hCounterexample :
      first.rearrangeCounterexample? second hFirstKind hSecondKind =
        some outputCoordinate) :
    ∃ hInputShape :
        first.value.normalized.input =
          second.value.normalized.input,
      ∃ hOutputShape :
          first.value.output = second.value.output,
        first.rearrangeCoordinateEquiv hFirstKind outputCoordinate ≠
          cast (congrArg Coord hInputShape.symm)
            (second.rearrangeCoordinateEquiv hSecondKind
              (cast (congrArg Coord hOutputShape) outputCoordinate)) := by
  unfold rearrangeCounterexample? at hCounterexample
  split at hCounterexample
  next hInputShape =>
    split at hCounterexample
    next hOutputShape =>
      refine ⟨hInputShape, hOutputShape, ?_⟩
      have hFound := List.find?_some hCounterexample
      exact of_decide_eq_true hFound
    next =>
      simp at hCounterexample
  next =>
    simp at hCounterexample

/-- Extensional rearrange equivalence is symmetric. -/
theorem RearrangeEquivalent.symm
    {first second : CheckedTransform}
    {hFirstKind : first.value.normalized.kind = .rearrange}
    {hSecondKind : second.value.normalized.kind = .rearrange}
    {hInputShape :
      first.value.normalized.input =
        second.value.normalized.input}
    {hOutputShape : first.value.output = second.value.output}
    (hEquivalent :
      first.RearrangeEquivalent second hFirstKind hSecondKind
        hInputShape hOutputShape) :
    second.RearrangeEquivalent first hSecondKind hFirstKind
      hInputShape.symm hOutputShape.symm := by
  intro outputCoordinate
  have hTransported :=
    congrArg (cast (congrArg Coord hInputShape))
      (hEquivalent
        (cast (congrArg Coord hOutputShape.symm) outputCoordinate))
  simpa using hTransported.symm

/-- Extensional rearrange equivalence is transitive. -/
theorem RearrangeEquivalent.trans
    {first second third : CheckedTransform}
    {hFirstKind : first.value.normalized.kind = .rearrange}
    {hSecondKind : second.value.normalized.kind = .rearrange}
    {hThirdKind : third.value.normalized.kind = .rearrange}
    {hFirstSecondInput :
      first.value.normalized.input =
        second.value.normalized.input}
    {hSecondThirdInput :
      second.value.normalized.input =
        third.value.normalized.input}
    {hFirstSecondOutput :
      first.value.output = second.value.output}
    {hSecondThirdOutput :
      second.value.output = third.value.output}
    (hFirstSecond :
      first.RearrangeEquivalent second hFirstKind hSecondKind
        hFirstSecondInput hFirstSecondOutput)
    (hSecondThird :
      second.RearrangeEquivalent third hSecondKind hThirdKind
        hSecondThirdInput hSecondThirdOutput) :
    first.RearrangeEquivalent third hFirstKind hThirdKind
      (hFirstSecondInput.trans hSecondThirdInput)
      (hFirstSecondOutput.trans hSecondThirdOutput) := by
  intro outputCoordinate
  rw [hFirstSecond outputCoordinate,
    hSecondThird
      (cast (congrArg Coord hFirstSecondOutput) outputCoordinate)]
  simp

/--
Linearizing the general checked output-to-input coordinate projection gives
the compact row-major index calculation.

Unlike `rearrangeCoordinateEquiv_linearize`, this theorem requires only that
every input axis occurs in the output. It therefore covers both permutations
and repeats that introduce arbitrarily many new axes.
-/
theorem inputCoordinateOfOutput_linearize
    (checked : Check.CheckedTransform)
    (hAxes :
      ∀ ⦃axis⦄,
        axis ∈ checked.value.normalized.inputAxes →
          axis ∈ checked.value.normalized.outputAxes)
    (outputIndex : Fin (Shape.size checked.value.output)) :
    (Coord.linearize
      (checked.inputCoordinateOfOutput hAxes <|
        Coord.unlinearize outputIndex)).val =
      rearrangeLinearIndex checked.value.axisLength
        checked.value.normalized.inputAxes
        checked.value.normalized.outputAxes outputIndex.val := by
  rw [Check.CheckedTransform.inputCoordinateOfOutput,
    Lowering.outputTensorCoordinateEquiv_eq_reshape,
    Lowering.inputTensorCoordinateEquiv_eq_reshape]
  let elementaryOutputIndex :
      Fin
        (Shape.size
          (checked.value.normalized.outputAxes.map
            checked.value.axisLength)) :=
    finCongr checked.elementary_output_size_eq.symm outputIndex
  simp only [Equiv.trans_apply, Equiv.symm_trans_apply,
    Equiv.symm_symm]
  rw [linearize_reshapeCoordEquiv_val]
  rw [reshapeCoordEquiv_unlinearize]
  simpa only [elementaryOutputIndex, finCongr_apply_coe] using
    linearize_axisTupleSelect checked.value.axisLength
      checked.value.normalized.inputAxes
      checked.value.normalized.outputAxes hAxes elementaryOutputIndex

/--
Linearize the general checked output-to-input projection at an arbitrary
output coordinate.
-/
@[simp] theorem inputCoordinateOfOutput_linearize_coord
    (checked : Check.CheckedTransform)
    (hAxes :
      ∀ ⦃axis⦄,
        axis ∈ checked.value.normalized.inputAxes →
          axis ∈ checked.value.normalized.outputAxes)
    (outputCoordinate : Coord checked.value.output) :
    (Coord.linearize
      (checked.inputCoordinateOfOutput hAxes outputCoordinate)).val =
      rearrangeLinearIndex checked.value.axisLength
        checked.value.normalized.inputAxes
        checked.value.normalized.outputAxes
        (Coord.linearize outputCoordinate).val := by
  simpa only [Coord.unlinearize_linearize] using
    checked.inputCoordinateOfOutput_linearize hAxes
      (Coord.linearize outputCoordinate)

/--
Compute the checked source flat index selected by an output flat index.

The value is the compact list-and-arithmetic program used by native lowering;
its bound follows from the independent coordinate semantics.
-/
def inputFlatIndexOfOutput
    (checked : Check.CheckedTransform)
    (hAxes :
      ∀ ⦃axis⦄,
        axis ∈ checked.value.normalized.inputAxes →
          axis ∈ checked.value.normalized.outputAxes)
    (outputIndex : Fin (Shape.size checked.value.output)) :
    Fin (Shape.size checked.value.normalized.input) :=
  ⟨rearrangeLinearIndex checked.value.axisLength
      checked.value.normalized.inputAxes
      checked.value.normalized.outputAxes outputIndex.val,
    by
      rw [← checked.inputCoordinateOfOutput_linearize hAxes outputIndex]
      exact (Coord.linearize
        (checked.inputCoordinateOfOutput hAxes <|
          Coord.unlinearize outputIndex)).isLt⟩

/--
The executable checked flat-index projection equals linearization of the
independent coordinate projection.
-/
theorem inputFlatIndexOfOutput_eq
    (checked : Check.CheckedTransform)
    (hAxes :
      ∀ ⦃axis⦄,
        axis ∈ checked.value.normalized.inputAxes →
          axis ∈ checked.value.normalized.outputAxes)
    (outputIndex : Fin (Shape.size checked.value.output)) :
    checked.inputFlatIndexOfOutput hAxes outputIndex =
      Coord.linearize
        (checked.inputCoordinateOfOutput hAxes <|
          Coord.unlinearize outputIndex) := by
  apply Fin.ext
  exact (checked.inputCoordinateOfOutput_linearize hAxes outputIndex).symm

/--
Linearizing the checked rearrangement coordinate map gives the compact
natural-number calculation `rearrangeLinearIndex`.

This theorem is the trust boundary for proof automation: the tactic may
reduce the compact calculation, while the kernel checks here that it denotes
the independent coordinate semantics.
-/
theorem rearrangeCoordinateEquiv_linearize
    (checked : Check.CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (outputIndex : Fin (Shape.size checked.value.output)) :
    (Coord.linearize
      (checked.rearrangeCoordinateEquiv hKind <|
        Coord.unlinearize outputIndex)).val =
      rearrangeLinearIndex checked.value.axisLength
        checked.value.normalized.inputAxes
        checked.value.normalized.outputAxes outputIndex.val := by
  rw [Check.CheckedTransform.rearrangeCoordinateEquiv,
    Lowering.outputTensorCoordinateEquiv_eq_reshape,
    Lowering.inputTensorCoordinateEquiv_eq_reshape]
  let elementaryOutputIndex :
      Fin
        (Shape.size
          (checked.value.normalized.outputAxes.map
            checked.value.axisLength)) :=
    finCongr checked.elementary_output_size_eq.symm outputIndex
  simp only [Equiv.trans_apply, Equiv.symm_trans_apply,
    Equiv.symm_symm, AxisTuple.selectEquiv, Equiv.coe_fn_mk]
  rw [linearize_reshapeCoordEquiv_val]
  rw [reshapeCoordEquiv_unlinearize]
  simpa only [elementaryOutputIndex, finCongr_apply_coe] using
      linearize_axisTupleSelect checked.value.axisLength
        checked.value.normalized.inputAxes
        checked.value.normalized.outputAxes
        (checked.valid.normalization.input_axes_subset_output_of_rearrange
          hKind)
        elementaryOutputIndex

/--
Linearize a checked rearrangement at an arbitrary output coordinate.

This coordinate form composes directly: repeated rewriting turns a chain of
checked equivalences into nested `rearrangeLinearIndex` calculations.
-/
theorem rearrangeCoordinateEquiv_linearize_coord
    (checked : Check.CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (outputCoordinate : Coord checked.value.output) :
    (Coord.linearize
      (checked.rearrangeCoordinateEquiv hKind outputCoordinate)).val =
      rearrangeLinearIndex checked.value.axisLength
        checked.value.normalized.inputAxes
        checked.value.normalized.outputAxes
        (Coord.linearize outputCoordinate).val := by
  simpa only [Coord.unlinearize_linearize] using
    checked.rearrangeCoordinateEquiv_linearize hKind
      (Coord.linearize outputCoordinate)

/--
The explicit function projection of a rearrangement equivalence has the same
compact row-major index semantics as its coercion to a function.

Generated fusion terms retain `.toFun` so their executable composition is
unambiguous; this form lets proof automation normalize that representation
without unfolding the checked coordinate map.
-/
theorem rearrangeCoordinateEquiv_toFun_linearize_coord
    (checked : Check.CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (outputCoordinate : Coord checked.value.output) :
    (Coord.linearize
      ((checked.rearrangeCoordinateEquiv hKind).toFun
        outputCoordinate)).val =
      rearrangeLinearIndex checked.value.axisLength
        checked.value.normalized.inputAxes
        checked.value.normalized.outputAxes
        (Coord.linearize outputCoordinate).val :=
  checked.rearrangeCoordinateEquiv_linearize_coord hKind outputCoordinate

/--
Linearizing a checked rearrangement is insensitive to transport of its output
coordinate across an equal physical shape.

This is the composition form used by proof-producing fusion. It absorbs the
dependent cast introduced when independently checked stages share a physical
shape, then exposes the same compact row-major index calculation as the
uncast coordinate theorem.
-/
theorem rearrangeCoordinateEquiv_linearize_cast
    (checked : Check.CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    {shape : Shape}
    (hShape : checked.value.output = shape)
    (outputCoordinate : Coord shape) :
    (Coord.linearize
      (checked.rearrangeCoordinateEquiv hKind
        (cast (congrArg Coord hShape.symm) outputCoordinate))).val =
      rearrangeLinearIndex checked.value.axisLength
        checked.value.normalized.inputAxes
        checked.value.normalized.outputAxes
        (Coord.linearize outputCoordinate).val := by
  rw [rearrangeCoordinateEquiv_linearize_coord, linearize_cast_val hShape]

end Check.CheckedTransform

end TorchLean.Tensor.Internal
