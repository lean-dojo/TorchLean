/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Equivalence.Index

/-!
# Checked rearrangement equivalence

Extensional comparison, executable decisions, counterexamples, and compact
row-major certificates for checked rearrangement plans.
-/

@[expose] public section

namespace TorchLean.Einops

open Shape.Coord

namespace Transform

open Shape (Coord)

/--
Extensional equivalence of two checked rearrangements with equal physical
input and output shapes.

The shape equalities transport the second plan's coordinates into the first
plan's coordinate types. Requiring equality at every output coordinate makes
the definition independent of axis spelling, ellipsis use, and grouping.
-/
def Equivalent
    (first second : Transform)
    (hFirstKind : first.value.normalized.kind = .rearrange)
    (hSecondKind : second.value.normalized.kind = .rearrange)
    (hInputShape :
      first.value.normalized.input =
        second.value.normalized.input)
    (hOutputShape : first.value.output = second.value.output) :
    Prop :=
  ∀ outputCoordinate : Coord first.value.output,
    first.rearrangeEquiv hFirstKind outputCoordinate =
      cast (congrArg Coord hInputShape.symm)
        (second.rearrangeEquiv hSecondKind
          (cast (congrArg Coord hOutputShape) outputCoordinate))

/--
Decide whether two checked rearrangements have equal shapes and the same
coordinate map.

Unlike comparison of parsed syntax, this accepts patterns that use different
axis names or ellipsis spellings but denote the same tensor transformation.
-/
def equivalent?
    (first second : Transform)
    (hFirstKind : first.value.normalized.kind = .rearrange)
    (hSecondKind : second.value.normalized.kind = .rearrange) : Bool :=
  if hInputShape :
    first.value.normalized.input =
        second.value.normalized.input then
    if hOutputShape : first.value.output = second.value.output then
      decide
        (∀ outputIndex : Fin (Shape.size first.value.output),
          first.rearrangeEquiv hFirstKind
              (Coord.unlinearize outputIndex) =
            cast (congrArg Coord hInputShape.symm)
              (second.rearrangeEquiv hSecondKind
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

This implementation materializes every output coordinate before searching.
It is an exhaustive diagnostic, not a constant-memory index comparison.
-/
def counterexample?
    (first second : Transform)
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
          (first.rearrangeEquiv hFirstKind outputCoordinate ≠
            cast (congrArg Coord hInputShape.symm)
              (second.rearrangeEquiv hSecondKind
                (cast (congrArg Coord hOutputShape) outputCoordinate)))
    else
      none
  else
    none

/--
For fixed equal shapes, the Boolean comparison returns `true` exactly when the two plans are
`Equivalent`.
-/
theorem equivalent?_iff_of_shapes
    {first second : Transform}
    {hFirstKind : first.value.normalized.kind = .rearrange}
    {hSecondKind : second.value.normalized.kind = .rearrange}
    (hInputShape :
      first.value.normalized.input =
        second.value.normalized.input)
    (hOutputShape : first.value.output = second.value.output) :
    first.equivalent? second hFirstKind hSecondKind = true ↔
      first.Equivalent second hFirstKind hSecondKind
        hInputShape hOutputShape := by
  simp only [equivalent?, dite_eq_left hInputShape,
    dite_eq_left hOutputShape, decide_eq_true_eq, Equivalent]
  constructor
  · intro hEquivalent outputCoordinate
    simpa using hEquivalent (Coord.linearize outputCoordinate)
  · intro hEquivalent outputIndex
    exact hEquivalent (Coord.unlinearize outputIndex)

/--
The Boolean comparison succeeds exactly when common input and output shapes
and an extensional coordinate proof exist.
-/
theorem equivalent?_iff
    {first second : Transform}
    {hFirstKind : first.value.normalized.kind = .rearrange}
    {hSecondKind : second.value.normalized.kind = .rearrange} :
    first.equivalent? second hFirstKind hSecondKind = true ↔
      ∃ hInputShape :
          first.value.normalized.input =
            second.value.normalized.input,
        ∃ hOutputShape :
            first.value.output = second.value.output,
          first.Equivalent second hFirstKind hSecondKind
            hInputShape hOutputShape := by
  by_cases hInputShape :
      first.value.normalized.input =
        second.value.normalized.input
  · by_cases hOutputShape :
        first.value.output = second.value.output
    · simp only [
        equivalent?_iff_of_shapes hInputShape hOutputShape]
      constructor
      · intro hEquivalent
        exact ⟨hInputShape, hOutputShape, hEquivalent⟩
      · rintro ⟨otherInputShape, otherOutputShape, hEquivalent⟩
        simpa only [Subsingleton.elim otherInputShape hInputShape,
          Subsingleton.elim otherOutputShape hOutputShape] using hEquivalent
    · simp [equivalent?, hInputShape, hOutputShape]
  · simp [equivalent?, hInputShape]

/--
With common shapes, absence of a counterexample is equivalent to extensional
equivalence.
-/
theorem counterexample?_eq_none_iff
    {first second : Transform}
    {hFirstKind : first.value.normalized.kind = .rearrange}
    {hSecondKind : second.value.normalized.kind = .rearrange}
    (hInputShape :
      first.value.normalized.input =
        second.value.normalized.input)
    (hOutputShape : first.value.output = second.value.output) :
    first.counterexample? second hFirstKind hSecondKind = none ↔
      first.Equivalent second hFirstKind hSecondKind
        hInputShape hOutputShape := by
  simp only [counterexample?, dite_eq_left hInputShape,
    dite_eq_left hOutputShape, List.find?_eq_none, Equivalent]
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
theorem counterexample?_sound
    (first second : Transform)
    (hFirstKind : first.value.normalized.kind = .rearrange)
    (hSecondKind : second.value.normalized.kind = .rearrange)
    (outputCoordinate : Coord first.value.output)
    (hCounterexample :
      first.counterexample? second hFirstKind hSecondKind =
        some outputCoordinate) :
    ∃ hInputShape :
        first.value.normalized.input =
          second.value.normalized.input,
      ∃ hOutputShape :
          first.value.output = second.value.output,
        first.rearrangeEquiv hFirstKind outputCoordinate ≠
          cast (congrArg Coord hInputShape.symm)
            (second.rearrangeEquiv hSecondKind
              (cast (congrArg Coord hOutputShape) outputCoordinate)) := by
  unfold counterexample? at hCounterexample
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
theorem Equivalent.symm
    {first second : Transform}
    {hFirstKind : first.value.normalized.kind = .rearrange}
    {hSecondKind : second.value.normalized.kind = .rearrange}
    {hInputShape :
      first.value.normalized.input =
        second.value.normalized.input}
    {hOutputShape : first.value.output = second.value.output}
    (hEquivalent :
      first.Equivalent second hFirstKind hSecondKind
        hInputShape hOutputShape) :
    second.Equivalent first hSecondKind hFirstKind
      hInputShape.symm hOutputShape.symm := by
  intro outputCoordinate
  have hTransported :=
    congrArg (cast (congrArg Coord hInputShape))
      (hEquivalent
        (cast (congrArg Coord hOutputShape.symm) outputCoordinate))
  simpa using hTransported.symm

/-- Extensional rearrange equivalence is transitive. -/
theorem Equivalent.trans
    {first second third : Transform}
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
      first.Equivalent second hFirstKind hSecondKind
        hFirstSecondInput hFirstSecondOutput)
    (hSecondThird :
      second.Equivalent third hSecondKind hThirdKind
        hSecondThirdInput hSecondThirdOutput) :
    first.Equivalent third hFirstKind hThirdKind
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

Unlike `rearrangeEquiv_linearize`, this theorem requires only that
every input axis occurs in the output. It therefore covers both permutations
and repeats that introduce arbitrarily many new axes.
-/
theorem inputOfOutput_linearize
    (checked : Einops.Transform)
    (hAxes :
      ∀ ⦃axis⦄,
        axis ∈ checked.value.normalized.inputAxes →
          axis ∈ checked.value.normalized.outputAxes)
    (outputIndex : Fin (Shape.size checked.value.output)) :
    (Coord.linearize
      (checked.inputOfOutput hAxes <|
        Coord.unlinearize outputIndex)).val =
      Rearrangement.linearIndex checked.value.axisLength
        checked.value.normalized.inputAxes
        checked.value.normalized.outputAxes outputIndex.val := by
  rw [Transform.inputOfOutput,
    outputCoordEquiv_eq_reshape,
    inputCoordEquiv_eq_reshape]
  let elementaryOutputIndex :
      Fin
        (Shape.size
          (checked.value.normalized.outputAxes.map
            checked.value.axisLength)) :=
    finCongr checked.elementary_output_size_eq.symm outputIndex
  simp only [Equiv.trans_apply, Equiv.symm_trans_apply,
    Equiv.symm_symm]
  rw [Coord.linearize_reshapeEquiv_val]
  rw [reshapeEquiv_unlinearize]
  simpa only [elementaryOutputIndex, finCongr_apply_coe] using
    Rearrangement.linearize_select checked.value.axisLength
      checked.value.normalized.inputAxes
      checked.value.normalized.outputAxes hAxes elementaryOutputIndex

/--
Linearize the general checked output-to-input projection at an arbitrary
output coordinate.
-/
@[simp] theorem inputOfOutput_linearize_coord
    (checked : Einops.Transform)
    (hAxes :
      ∀ ⦃axis⦄,
        axis ∈ checked.value.normalized.inputAxes →
          axis ∈ checked.value.normalized.outputAxes)
    (outputCoordinate : Coord checked.value.output) :
    (Coord.linearize
      (checked.inputOfOutput hAxes outputCoordinate)).val =
      Rearrangement.linearIndex checked.value.axisLength
        checked.value.normalized.inputAxes
        checked.value.normalized.outputAxes
        (Coord.linearize outputCoordinate).val := by
  simpa only [Coord.unlinearize_linearize] using
    checked.inputOfOutput_linearize hAxes
      (Coord.linearize outputCoordinate)

/--
Compute the checked source flat index selected by an output flat index.

The value is the compact list-and-arithmetic program used by native lowering.
Its bound follows from the independent coordinate semantics.
-/
def inputFlatIndexOfOutput
    (checked : Einops.Transform)
    (hAxes :
      ∀ ⦃axis⦄,
        axis ∈ checked.value.normalized.inputAxes →
          axis ∈ checked.value.normalized.outputAxes)
    (outputIndex : Fin (Shape.size checked.value.output)) :
    Fin (Shape.size checked.value.normalized.input) :=
  ⟨Rearrangement.linearIndex checked.value.axisLength
      checked.value.normalized.inputAxes
      checked.value.normalized.outputAxes outputIndex.val,
    by
      rw [← checked.inputOfOutput_linearize hAxes outputIndex]
      exact (Coord.linearize
        (checked.inputOfOutput hAxes <|
          Coord.unlinearize outputIndex)).isLt⟩

/--
The executable checked flat-index projection equals linearization of the
independent coordinate projection.
-/
theorem inputFlatIndexOfOutput_eq
    (checked : Einops.Transform)
    (hAxes :
      ∀ ⦃axis⦄,
        axis ∈ checked.value.normalized.inputAxes →
          axis ∈ checked.value.normalized.outputAxes)
    (outputIndex : Fin (Shape.size checked.value.output)) :
    checked.inputFlatIndexOfOutput hAxes outputIndex =
      Coord.linearize
        (checked.inputOfOutput hAxes <|
          Coord.unlinearize outputIndex) := by
  apply Fin.ext
  exact (checked.inputOfOutput_linearize hAxes outputIndex).symm

/--
Linearizing the checked rearrangement coordinate map gives the compact
natural-number calculation `Rearrangement.linearIndex`.

This theorem is the trust boundary for proof automation: the tactic may
reduce the compact calculation, while the kernel checks here that it denotes
the independent coordinate semantics.
-/
theorem rearrangeEquiv_linearize
    (checked : Einops.Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (outputIndex : Fin (Shape.size checked.value.output)) :
    (Coord.linearize
      (checked.rearrangeEquiv hKind <|
        Coord.unlinearize outputIndex)).val =
      Rearrangement.linearIndex checked.value.axisLength
        checked.value.normalized.inputAxes
        checked.value.normalized.outputAxes outputIndex.val := by
  rw [Transform.rearrangeEquiv_apply]
  exact checked.inputOfOutput_linearize
    (checked.valid.normalization.input_subset_of_rearrange hKind)
    outputIndex

/--
Linearize a checked rearrangement at an arbitrary output coordinate.

The coordinate form composes: repeated rewriting turns a chain of checked equivalences into
nested `Rearrangement.linearIndex` calculations.
-/
theorem rearrangeEquiv_linearize_coord
    (checked : Einops.Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (outputCoordinate : Coord checked.value.output) :
    (Coord.linearize
      (checked.rearrangeEquiv hKind outputCoordinate)).val =
      Rearrangement.linearIndex checked.value.axisLength
        checked.value.normalized.inputAxes
        checked.value.normalized.outputAxes
        (Coord.linearize outputCoordinate).val := by
  simpa only [Coord.unlinearize_linearize] using
    checked.rearrangeEquiv_linearize hKind
      (Coord.linearize outputCoordinate)

end Transform

end TorchLean.Einops
